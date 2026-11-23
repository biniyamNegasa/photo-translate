import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from app.config import settings
from app.pipeline.renderer import composite_and_render
from app.schemas import (
    BlockUpdateItem,
    ItemStatus,
    JobInfo,
    JobItemInfo,
    JobStatus,
)
from app.services.pipeline import process_image_pipeline
from app.services.storage import extract_uploaded_images, storage

logger = logging.getLogger(__name__)


class JobManager:
    """In-process async batch job worker with SSE live progress streaming and TTL cleanup."""

    def __init__(self) -> None:
        self._jobs: Dict[str, JobInfo] = {}
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._queue: Optional[asyncio.Queue[str]] = None
        self._workers: List[asyncio.Task] = []
        self._reaper_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._workers = [
            asyncio.create_task(self._worker_loop(idx))
            for idx in range(settings.worker_concurrency)
        ]
        self._reaper_task = asyncio.create_task(self._ttl_reaper_loop())
        logger.info("JobManager started with %d worker(s).", settings.worker_concurrency)

    async def stop(self) -> None:
        for t in self._workers:
            t.cancel()
        if self._reaper_task:
            self._reaper_task.cancel()
        self._workers.clear()
        self._reaper_task = None
        self._queue = None

    def get_job(self, job_id: str) -> Optional[JobInfo]:
        return self._jobs.get(job_id)

    def delete_job(self, job_id: str) -> bool:
        existed = job_id in self._jobs
        self._jobs.pop(job_id, None)
        self._subscribers.pop(job_id, None)
        storage.delete_job(job_id)
        return existed

    async def create_job(
        self,
        uploaded_files: List[Tuple[str, bytes]],
        source_lang: str,
        target_lang: str,
        translator: str,
    ) -> JobInfo:
        extracted = extract_uploaded_images(uploaded_files, max_images=settings.max_batch_images)
        if not extracted:
            raise ValueError("No supported images found in upload (PNG, JPG, WebP, or ZIP).")

        job_id = uuid.uuid4().hex[:12]
        now = time.time()
        items: List[JobItemInfo] = []

        for idx, (fname, raw_bytes) in enumerate(extracted):
            item_id = f"img_{idx + 1:03d}"
            try:
                w, h, fmt = storage.save_original_image(job_id, item_id, fname, raw_bytes)
            except Exception as exc:
                logger.warning("Skipping invalid image '%s': %s", fname, exc)
                continue

            items.append(
                JobItemInfo(
                    item_id=item_id,
                    filename=fname,
                    width=w,
                    height=h,
                    format=fmt,
                    status=ItemStatus.QUEUED,
                    stage_message="Queued",
                    original_url=f"/api/jobs/{job_id}/items/{item_id}/original",
                    translated_url=f"/api/jobs/{job_id}/items/{item_id}/translated",
                )
            )

        if not items:
            storage.delete_job(job_id)
            raise ValueError("None of the uploaded files could be decoded as valid images.")

        job = JobInfo(
            job_id=job_id,
            status=JobStatus.QUEUED,
            source_lang=source_lang,
            target_lang=target_lang,
            translator=translator,
            created_at=now,
            expires_at=now + settings.job_ttl_seconds,
            total_items=len(items),
            completed_items=0,
            items=items,
        )
        self._jobs[job_id] = job
        storage.save_job_meta(job)
        if self._queue is None:
            await self.start()
        assert self._queue is not None
        await self._queue.put(job_id)
        return job

    def _publish(self, job: JobInfo, event: str = "update") -> None:
        subs = self._subscribers.get(job.job_id, [])
        if not subs:
            return
        payload = job.model_dump_json()
        for q in list(subs):
            try:
                q.put_nowait((event, payload))
            except asyncio.QueueFull:
                # Drop oldest stale snapshot so clients always receive the latest state & final "done"
                try:
                    q.get_nowait()
                    q.put_nowait((event, payload))
                except Exception:
                    pass

    def _publish_threadsafe(self, job: JobInfo, event: str = "update") -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._publish, job, event)

    async def subscribe(self, job_id: str) -> AsyncGenerator[str, None]:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subscribers.setdefault(job_id, []).append(q)
        try:
            job = self.get_job(job_id)
            if job:
                yield f"event: update\ndata: {job.model_dump_json()}\n\n"
                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    yield f"event: done\ndata: {json.dumps({'job_id': job_id, 'status': job.status})}\n\n"
                    return

            while True:
                try:
                    event, payload = await asyncio.wait_for(q.get(), timeout=20.0)
                    yield f"event: {event}\ndata: {payload}\n\n"
                    if event == "done":
                        break
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            subs = self._subscribers.get(job_id, [])
            if q in subs:
                subs.remove(q)

    async def _worker_loop(self, worker_idx: int) -> None:
        queue = self._queue
        assert queue is not None
        while True:
            job_id = await queue.get()
            try:
                job = self._jobs.get(job_id)
                if job is None:
                    continue
                await self._run_job(job)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("Worker %d failed on job %s: %s", worker_idx, job_id, exc)
            finally:
                queue.task_done()

    async def _run_job(self, job: JobInfo) -> None:
        job.status = JobStatus.PROCESSING
        storage.save_job_meta(job)
        self._publish(job, "update")

        sem = asyncio.Semaphore(max(1, settings.item_concurrency))

        async def _run_item(item: JobItemInfo) -> bool:
            async with sem:
                succeeded = False
                try:
                    await asyncio.to_thread(self._process_single_item, job, item)
                    if item.status in (ItemStatus.COMPLETED, ItemStatus.NO_TEXT):
                        succeeded = True
                except Exception as exc:
                    logger.exception("Failed processing item %s/%s: %s", job.job_id, item.item_id, exc)
                    item.status = ItemStatus.FAILED
                    item.error = str(exc)
                    item.stage_message = "Failed"
                finally:
                    job.completed_items = sum(
                        1
                        for it in job.items
                        if it.status in (ItemStatus.COMPLETED, ItemStatus.NO_TEXT, ItemStatus.FAILED)
                    )
                    storage.save_job_meta(job)
                    self._publish(job, "update")
                return succeeded

        results = await asyncio.gather(*(_run_item(item) for item in job.items))
        any_succeeded = any(results)

        job.status = JobStatus.COMPLETED if (any_succeeded or not job.items) else JobStatus.FAILED
        storage.save_job_meta(job)
        self._publish(job, "update")
        self._publish(job, "done")

    def _process_single_item(self, job: JobInfo, item: JobItemInfo) -> None:
        def on_stage(status: ItemStatus, msg: str) -> None:
            item.status = status
            item.stage_message = msg
            self._publish_threadsafe(job, "update")

        orig_rgb = storage.load_original_rgb(job.job_id, item.item_id)
        translated_rgb, inpainted_rgb, block_masks, blocks = process_image_pipeline(
            image_rgb=orig_rgb,
            source_lang=job.source_lang,
            target_lang=job.target_lang,
            translator_id=job.translator,
            use_lama=True,
            on_stage=on_stage,
        )

        item.blocks = blocks
        storage.save_inpainted_rgb(job.job_id, item.item_id, inpainted_rgb)
        storage.save_block_masks(job.job_id, item.item_id, block_masks)
        storage.save_translated_image(job.job_id, item.item_id, translated_rgb, item.format)

        # Bust browser image cache when updated
        ts = int(time.time() * 1000)
        item.translated_url = f"/api/jobs/{job.job_id}/items/{item.item_id}/translated?v={ts}"

        if not blocks:
            item.status = ItemStatus.NO_TEXT
            item.stage_message = "No text detected (original preserved)"
        else:
            item.status = ItemStatus.COMPLETED
            item.stage_message = f"Translated {len(blocks)} block(s)"

    async def rerender_item(
        self,
        job_id: str,
        item_id: str,
        updates: List[BlockUpdateItem],
    ) -> JobItemInfo:
        """
        Instant (<200ms) re-render using cached original image, inpainted canvas,
        and per-block masks. Toggling a block off restores original pixels bit-for-bit.
        """
        job = self.get_job(job_id)
        if not job:
            raise KeyError(f"Job '{job_id}' not found.")

        item = next((it for it in job.items if it.item_id == item_id), None)
        if not item:
            raise KeyError(f"Item '{item_id}' not found.")

        update_map = {u.id: u for u in updates}
        for blk in item.blocks:
            upd = update_map.get(blk.id)
            if upd is not None:
                if upd.translated_text is not None:
                    blk.translated_text = upd.translated_text
                if upd.enabled is not None:
                    blk.enabled = upd.enabled

        def _do_render() -> None:
            orig_rgb = storage.load_original_rgb(job_id, item_id)
            inpainted_rgb = storage.load_inpainted_rgb(job_id, item_id)
            block_masks = storage.load_block_masks(job_id, item_id)
            rendered = composite_and_render(
                original_rgb=orig_rgb,
                inpainted_rgb=inpainted_rgb,
                block_masks=block_masks,
                blocks=item.blocks,
                target_lang=job.target_lang,
            )
            storage.save_translated_image(job_id, item_id, rendered, item.format)

        await asyncio.to_thread(_do_render)

        ts = int(time.time() * 1000)
        item.translated_url = f"/api/jobs/{job_id}/items/{item_id}/translated?v={ts}"
        storage.save_job_meta(job)
        self._publish(job, "update")
        return item

    async def _ttl_reaper_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(300)
                removed = storage.cleanup_expired()
                for jid in removed:
                    self._jobs.pop(jid, None)
                    self._subscribers.pop(jid, None)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("TTL reaper error: %s", exc)


job_manager = JobManager()
