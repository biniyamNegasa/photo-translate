import asyncio
import json
import logging
import re
import secrets
import time
from collections import OrderedDict
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
from app.services.storage import (
    MAX_BATCH_PIXELS,
    ImageLimitError,
    extract_uploaded_images,
    storage,
)

logger = logging.getLogger(__name__)

MAX_SSE_SUBSCRIBERS_PER_JOB = 10
MAX_SSE_SUBSCRIBERS_GLOBAL = 64
SSE_QUEUE_MAXSIZE = 16
MAX_CONCURRENT_RERENDERS = 4
_MAX_DELETED_HISTORY = 4096

_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_POSIX_PATH_RE = re.compile(r"(?:/[A-Za-z0-9._-]+){2,}")
_WIN_PATH_RE = re.compile(r"[A-Za-z]:\\[^\s]+")


class UserFacingError(Exception):
    """Exception whose message is intentionally safe to return to API clients."""


class JobNotFoundError(Exception):
    """Raised when a requested job or item does not exist."""


class ConflictError(Exception):
    """Raised when an operation is invalid for the current item/job state."""


class CapacityError(Exception):
    """Raised when active job, queue, or SSE subscriber capacity limits are reached."""

    def __init__(self, message: str, status_code: int = 503) -> None:
        super().__init__(message)
        self.status_code = status_code


def sanitize_error_message(message: str, default: str = "Failed to process image.") -> str:
    """Strip control characters, URLs, and filesystem paths, and cap length."""
    cleaned = re.sub(r"[\r\n\t\x00-\x1f\x7f]+", " ", str(message or "")).strip()
    cleaned = _URL_RE.sub("[redacted-url]", cleaned)
    cleaned = _WIN_PATH_RE.sub("[redacted-path]", cleaned)
    cleaned = _POSIX_PATH_RE.sub("[redacted-path]", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    if not cleaned:
        return default
    return cleaned[:200]


class JobManager:
    """In-process async batch job worker with bounded SSE streaming, locks, and TTL cleanup."""

    def __init__(self) -> None:
        self._jobs: Dict[str, JobInfo] = {}
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._item_locks: Dict[Tuple[str, str], asyncio.Lock] = {}
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._deleted_jobs: "OrderedDict[str, None]" = OrderedDict()
        self._queue: Optional[asyncio.Queue[str]] = None
        self._rerender_sem: Optional[asyncio.Semaphore] = None
        self._workers: List[asyncio.Task] = []
        self._reaper_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=settings.max_active_jobs)
        self._rerender_sem = asyncio.Semaphore(MAX_CONCURRENT_RERENDERS)
        self._workers = [
            asyncio.create_task(self._worker_loop(idx))
            for idx in range(settings.worker_concurrency)
        ]
        self._reaper_task = asyncio.create_task(self._ttl_reaper_loop())
        logger.info("JobManager started with %d worker(s).", settings.worker_concurrency)

    async def stop(self) -> None:
        # Notify all active SSE subscribers so streaming generators exit cleanly
        for jid in list(self._subscribers.keys()):
            self._close_job_subscribers(jid, status="shutdown")

        tasks_to_cancel: List[asyncio.Task] = []
        for task in list(self._active_tasks.values()):
            task.cancel()
            tasks_to_cancel.append(task)
        for worker in self._workers:
            worker.cancel()
            tasks_to_cancel.append(worker)
        if self._reaper_task:
            self._reaper_task.cancel()
            tasks_to_cancel.append(self._reaper_task)

        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        self._active_tasks.clear()
        self._workers.clear()
        self._item_locks.clear()
        self._reaper_task = None
        self._queue = None
        self._rerender_sem = None
        self._loop = None

    def _get_rerender_sem(self) -> asyncio.Semaphore:
        if self._rerender_sem is None:
            self._rerender_sem = asyncio.Semaphore(MAX_CONCURRENT_RERENDERS)
        return self._rerender_sem

    def _get_item_lock(self, job_id: str, item_id: str) -> asyncio.Lock:
        key = (job_id, item_id)
        lock = self._item_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._item_locks[key] = lock
        return lock

    def _clear_item_locks(self, job_id: str) -> None:
        for key in [k for k in self._item_locks if k[0] == job_id]:
            self._item_locks.pop(key, None)

    def _mark_deleted(self, job_id: str) -> None:
        self._deleted_jobs[job_id] = None
        self._deleted_jobs.move_to_end(job_id)
        while len(self._deleted_jobs) > _MAX_DELETED_HISTORY:
            self._deleted_jobs.popitem(last=False)

    def _is_deleted(self, job_id: str) -> bool:
        return job_id in self._deleted_jobs or job_id not in self._jobs

    def get_job(self, job_id: str) -> Optional[JobInfo]:
        return self._jobs.get(job_id)

    @staticmethod
    def _push_queue_nowait(q: asyncio.Queue, item: Tuple[str, str]) -> None:
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            # Drop oldest stale snapshot so clients always receive the latest state & final "done"
            try:
                q.get_nowait()
                q.put_nowait(item)
            except Exception:
                pass

    def _close_job_subscribers(self, job_id: str, status: str = "closed") -> None:
        subs = self._subscribers.pop(job_id, [])
        if not subs:
            return
        sentinel = ("done", json.dumps({"job_id": job_id, "status": status}))
        for q in list(subs):
            self._push_queue_nowait(q, sentinel)

    def delete_job(self, job_id: str) -> bool:
        if job_id not in self._jobs:
            return False

        self._mark_deleted(job_id)
        self._jobs.pop(job_id, None)
        self._clear_item_locks(job_id)

        task = self._active_tasks.pop(job_id, None)
        if task is not None and not task.done():
            task.cancel()

        self._close_job_subscribers(job_id, status="deleted")
        storage.delete_job(job_id)
        return True

    async def create_job(
        self,
        uploaded_files: List[Tuple[str, bytes]],
        source_lang: str,
        target_lang: str,
        translator: str,
    ) -> JobInfo:
        if len(self._jobs) >= settings.max_active_jobs:
            raise CapacityError(
                "Server is at maximum active job capacity. Please try again later.",
                status_code=503,
            )

        extracted = await asyncio.to_thread(
            extract_uploaded_images,
            uploaded_files,
            settings.max_batch_images,
        )
        if not extracted:
            raise ValueError("No supported images found in upload (PNG, JPG, WebP, BMP, or ZIP).")

        job_id = secrets.token_hex(16)
        now = time.time()

        def _save_originals() -> List[JobItemInfo]:
            saved_items: List[JobItemInfo] = []
            cumulative_pixels = 0
            for idx, (fname, raw_bytes) in enumerate(extracted):
                item_id = f"img_{idx + 1:03d}"
                remaining_pixels = MAX_BATCH_PIXELS - cumulative_pixels
                try:
                    w, h, fmt = storage.save_original_image(
                        job_id,
                        item_id,
                        fname,
                        raw_bytes,
                        remaining_batch_pixels=remaining_pixels,
                    )
                except ImageLimitError:
                    raise
                except Exception as exc:
                    logger.warning("Skipping invalid image '%s': %s", fname, exc)
                    continue

                cumulative_pixels += w * h
                saved_items.append(
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
            return saved_items

        try:
            items = await asyncio.to_thread(_save_originals)
        except Exception:
            await asyncio.to_thread(storage.delete_job, job_id)
            raise

        if not items:
            await asyncio.to_thread(storage.delete_job, job_id)
            raise ValueError("None of the uploaded files could be decoded as valid images.")

        if len(self._jobs) >= settings.max_active_jobs:
            await asyncio.to_thread(storage.delete_job, job_id)
            raise CapacityError(
                "Server is at maximum active job capacity. Please try again later.",
                status_code=503,
            )

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

        if self._queue is None:
            await self.start()
        assert self._queue is not None

        self._jobs[job_id] = job
        try:
            await asyncio.to_thread(storage.save_job_meta, job)
            self._queue.put_nowait(job_id)
        except asyncio.QueueFull as exc:
            self._jobs.pop(job_id, None)
            await asyncio.to_thread(storage.delete_job, job_id)
            raise CapacityError(
                "Job queue is currently full. Please try again later.",
                status_code=503,
            ) from exc
        except Exception:
            self._jobs.pop(job_id, None)
            await asyncio.to_thread(storage.delete_job, job_id)
            raise

        return job

    def _publish(self, job: JobInfo, event: str = "update") -> None:
        subs = self._subscribers.get(job.job_id, [])
        if not subs:
            return
        payload = (
            json.dumps({"job_id": job.job_id, "status": job.status})
            if event == "done"
            else job.model_dump_json()
        )
        for q in list(subs):
            self._push_queue_nowait(q, (event, payload))

    def _publish_threadsafe(self, job: JobInfo, event: str = "update") -> None:
        if self._is_deleted(job.job_id):
            return
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._publish, job, event)

    def reserve_subscriber(self, job_id: str) -> asyncio.Queue:
        if not self.get_job(job_id):
            raise JobNotFoundError("Job not found or expired.")
        per_job = self._subscribers.get(job_id, [])
        if len(per_job) >= MAX_SSE_SUBSCRIBERS_PER_JOB:
            raise CapacityError(
                "Too many active event streams for this job.",
                status_code=429,
            )
        total_subs = sum(len(qs) for qs in self._subscribers.values())
        if total_subs >= MAX_SSE_SUBSCRIBERS_GLOBAL:
            raise CapacityError(
                "Too many active event streams.",
                status_code=429,
            )
        q: asyncio.Queue = asyncio.Queue(maxsize=SSE_QUEUE_MAXSIZE)
        self._subscribers.setdefault(job_id, []).append(q)
        return q

    async def subscribe(
        self,
        job_id: str,
        queue: Optional[asyncio.Queue] = None,
    ) -> AsyncGenerator[str, None]:
        q = queue if queue is not None else self.reserve_subscriber(job_id)
        try:
            job = self.get_job(job_id)
            if not job:
                yield f"event: done\ndata: {json.dumps({'job_id': job_id, 'status': 'deleted'})}\n\n"
                return

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
                    if self._is_deleted(job_id):
                        yield f"event: done\ndata: {json.dumps({'job_id': job_id, 'status': 'deleted'})}\n\n"
                        break
                    yield ": keepalive\n\n"
        finally:
            subs = self._subscribers.get(job_id)
            if subs is not None:
                if q in subs:
                    subs.remove(q)
                if not subs:
                    self._subscribers.pop(job_id, None)

    async def _worker_loop(self, worker_idx: int) -> None:
        queue = self._queue
        assert queue is not None
        while True:
            job_id = await queue.get()
            try:
                job = self._jobs.get(job_id)
                if job is None or self._is_deleted(job_id):
                    continue
                task = asyncio.create_task(self._run_job(job))
                self._active_tasks[job_id] = task
                try:
                    await task
                except asyncio.CancelledError:
                    if self._is_deleted(job_id):
                        logger.info("Worker %d aborted deleted job %s.", worker_idx, job_id)
                    else:
                        task.cancel()
                        raise
                finally:
                    self._active_tasks.pop(job_id, None)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("Worker %d failed on job %s: %s", worker_idx, job_id, exc)
            finally:
                queue.task_done()

    async def _run_job(self, job: JobInfo) -> None:
        if self._is_deleted(job.job_id):
            return

        job.status = JobStatus.PROCESSING
        await asyncio.to_thread(storage.save_job_meta, job)
        self._publish(job, "update")

        sem = asyncio.Semaphore(max(1, settings.item_concurrency))

        async def _run_item(item: JobItemInfo) -> bool:
            async with sem:
                if self._is_deleted(job.job_id):
                    return False
                succeeded = False
                try:
                    await asyncio.to_thread(self._process_single_item, job, item)
                    if item.status in (ItemStatus.COMPLETED, ItemStatus.NO_TEXT):
                        succeeded = True
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception("Failed processing item %s/%s: %s", job.job_id, item.item_id, exc)
                    item.status = ItemStatus.FAILED
                    item.error = (
                        sanitize_error_message(str(exc))
                        if isinstance(exc, UserFacingError)
                        else "Failed to process image."
                    )
                    item.stage_message = "Failed"
                finally:
                    if not self._is_deleted(job.job_id):
                        job.completed_items = sum(
                            1
                            for it in job.items
                            if it.status in (ItemStatus.COMPLETED, ItemStatus.NO_TEXT, ItemStatus.FAILED)
                        )
                        await asyncio.to_thread(storage.save_job_meta, job)
                        self._publish(job, "update")
                return succeeded

        results = await asyncio.gather(*(_run_item(item) for item in job.items))
        if self._is_deleted(job.job_id):
            return

        any_succeeded = any(results)
        job.status = JobStatus.COMPLETED if (any_succeeded or not job.items) else JobStatus.FAILED
        await asyncio.to_thread(storage.save_job_meta, job)
        self._publish(job, "update")
        self._publish(job, "done")

    def _process_single_item(self, job: JobInfo, item: JobItemInfo) -> None:
        if self._is_deleted(job.job_id):
            return

        def on_stage(status: ItemStatus, msg: str) -> None:
            if self._is_deleted(job.job_id):
                return
            item.status = status
            item.stage_message = msg
            self._publish_threadsafe(job, "update")

        orig_rgb = storage.load_original_rgb(job.job_id, item.item_id)
        if self._is_deleted(job.job_id):
            return

        translated_rgb, inpainted_rgb, block_masks, blocks = process_image_pipeline(
            image_rgb=orig_rgb,
            source_lang=job.source_lang,
            target_lang=job.target_lang,
            translator_id=job.translator,
            use_lama=True,
            on_stage=on_stage,
        )

        # Prevent post-delete zombie resurrection on disk
        if self._is_deleted(job.job_id):
            return

        item.blocks = blocks
        storage.save_inpainted_rgb(job.job_id, item.item_id, inpainted_rgb)
        storage.save_block_masks(job.job_id, item.item_id, block_masks)
        storage.save_translated_image(job.job_id, item.item_id, translated_rgb, item.format)

        if self._is_deleted(job.job_id):
            return

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
        and per-block masks. Guarded by per-item lock, global rerender semaphore,
        and completion state check.
        """
        job = self.get_job(job_id)
        if not job:
            raise JobNotFoundError("Job not found.")

        item = next((it for it in job.items if it.item_id == item_id), None)
        if not item:
            raise JobNotFoundError("Item not found.")

        if item.status not in (ItemStatus.COMPLETED, ItemStatus.NO_TEXT):
            raise ConflictError("Item is not ready for re-rendering.")

        item_lock = self._get_item_lock(job_id, item_id)
        async with self._get_rerender_sem():
            async with item_lock:
                if self._is_deleted(job_id):
                    raise JobNotFoundError("Job not found.")
                if item.status not in (ItemStatus.COMPLETED, ItemStatus.NO_TEXT):
                    raise ConflictError("Item is not ready for re-rendering.")

                update_map = {u.id: u for u in updates}
                for blk in item.blocks:
                    upd = update_map.get(blk.id)
                    if upd is not None:
                        if upd.translated_text is not None:
                            blk.translated_text = upd.translated_text
                        if upd.enabled is not None:
                            blk.enabled = upd.enabled

                blocks_snapshot = [blk.model_copy(deep=True) for blk in item.blocks]

                def _do_render() -> None:
                    if self._is_deleted(job_id):
                        return
                    orig_rgb = storage.load_original_rgb(job_id, item_id)
                    inpainted_rgb = storage.load_inpainted_rgb(job_id, item_id)
                    block_masks = storage.load_block_masks(job_id, item_id)
                    rendered = composite_and_render(
                        original_rgb=orig_rgb,
                        inpainted_rgb=inpainted_rgb,
                        block_masks=block_masks,
                        blocks=blocks_snapshot,
                        target_lang=job.target_lang,
                    )
                    if self._is_deleted(job_id):
                        return
                    storage.save_translated_image(job_id, item_id, rendered, item.format)

                await asyncio.to_thread(_do_render)

                if self._is_deleted(job_id):
                    raise JobNotFoundError("Job not found.")

                ts = int(time.time() * 1000)
                item.translated_url = f"/api/jobs/{job_id}/items/{item_id}/translated?v={ts}"
                await asyncio.to_thread(storage.save_job_meta, job)
                self._publish(job, "update")
                return item

    async def _ttl_reaper_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(300)
                await self.evict_expired_jobs()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("TTL reaper error: %s", exc)

    async def evict_expired_jobs(self, now: Optional[float] = None) -> List[str]:
        """Evict expired jobs both from in-memory state and from disk, closing active SSE streams."""
        now = now if now is not None else time.time()
        expired_in_mem = [
            jid for jid, job in list(self._jobs.items()) if job.expires_at < now
        ]
        removed_on_disk = await asyncio.to_thread(storage.cleanup_expired, now)
        evicted = sorted(set(expired_in_mem) | set(removed_on_disk))

        for jid in evicted:
            self._mark_deleted(jid)
            self._jobs.pop(jid, None)
            self._clear_item_locks(jid)
            task = self._active_tasks.pop(jid, None)
            if task is not None and not task.done():
                task.cancel()
            self._close_job_subscribers(jid, status="expired")
            if jid not in removed_on_disk:
                try:
                    await asyncio.to_thread(storage.delete_job, jid)
                except ValueError:
                    pass
        return evicted


job_manager = JobManager()
