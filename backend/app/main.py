import asyncio
import hmac
import logging
import math
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Deque, Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, File, Form, HTTPException, Path as FastAPIPath, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from app.config import settings
from app.languages import LANGUAGE_MAP, LANGUAGES
from app.pipeline.translator import translator_registry
from app.schemas import JobInfo, JobItemInfo, LanguageDTO, RerenderRequest, SystemStatus
from app.services.jobs import (
    CapacityError,
    ConflictError,
    JobNotFoundError,
    job_manager,
    sanitize_error_message,
)
from app.services.storage import (
    build_content_disposition,
    format_to_media_type,
    storage,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

JOB_ID_PATTERN = r"^[a-f0-9]{32}$"
ITEM_ID_PATTERN = r"^img_[0-9]{3,6}$"
UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB
MULTIPART_OVERHEAD_BYTES = 2 * 1024 * 1024  # 2 MiB


class SlidingWindowRateLimiter:
    """Thread-safe in-memory sliding-window rate limiter keyed by (bucket, client)."""

    def __init__(self, max_clients: int = 4096) -> None:
        self._lock = threading.Lock()
        self._buckets: Dict[Tuple[str, str], Deque[float]] = {}
        self._max_clients = max_clients

    def check(
        self,
        bucket: str,
        client_key: str,
        limit: int,
        window_seconds: float = 60.0,
    ) -> Optional[int]:
        now = time.monotonic()
        cutoff = now - window_seconds
        key = (bucket, client_key)
        with self._lock:
            dq = self._buckets.get(key)
            if dq is None:
                if len(self._buckets) >= self._max_clients:
                    # Prune stale keys to prevent unbounded memory growth
                    stale = [k for k, v in self._buckets.items() if not v or v[-1] <= cutoff]
                    for k in stale:
                        self._buckets.pop(k, None)
                    if len(self._buckets) >= self._max_clients:
                        oldest_key = next(iter(self._buckets))
                        self._buckets.pop(oldest_key, None)
                dq = deque()
                self._buckets[key] = dq

            while dq and dq[0] <= cutoff:
                dq.popleft()

            if len(dq) >= limit:
                retry_after = max(1, math.ceil(dq[0] + window_seconds - now))
                return retry_after

            dq.append(now)
            return None

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


rate_limiter = SlidingWindowRateLimiter()


def _client_key(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _enforce_rate_limit(request: Request, bucket: str, limit: int) -> None:
    retry_after = rate_limiter.check(bucket, _client_key(request), limit)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": str(retry_after)},
        )


def verify_api_key(request: Request) -> None:
    """Require Bearer / X-API-Key (or ?api_key= on SSE) when PHOTO_TRANSLATE_API_KEY is configured."""
    expected = settings.get_secret("api_key")
    if not expected:
        return

    candidate = ""
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        candidate = auth_header[7:].strip()
    if not candidate:
        candidate = request.headers.get("x-api-key", "").strip()
    if not candidate and request.url.path.endswith("/events"):
        candidate = request.query_params.get("api_key", "").strip()

    if not candidate or not hmac.compare_digest(
        candidate.encode("utf-8"),
        expected.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized.",
            headers={"WWW-Authenticate": "Bearer"},
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.to_thread(storage.cleanup_expired)
    await job_manager.start()
    yield
    await job_manager.stop()


app = FastAPI(
    title="Photo Translate API",
    version="0.1.0",
    description="Local multilingual image text translation (memes, drawings, comics & photos).",
    lifespan=lifespan,
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url=None,
    openapi_url="/openapi.json" if settings.enable_docs else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Authorization", "X-API-Key", "Cache-Control"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'none'; frame-ancestors 'none'",
    )
    if request.url.path == "/api/jobs" or request.url.path.startswith("/api/jobs/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/status", response_model=SystemStatus)
async def system_status():
    return SystemStatus(
        max_upload_mb=settings.max_upload_mb,
        max_batch_images=settings.max_batch_images,
        translators=translator_registry.list_info(),
    )


@app.get("/api/languages", response_model=List[LanguageDTO])
async def list_languages():
    return [
        LanguageDTO(
            code=l.code,
            name=l.name,
            native_name=l.native_name,
            script=l.script,
            rtl=l.rtl,
        )
        for l in LANGUAGES
    ]


@app.post("/api/jobs", response_model=JobInfo, dependencies=[Depends(verify_api_key)])
async def create_job(
    request: Request,
    files: List[UploadFile] = File(...),
    target_lang: str = Form("en", min_length=1, max_length=32),
    source_lang: str = Form("auto", min_length=1, max_length=32),
    translator: str = Form("google", min_length=1, max_length=32),
):
    try:
        _enforce_rate_limit(request, "create_job", settings.rate_limit_jobs_per_minute)

        if target_lang not in LANGUAGE_MAP:
            raise HTTPException(status_code=400, detail="Unsupported target language.")
        if source_lang != "auto" and source_lang not in LANGUAGE_MAP:
            raise HTTPException(status_code=400, detail="Unsupported source language.")

        translators_by_id = {t.id: t for t in translator_registry.list_info()}
        tr_info = translators_by_id.get(translator)
        if tr_info is None:
            raise HTTPException(status_code=400, detail="Unsupported translator.")
        if not tr_info.available:
            raise HTTPException(status_code=400, detail="Selected translator is not available.")

        if not files:
            raise HTTPException(
                status_code=400,
                detail="Please upload at least one image or ZIP archive.",
            )
        if len(files) > settings.max_batch_images:
            raise HTTPException(
                status_code=413,
                detail=f"Too many files uploaded (maximum {settings.max_batch_images}).",
            )

        max_bytes = settings.max_upload_mb * 1024 * 1024
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > max_bytes + MULTIPART_OVERHEAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds the {settings.max_upload_mb} MB limit.",
                    )
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid Content-Length header.")

        total_bytes = 0
        payloads: List[Tuple[str, bytes]] = []
        for f in files:
            chunks: List[bytes] = []
            while True:
                chunk = await f.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds the {settings.max_upload_mb} MB limit.",
                    )
                chunks.append(chunk)
            payloads.append((f.filename or "image.png", b"".join(chunks)))
    finally:
        for f in files:
            try:
                await f.close()
            except Exception:
                pass

    try:
        return await job_manager.create_job(
            uploaded_files=payloads,
            source_lang=source_lang,
            target_lang=target_lang,
            translator=translator,
        )
    except CapacityError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=sanitize_error_message(str(exc), default="Server is busy. Please try again later."),
            headers={"Retry-After": "5"},
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=sanitize_error_message(str(exc), default="Invalid upload."),
        )


@app.get("/api/jobs/{job_id}", response_model=JobInfo, dependencies=[Depends(verify_api_key)])
async def get_job(
    job_id: str = FastAPIPath(..., pattern=JOB_ID_PATTERN),
):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired.")
    return job


@app.get("/api/jobs/{job_id}/events", dependencies=[Depends(verify_api_key)])
async def stream_job_events(
    job_id: str = FastAPIPath(..., pattern=JOB_ID_PATTERN),
):
    try:
        queue = job_manager.reserve_subscriber(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="Job not found or expired.")
    except CapacityError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=sanitize_error_message(str(exc), default="Too many active event streams."),
            headers={"Retry-After": "5"},
        )

    return StreamingResponse(
        job_manager.subscribe(job_id, queue=queue),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get(
    "/api/jobs/{job_id}/items/{item_id}/original",
    dependencies=[Depends(verify_api_key)],
)
async def get_original_image(
    job_id: str = FastAPIPath(..., pattern=JOB_ID_PATTERN),
    item_id: str = FastAPIPath(..., pattern=ITEM_ID_PATTERN),
):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    item = next((it for it in job.items if it.item_id == item_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found.")
    path = storage.get_original_path(job.job_id, item.item_id, item.format)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Original image missing.")
    return FileResponse(
        path,
        media_type=format_to_media_type(item.format),
        headers={"Cache-Control": "no-store"},
    )


@app.get(
    "/api/jobs/{job_id}/items/{item_id}/translated",
    dependencies=[Depends(verify_api_key)],
)
async def get_translated_image(
    job_id: str = FastAPIPath(..., pattern=JOB_ID_PATTERN),
    item_id: str = FastAPIPath(..., pattern=ITEM_ID_PATTERN),
    download: bool = False,
):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    item = next((it for it in job.items if it.item_id == item_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found.")
    path = storage.get_translated_path(job.job_id, item.item_id, item.format)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Translated image not ready.")
    headers = {"Cache-Control": "no-store"}
    if download:
        headers["Content-Disposition"] = build_content_disposition(f"translated_{item.filename}")
    return FileResponse(path, media_type=format_to_media_type(item.format), headers=headers)


@app.post(
    "/api/jobs/{job_id}/items/{item_id}/rerender",
    response_model=JobItemInfo,
    dependencies=[Depends(verify_api_key)],
)
async def rerender_item(
    request: Request,
    body: RerenderRequest,
    job_id: str = FastAPIPath(..., pattern=JOB_ID_PATTERN),
    item_id: str = FastAPIPath(..., pattern=ITEM_ID_PATTERN),
):
    _enforce_rate_limit(request, "action", settings.rate_limit_actions_per_minute)
    try:
        return await job_manager.rerender_item(job_id, item_id, body.blocks)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="Job or item not found.")
    except ConflictError:
        raise HTTPException(status_code=409, detail="Item is not ready for re-rendering.")
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=sanitize_error_message(str(exc), default="Invalid rerender request."),
        )


@app.get("/api/jobs/{job_id}/download", dependencies=[Depends(verify_api_key)])
async def download_batch_zip(
    request: Request,
    job_id: str = FastAPIPath(..., pattern=JOB_ID_PATTERN),
):
    _enforce_rate_limit(request, "action", settings.rate_limit_actions_per_minute)
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    try:
        zip_path = await asyncio.to_thread(storage.build_batch_zip_file, job)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Job files not found.")
    return FileResponse(
        zip_path,
        media_type="application/zip",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": build_content_disposition(f"photo-translate-{job.job_id}.zip"),
        },
    )


@app.delete("/api/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
async def delete_job(
    job_id: str = FastAPIPath(..., pattern=JOB_ID_PATTERN),
):
    deleted = job_manager.delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"deleted": True}
