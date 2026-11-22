import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse

from app.config import settings
from app.languages import LANGUAGES
from app.pipeline.comic_detector import comic_detector
from app.pipeline.fonts import is_raqm_available
from app.pipeline.inpainter import get_onnx_providers, inpainter
from app.pipeline.manga_ocr import manga_ocr
from app.pipeline.translator import translator_registry
from app.schemas import JobInfo, JobItemInfo, LanguageDTO, RerenderRequest, SystemStatus
from app.services.jobs import job_manager
from app.services.storage import format_to_media_type, storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    storage.cleanup_expired()
    await job_manager.start()
    yield
    await job_manager.stop()


app = FastAPI(
    title="Photo Translate API",
    version="0.1.0",
    description="Local multilingual image text translation (memes, drawings, comics & photos).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/status", response_model=SystemStatus)
async def system_status():
    providers = get_onnx_providers()
    device = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
    return SystemStatus(
        lama_downloaded=inpainter.is_downloaded(),
        comic_detector_downloaded=comic_detector.is_downloaded(),
        manga_ocr_downloaded=manga_ocr.is_downloaded(),
        raqm_available=is_raqm_available(),
        device=device,
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


@app.post("/api/jobs", response_model=JobInfo)
async def create_job(
    files: List[UploadFile] = File(...),
    target_lang: str = Form("en"),
    source_lang: str = Form("auto"),
    translator: str = Form("google"),
):
    if not files:
        raise HTTPException(status_code=400, detail="Please upload at least one image or ZIP archive.")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    total_bytes = 0
    payloads = []
    for f in files:
        data = await f.read()
        total_bytes += len(data)
        if total_bytes > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds the {settings.max_upload_mb} MB limit.",
            )
        payloads.append((f.filename or "image.png", data))

    try:
        job = await job_manager.create_job(
            uploaded_files=payloads,
            source_lang=source_lang,
            target_lang=target_lang,
            translator=translator,
        )
        return job
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/jobs/{job_id}", response_model=JobInfo)
async def get_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired.")
    return job


@app.get("/api/jobs/{job_id}/events")
async def stream_job_events(job_id: str):
    if not job_manager.get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found or expired.")
    return StreamingResponse(
        job_manager.subscribe(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/jobs/{job_id}/items/{item_id}/original")
async def get_original_image(job_id: str, item_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    item = next((it for it in job.items if it.item_id == item_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found.")
    path = storage.get_original_path(job_id, item_id, item.format)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Original image missing.")
    return FileResponse(path, media_type=format_to_media_type(item.format))


@app.get("/api/jobs/{job_id}/items/{item_id}/translated")
async def get_translated_image(job_id: str, item_id: str, download: bool = False):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    item = next((it for it in job.items if it.item_id == item_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found.")
    path = storage.get_translated_path(job_id, item_id, item.format)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Translated image not ready.")
    headers = {"Cache-Control": "no-store"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="translated_{item.filename}"'
    return FileResponse(path, media_type=format_to_media_type(item.format), headers=headers)


@app.post("/api/jobs/{job_id}/items/{item_id}/rerender", response_model=JobItemInfo)
async def rerender_item(job_id: str, item_id: str, body: RerenderRequest):
    try:
        return await job_manager.rerender_item(job_id, item_id, body.blocks)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/jobs/{job_id}/download")
async def download_batch_zip(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    zip_bytes = storage.build_batch_zip(job)
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="photo-translate-{job_id}.zip"',
        },
    )


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    job_manager.delete_job(job_id)
    return {"deleted": True}
