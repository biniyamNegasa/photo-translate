import io
import json
import logging
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.config import settings
from app.schemas import JobInfo

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def normalize_format(ext: str) -> str:
    ext = ext.lower()
    if ext in (".jpg", ".jpeg"):
        return "JPEG"
    if ext == ".webp":
        return "WEBP"
    return "PNG"


def format_to_media_type(fmt: str) -> str:
    if fmt == "JPEG":
        return "image/jpeg"
    if fmt == "WEBP":
        return "image/webp"
    return "image/png"


def format_to_ext(fmt: str) -> str:
    if fmt == "JPEG":
        return ".jpg"
    if fmt == "WEBP":
        return ".webp"
    return ".png"


def sanitize_filename(name: str, fallback: str = "image.png") -> str:
    base = Path(name).name.strip()
    if not base or base in (".", ".."):
        return fallback
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".", " ") else "_" for c in base)
    return safe or fallback


def extract_uploaded_images(
    files: List[Tuple[str, bytes]],
    max_images: int = settings.max_batch_images,
) -> List[Tuple[str, bytes]]:
    """
    Accept individual images and/or .zip archives and return a flat list of (filename, raw_bytes).
    Guards against zip-slip path traversal and hidden macOS metadata files.
    """
    extracted: List[Tuple[str, bytes]] = []

    for orig_name, data in files:
        if not data:
            continue
        lower = orig_name.lower()
        if lower.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                for member in sorted(zf.infolist(), key=lambda m: m.filename):
                    if member.is_dir():
                        continue
                    # Ignore macOS resource forks and hidden files
                    parts = Path(member.filename).parts
                    if any(p.startswith(".") or p == "__MACOSX" for p in parts):
                        continue
                    ext = Path(member.filename).suffix.lower()
                    if ext not in SUPPORTED_EXTENSIONS:
                        continue
                    if len(extracted) >= max_images:
                        break
                    extracted.append((sanitize_filename(member.filename), zf.read(member)))
        else:
            ext = Path(orig_name).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                # Try sniffing via PIL even if extension was missing (e.g., pasted clipboard blob)
                try:
                    with Image.open(io.BytesIO(data)) as im:
                        fmt = (im.format or "PNG").upper()
                        ext = format_to_ext(fmt)
                        orig_name = f"{Path(orig_name).stem or 'pasted'}{ext}"
                except Exception:
                    continue
            if len(extracted) >= max_images:
                break
            extracted.append((sanitize_filename(orig_name), data))

    return extracted


class JobStorage:
    """Manages ephemeral job folders, cached masks/inpainted layers, and TTL cleanup."""

    def __init__(self, root: Path = settings.jobs_dir) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def item_dir(self, job_id: str, item_id: str) -> Path:
        d = self.job_dir(job_id) / "items" / item_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_original_image(
        self,
        job_id: str,
        item_id: str,
        filename: str,
        raw_bytes: bytes,
    ) -> Tuple[int, int, str]:
        """Decode, EXIF-orient, and save original image in its native format + lossless RGB PNG cache."""
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img = ImageOps.exif_transpose(img)
            ext = Path(filename).suffix.lower()
            fmt = normalize_format(ext) if ext in SUPPORTED_EXTENSIONS else normalize_format(f".{(img.format or 'png').lower()}")
            rgb = img.convert("RGB")
            w, h = rgb.size

            idir = self.item_dir(job_id, item_id)
            orig_ext = format_to_ext(fmt)
            self._save_pil(rgb, idir / f"original{orig_ext}", fmt)
            # Lossless working copy for exact pixel compositing
            rgb.save(idir / "original_rgb.png", format="PNG")
            return w, h, fmt

    def _save_pil(self, img: Image.Image, path: Path, fmt: str) -> None:
        if fmt == "JPEG":
            img.save(path, format="JPEG", quality=95, subsampling=0)
        elif fmt == "WEBP":
            img.save(path, format="WEBP", quality=92)
        else:
            img.save(path, format="PNG")

    def load_original_rgb(self, job_id: str, item_id: str) -> np.ndarray:
        path = self.item_dir(job_id, item_id) / "original_rgb.png"
        with Image.open(path) as img:
            return np.array(img.convert("RGB"), dtype=np.uint8)

    def save_inpainted_rgb(self, job_id: str, item_id: str, inpainted_rgb: np.ndarray) -> None:
        path = self.item_dir(job_id, item_id) / "inpainted_rgb.png"
        Image.fromarray(inpainted_rgb, mode="RGB").save(path, format="PNG")

    def load_inpainted_rgb(self, job_id: str, item_id: str) -> np.ndarray:
        path = self.item_dir(job_id, item_id) / "inpainted_rgb.png"
        if not path.is_file():
            return self.load_original_rgb(job_id, item_id)
        with Image.open(path) as img:
            return np.array(img.convert("RGB"), dtype=np.uint8)

    def save_block_masks(self, job_id: str, item_id: str, masks: Dict[str, np.ndarray]) -> None:
        mdir = self.item_dir(job_id, item_id) / "masks"
        mdir.mkdir(parents=True, exist_ok=True)
        for bid, mask in masks.items():
            cv2.imwrite(str(mdir / f"{bid}.png"), mask)

    def load_block_masks(self, job_id: str, item_id: str) -> Dict[str, np.ndarray]:
        mdir = self.item_dir(job_id, item_id) / "masks"
        masks: Dict[str, np.ndarray] = {}
        if not mdir.is_dir():
            return masks
        for p in mdir.glob("*.png"):
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                masks[p.stem] = m
        return masks

    def save_translated_image(self, job_id: str, item_id: str, rgb: np.ndarray, fmt: str) -> Path:
        idir = self.item_dir(job_id, item_id)
        out_path = idir / f"translated{format_to_ext(fmt)}"
        self._save_pil(Image.fromarray(rgb, mode="RGB"), out_path, fmt)
        return out_path

    def get_original_path(self, job_id: str, item_id: str, fmt: str) -> Path:
        return self.item_dir(job_id, item_id) / f"original{format_to_ext(fmt)}"

    def get_translated_path(self, job_id: str, item_id: str, fmt: str) -> Path:
        p = self.item_dir(job_id, item_id) / f"translated{format_to_ext(fmt)}"
        if p.is_file():
            return p
        return self.get_original_path(job_id, item_id, fmt)

    def save_job_meta(self, job: JobInfo) -> None:
        jdir = self.job_dir(job.job_id)
        jdir.mkdir(parents=True, exist_ok=True)
        tmp_path = jdir / "job.json.tmp"
        tmp_path.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp_path, jdir / "job.json")

    def build_batch_zip(self, job: JobInfo) -> bytes:
        buf = io.BytesIO()
        used_names: Dict[str, int] = {}
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for item in job.items:
                path = self.get_translated_path(job.job_id, item.item_id, item.format)
                if not path.is_file():
                    continue
                # Deduplicate filenames inside the archive cleanly
                arc_name = item.filename
                if arc_name in used_names:
                    used_names[arc_name] += 1
                    p = Path(arc_name)
                    arc_name = f"{p.stem}_{used_names[arc_name]}{p.suffix}"
                else:
                    used_names[arc_name] = 1
                zf.write(path, arcname=arc_name)
        return buf.getvalue()

    def delete_job(self, job_id: str) -> None:
        jdir = self.job_dir(job_id)
        if jdir.exists():
            shutil.rmtree(jdir, ignore_errors=True)

    def cleanup_expired(self, now: Optional[float] = None) -> List[str]:
        now = now or time.time()
        removed: List[str] = []
        if not self.root.is_dir():
            return removed
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            meta_file = child / "job.json"
            expired = False
            if meta_file.is_file():
                try:
                    data = json.loads(meta_file.read_text(encoding="utf-8"))
                    if float(data.get("expires_at", 0)) < now:
                        expired = True
                except Exception:
                    expired = (now - child.stat().st_mtime) > settings.job_ttl_seconds
            else:
                expired = (now - child.stat().st_mtime) > settings.job_ttl_seconds

            if expired:
                shutil.rmtree(child, ignore_errors=True)
                removed.append(child.name)
        if removed:
            logger.info("Cleaned up %d expired job(s).", len(removed))
        return removed


storage = JobStorage()
