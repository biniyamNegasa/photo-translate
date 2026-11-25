import io
import json
import logging
import os
import re
import shutil
import stat
import tempfile
import time
import unicodedata
import urllib.parse
import zipfile
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.config import settings
from app.schemas import JobInfo

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ALLOWED_PIL_FORMATS = ("PNG", "JPEG", "WEBP", "BMP")

MAX_IMAGE_PIXELS = 25_000_000
MAX_IMAGE_DIMENSION = 8192
MAX_BATCH_PIXELS = 150_000_000

MAX_ZIP_ENTRIES = 1000
ZIP_CHUNK_SIZE = 64 * 1024
DEFAULT_MAX_UNCOMPRESSED_CAP = 250 * 1024 * 1024

JOB_ID_RE = re.compile(r"^[a-f0-9]{32}$")
ITEM_ID_RE = re.compile(r"^img_[0-9]{3,6}$")
BLOCK_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

_BIDI_AND_FORMAT_CHARS = frozenset(
    {
        "\u061c",
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
        "\ufeff",
    }
)

# Enforce global Pillow decompression bomb limit (25 MP)
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class ImageLimitError(ValueError):
    """Raised when an uploaded image violates pixel, dimension, or batch pixel safety limits."""


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


def _strip_control_and_bidi(text: str) -> str:
    out: List[str] = []
    for ch in text:
        if ch in _BIDI_AND_FORMAT_CHARS:
            continue
        if unicodedata.category(ch).startswith("C"):
            continue
        out.append(ch)
    return "".join(out)


def sanitize_filename(name: str, fallback: str = "image.png", max_length: int = 120) -> str:
    cleaned = _strip_control_and_bidi(name or "")
    # Normalize Windows separators before extracting basename
    base = Path(cleaned.replace("\\", "/")).name.strip()
    if not base or base in (".", ".."):
        return fallback

    safe = "".join(c if c.isalnum() or c in ("-", "_", ".", " ") else "_" for c in base)
    safe = safe.lstrip(". ").rstrip(" ")
    if not safe or safe in (".", ".."):
        return fallback

    if len(safe) > max_length:
        p = Path(safe)
        ext = p.suffix[:16]
        stem_budget = max(1, max_length - len(ext))
        stem = p.stem[:stem_budget].rstrip(" .") or "image"
        safe = f"{stem}{ext}"

    return safe or fallback


def build_content_disposition(filename: str) -> str:
    """Build an RFC 5987-safe Content-Disposition header that is always pure ASCII/latin-1 safe."""
    safe_name = sanitize_filename(filename, fallback="download.png")
    p = Path(safe_name)
    ascii_chars = []
    for ch in p.stem:
        code = ord(ch)
        if 0x20 <= code <= 0x7E and ch not in ('"', "\\", ";", "/", "\r", "\n"):
            ascii_chars.append(ch)
        else:
            ascii_chars.append("_")
    ascii_stem = "".join(ascii_chars).strip(" ._")
    if not ascii_stem:
        ascii_stem = "download"

    ascii_ext_chars = [
        ch
        for ch in p.suffix
        if 0x20 <= ord(ch) <= 0x7E and ch not in ('"', "\\", ";", "/", "\r", "\n")
    ]
    ascii_ext = "".join(ascii_ext_chars)
    ascii_fallback = f"{ascii_stem}{ascii_ext}"
    if len(ascii_fallback) > 120:
        ascii_fallback = f"{ascii_stem[: max(1, 120 - len(ascii_ext))]}{ascii_ext}"

    utf8_encoded = urllib.parse.quote(safe_name.encode("utf-8"), safe="!#$&+-.^_`|~")
    return f'attachment; filename="{ascii_fallback}"; filename*=UTF-8\'\'{utf8_encoded}'


def extract_uploaded_images(
    files: List[Tuple[str, bytes]],
    max_images: Optional[int] = None,
    max_uncompressed_bytes: Optional[int] = None,
) -> List[Tuple[str, bytes]]:
    """
    Accept individual images and/or .zip archives and return a flat list of (filename, raw_bytes).
    Guards against zip bombs (including spoofed headers & multi-ZIP exhaustion), zip-slip,
    symlinks, entry-count bombs, and disallowed image formats.
    """
    if max_images is None:
        max_images = settings.max_batch_images
    if max_uncompressed_bytes is None:
        max_uncompressed_bytes = min(
            settings.max_upload_mb * 4 * 1024 * 1024,
            DEFAULT_MAX_UNCOMPRESSED_CAP,
        )

    extracted: List[Tuple[str, bytes]] = []
    uncompressed_total = 0

    for orig_name, data in files:
        if not data:
            continue
        if len(extracted) >= max_images:
            break
        lower = orig_name.lower()
        if lower.endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                    infolist = zf.infolist()
                    if len(infolist) > MAX_ZIP_ENTRIES:
                        raise ValueError(
                            f"ZIP archive contains too many entries (maximum {MAX_ZIP_ENTRIES})."
                        )
                    for member in sorted(infolist, key=lambda m: m.filename):
                        if member.is_dir():
                            continue
                        # Skip UNIX symlinks stored inside ZIP archives
                        if stat.S_ISLNK(member.external_attr >> 16):
                            continue
                        # Ignore macOS resource forks, hidden files, and dot-segments
                        parts = Path(member.filename.replace("\\", "/")).parts
                        if any(p.startswith(".") or p == "__MACOSX" for p in parts):
                            continue
                        ext = Path(member.filename).suffix.lower()
                        if ext not in SUPPORTED_EXTENSIONS:
                            continue
                        if len(extracted) >= max_images:
                            break

                        # Fast-fail on declared size if already over budget
                        if member.file_size < 0 or uncompressed_total + member.file_size > max_uncompressed_bytes:
                            raise ValueError("ZIP archive exceeds maximum allowed uncompressed size.")

                        # Stream actual decompressed bytes; never trust header file_size or call zf.read()
                        remaining = max_uncompressed_bytes - uncompressed_total
                        chunks: List[bytes] = []
                        read_so_far = 0
                        with zf.open(member, "r") as fp:
                            while True:
                                to_read = min(ZIP_CHUNK_SIZE, remaining - read_so_far + 1)
                                chunk = fp.read(to_read)
                                if not chunk:
                                    break
                                read_so_far += len(chunk)
                                if read_so_far > remaining:
                                    raise ValueError(
                                        "ZIP archive exceeds maximum allowed uncompressed size."
                                    )
                                chunks.append(chunk)

                        uncompressed_total += read_so_far
                        extracted.append((sanitize_filename(member.filename), b"".join(chunks)))
            except ValueError:
                raise
            except (zipfile.BadZipFile, zlib.error, RuntimeError, EOFError) as exc:
                raise ValueError("Invalid or unsupported ZIP archive.") from exc
        else:
            ext = Path(orig_name).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                # Try sniffing via PIL with strict format allowlist (e.g., pasted clipboard blob)
                try:
                    with Image.open(io.BytesIO(data), formats=ALLOWED_PIL_FORMATS) as im:
                        fmt = (im.format or "").upper()
                        if fmt not in ALLOWED_PIL_FORMATS:
                            continue
                        ext = format_to_ext(fmt)
                        orig_name = f"{Path(orig_name).stem or 'pasted'}{ext}"
                except Image.DecompressionBombError as exc:
                    raise ImageLimitError("Image exceeds maximum allowed resolution.") from exc
                except Exception:
                    continue
            if len(extracted) >= max_images:
                break
            extracted.append((sanitize_filename(orig_name), data))

    return extracted


class JobStorage:
    """Manages ephemeral job folders, cached masks/inpainted layers, and TTL cleanup."""

    def __init__(self, root: Path = settings.jobs_dir) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass

    @staticmethod
    def _validate_job_id(job_id: str) -> str:
        if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
            raise ValueError("Invalid job ID.")
        return job_id

    @staticmethod
    def _validate_item_id(item_id: str) -> str:
        if not isinstance(item_id, str) or not ITEM_ID_RE.fullmatch(item_id):
            raise ValueError("Invalid item ID.")
        return item_id

    @staticmethod
    def _validate_block_id(block_id: str) -> str:
        if not isinstance(block_id, str) or not BLOCK_ID_RE.fullmatch(block_id):
            raise ValueError("Invalid block ID.")
        return block_id

    def job_dir(self, job_id: str) -> Path:
        self._validate_job_id(job_id)
        root_resolved = self.root.resolve()
        resolved = (root_resolved / job_id).resolve()
        if resolved.parent != root_resolved or resolved == root_resolved:
            raise ValueError("Invalid job path.")
        return resolved

    def item_dir(self, job_id: str, item_id: str, create: bool = False) -> Path:
        self._validate_item_id(item_id)
        jdir = self.job_dir(job_id)
        items_root = (jdir / "items").resolve()
        if items_root.parent != jdir:
            raise ValueError("Invalid items directory path.")
        resolved = (items_root / item_id).resolve()
        if resolved.parent != items_root or resolved == items_root:
            raise ValueError("Invalid item path.")
        if create:
            jdir.mkdir(parents=True, exist_ok=True, mode=0o700)
            resolved.mkdir(parents=True, exist_ok=True, mode=0o700)
        return resolved

    def save_original_image(
        self,
        job_id: str,
        item_id: str,
        filename: str,
        raw_bytes: bytes,
        remaining_batch_pixels: Optional[int] = None,
    ) -> Tuple[int, int, str]:
        """Decode, validate dimensions/pixels, EXIF-orient, and save original image atomically."""
        try:
            with Image.open(io.BytesIO(raw_bytes), formats=ALLOWED_PIL_FORMATS) as img:
                raw_fmt = (img.format or "").upper()
                if raw_fmt not in ALLOWED_PIL_FORMATS:
                    raise ValueError("Unsupported image format.")

                w_raw, h_raw = img.size
                if w_raw <= 0 or h_raw <= 0:
                    raise ValueError("Invalid image dimensions.")
                if w_raw > MAX_IMAGE_DIMENSION or h_raw > MAX_IMAGE_DIMENSION:
                    raise ImageLimitError(
                        f"Image dimensions exceed maximum allowed size ({MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})."
                    )
                if w_raw * h_raw > MAX_IMAGE_PIXELS:
                    raise ImageLimitError(
                        f"Image exceeds maximum allowed resolution ({MAX_IMAGE_PIXELS // 1_000_000} MP)."
                    )
                if remaining_batch_pixels is not None and (w_raw * h_raw) > remaining_batch_pixels:
                    raise ImageLimitError("Batch exceeds maximum allowed total image resolution.")

                img = ImageOps.exif_transpose(img)
                ext = Path(filename).suffix.lower()
                fmt = (
                    normalize_format(ext)
                    if ext in SUPPORTED_EXTENSIONS
                    else normalize_format(f".{raw_fmt.lower()}")
                )
                rgb = img.convert("RGB")
                w, h = rgb.size
                if w > MAX_IMAGE_DIMENSION or h > MAX_IMAGE_DIMENSION or (w * h) > MAX_IMAGE_PIXELS:
                    raise ImageLimitError("Image exceeds maximum allowed resolution.")

                idir = self.item_dir(job_id, item_id, create=True)
                orig_ext = format_to_ext(fmt)
                self._atomic_save_pil(rgb, idir / f"original{orig_ext}", fmt)
                # Lossless working copy for exact pixel compositing
                self._atomic_save_pil(rgb, idir / "original_rgb.png", "PNG")
                return w, h, fmt
        except Image.DecompressionBombError as exc:
            raise ImageLimitError("Image exceeds maximum allowed resolution.") from exc

    def _save_pil(self, img: Image.Image, path: Path, fmt: str) -> None:
        if fmt == "JPEG":
            img.save(path, format="JPEG", quality=95, subsampling=0)
        elif fmt == "WEBP":
            img.save(path, format="WEBP", quality=92)
        else:
            img.save(path, format="PNG")

    def _atomic_save_pil(self, img: Image.Image, target: Path, fmt: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(target.parent),
            prefix=f".{target.stem}.",
            suffix=f"{target.suffix}.tmp",
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            self._save_pil(img, tmp_path, fmt)
            os.replace(tmp_path, target)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def load_original_rgb(self, job_id: str, item_id: str) -> np.ndarray:
        path = self.item_dir(job_id, item_id, create=False) / "original_rgb.png"
        with Image.open(path, formats=ALLOWED_PIL_FORMATS) as img:
            return np.array(img.convert("RGB"), dtype=np.uint8)

    def save_inpainted_rgb(self, job_id: str, item_id: str, inpainted_rgb: np.ndarray) -> None:
        path = self.item_dir(job_id, item_id, create=True) / "inpainted_rgb.png"
        self._atomic_save_pil(Image.fromarray(inpainted_rgb, mode="RGB"), path, "PNG")

    def load_inpainted_rgb(self, job_id: str, item_id: str) -> np.ndarray:
        path = self.item_dir(job_id, item_id, create=False) / "inpainted_rgb.png"
        if not path.is_file():
            return self.load_original_rgb(job_id, item_id)
        with Image.open(path, formats=ALLOWED_PIL_FORMATS) as img:
            return np.array(img.convert("RGB"), dtype=np.uint8)

    def save_block_masks(self, job_id: str, item_id: str, masks: Dict[str, np.ndarray]) -> None:
        mdir = self.item_dir(job_id, item_id, create=True) / "masks"
        mdir.mkdir(parents=True, exist_ok=True, mode=0o700)
        mdir_resolved = mdir.resolve()
        for bid, mask in masks.items():
            self._validate_block_id(bid)
            target = (mdir_resolved / f"{bid}.png").resolve()
            if target.parent != mdir_resolved:
                raise ValueError("Invalid mask path.")
            cv2.imwrite(str(target), mask)

    def load_block_masks(self, job_id: str, item_id: str) -> Dict[str, np.ndarray]:
        mdir = self.item_dir(job_id, item_id, create=False) / "masks"
        masks: Dict[str, np.ndarray] = {}
        if not mdir.is_dir():
            return masks
        for p in mdir.glob("*.png"):
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                masks[p.stem] = m
        return masks

    def invalidate_batch_zip(self, job_id: str) -> None:
        zip_path = self.job_dir(job_id) / "batch_translated.zip"
        try:
            zip_path.unlink(missing_ok=True)
        except OSError:
            pass

    def save_translated_image(self, job_id: str, item_id: str, rgb: np.ndarray, fmt: str) -> Path:
        idir = self.item_dir(job_id, item_id, create=True)
        out_path = idir / f"translated{format_to_ext(fmt)}"
        self._atomic_save_pil(Image.fromarray(rgb, mode="RGB"), out_path, fmt)
        self.invalidate_batch_zip(job_id)
        return out_path

    def get_original_path(self, job_id: str, item_id: str, fmt: str) -> Path:
        return self.item_dir(job_id, item_id, create=False) / f"original{format_to_ext(fmt)}"

    def get_translated_path(self, job_id: str, item_id: str, fmt: str) -> Path:
        p = self.item_dir(job_id, item_id, create=False) / f"translated{format_to_ext(fmt)}"
        if p.is_file():
            return p
        return self.get_original_path(job_id, item_id, fmt)

    def save_job_meta(self, job: JobInfo) -> None:
        jdir = self.job_dir(job.job_id)
        jdir.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=str(jdir),
                prefix="job.",
                suffix=".tmp",
                delete=False,
                mode="w",
                encoding="utf-8",
            ) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(job.model_dump_json(indent=2))
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, jdir / "job.json")
        except Exception:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def build_batch_zip_file(self, job: JobInfo) -> Path:
        """Build (or reuse) a disk-backed batch ZIP atomically inside the job directory."""
        jdir = self.job_dir(job.job_id)
        if not jdir.is_dir():
            raise FileNotFoundError("Job directory not found.")

        target = jdir / "batch_translated.zip"
        if target.is_file():
            return target

        fd, tmp_name = tempfile.mkstemp(
            dir=str(jdir),
            prefix=".batch.",
            suffix=".zip.tmp",
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        used_names: Dict[str, int] = {}
        try:
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for item in job.items:
                    path = self.get_translated_path(job.job_id, item.item_id, item.format)
                    if not path.is_file():
                        continue
                    arc_name = sanitize_filename(item.filename)
                    if arc_name in used_names:
                        used_names[arc_name] += 1
                        p = Path(arc_name)
                        arc_name = f"{p.stem}_{used_names[arc_name]}{p.suffix}"
                    else:
                        used_names[arc_name] = 1
                    zf.write(path, arcname=arc_name)
            os.replace(tmp_path, target)
            return target
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def build_batch_zip(self, job: JobInfo) -> bytes:
        return self.build_batch_zip_file(job).read_bytes()

    def delete_job(self, job_id: str) -> None:
        jdir = self.job_dir(job_id)
        if jdir.exists():
            shutil.rmtree(jdir, ignore_errors=True)

    def cleanup_expired(self, now: Optional[float] = None) -> List[str]:
        now = now or time.time()
        removed: List[str] = []
        root_resolved = self.root.resolve()
        if not root_resolved.is_dir():
            return removed
        for child in root_resolved.iterdir():
            if child.is_symlink() or not child.is_dir():
                continue
            if child.resolve().parent != root_resolved:
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
