import hashlib
import hmac
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

import httpx

logger = logging.getLogger(__name__)

# Verified default HuggingFace repos, filenames, commit SHAs, and SHA-256 digests
DEFAULT_LAMA_REPO = "Carve/LaMa-ONNX"
DEFAULT_LAMA_FILENAME = "lama_fp32.onnx"
LAMA_REVISION = "c3c0c9e468934d62e79c329e35d82dd09ff8c444"
LAMA_SHA256 = "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6"

DEFAULT_COMIC_DETECTOR_REPO = "mayocream/comic-text-detector-onnx"
DEFAULT_COMIC_DETECTOR_FILENAME = "comic-text-detector.onnx"
COMIC_DETECTOR_REVISION = "a5d67ec772adef819ef5b0e7aa701fcf4c8bf74a"
COMIC_DETECTOR_SHA256 = "1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f"

DEFAULT_MANGA_OCR_REPO = "onnx-community/manga-ocr-base-ONNX"
DEFAULT_MANGA_OCR_ENCODER_FILENAME = "onnx/encoder_model_int8.onnx"
DEFAULT_MANGA_OCR_DECODER_FILENAME = "onnx/decoder_model_int8.onnx"
MANGA_OCR_REVISION = "f9023406bb2f6b17df67bc4a327c56ecd20611f0"
MANGA_OCR_ENCODER_SHA256 = "ddd1af56963093795705fa38da6ce7e6567d1658e7c7359db7e13fcd37dbf279"
MANGA_OCR_DECODER_SHA256 = "2e7177d2b0a59f1c612b694ed70c13971bee765cc2b2bc7bc9376e4753652f27"

DEFAULT_MANGA_OCR_VOCAB_REPO = "mayocream/manga-ocr-onnx"
DEFAULT_MANGA_OCR_VOCAB_FILENAME = "vocab.txt"
MANGA_OCR_VOCAB_REVISION = "24b12778d85800835e2ca409236de281b8ab7b9f"
MANGA_OCR_VOCAB_SHA256 = "5cb5c5586d98a2f331d9f8828e4586479b0611bfba5d8c3b6dadffc84d6a36a3"

COMIC_NEUE_SHA256: Dict[str, str] = {
    "ComicNeue-Bold.ttf": "3e7e5fccfd7e0788f317b43312151c1bd5cf058c9697a8d83eac3939050bd61e",
    "ComicNeue-Regular.ttf": "a0ee5a37c8b27c4db0700137d928598b1e23b0089e1546a8961909176b779360",
}

_HASH_CHUNK_BYTES = 1024 * 1024  # 1 MiB streaming chunks


def verify_file_sha256(path: Path, expected_sha256: str) -> bool:
    """Verify a file's SHA-256 digest in streaming 1 MiB chunks using constant-time comparison."""
    expected = (expected_sha256 or "").strip().lower()
    if len(expected) != 64:
        return False
    try:
        if not path.is_file():
            return False
        hasher = hashlib.sha256()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(_HASH_CHUNK_BYTES)
                if not chunk:
                    break
                hasher.update(chunk)
        actual = hasher.hexdigest().lower()
        return hmac.compare_digest(actual, expected)
    except OSError as exc:
        logger.warning("Failed reading %s for SHA-256 verification: %s", path.name, exc)
        return False


def download_verified_file(
    url: str,
    target_path: Path,
    expected_sha256: str,
    max_bytes: int,
    timeout: float = 8.0,
) -> Path:
    """
    Safely stream-download a file without following redirects, enforcing a strict byte cap,
    verifying its SHA-256 digest, fsyncing, and atomically moving it into `target_path`.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None

    try:
        with httpx.stream("GET", url, follow_redirects=False, timeout=timeout) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"Download rejected with HTTP {resp.status_code}")

            content_len = resp.headers.get("content-length")
            if content_len is not None:
                try:
                    if int(content_len) > max_bytes:
                        raise ValueError("Remote file exceeds maximum allowed size.")
                except ValueError as exc:
                    if "exceeds maximum" in str(exc):
                        raise

            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{target_path.name}.",
                suffix=".tmp",
                dir=str(target_path.parent),
            )
            tmp_path = Path(tmp_name)
            hasher = hashlib.sha256()
            total = 0
            with os.fdopen(fd, "wb") as out_fh:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("Downloaded file exceeds maximum allowed size.")
                    hasher.update(chunk)
                    out_fh.write(chunk)
                out_fh.flush()
                os.fsync(out_fh.fileno())

        actual = hasher.hexdigest().lower()
        expected = expected_sha256.strip().lower()
        if not hmac.compare_digest(actual, expected):
            raise ValueError(f"SHA-256 digest mismatch for {target_path.name}.")

        os.replace(tmp_path, target_path)
        tmp_path = None
        return target_path
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
