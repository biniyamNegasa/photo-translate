import ipaddress
import os
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_BACKEND_DIR = Path(__file__).resolve().parents[1]


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _validate_https_outbound_url(value: str) -> str:
    url = value.strip()
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("Outbound API URL must use the https:// scheme.")
    if parsed.username or parsed.password:
        raise ValueError("Outbound API URL must not contain embedded credentials.")
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        raise ValueError("Outbound API URL must specify a hostname.")
    if host in ("localhost",) or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("Outbound API URL must not target localhost or internal hostnames.")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is not None and (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        raise ValueError("Outbound API URL must not target private, loopback, or link-local IP addresses.")
    return url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PHOTO_TRANSLATE_",
        env_file=str(_BACKEND_DIR / ".env"),
        extra="ignore",
    )

    # Directories
    cache_dir: Path = _BACKEND_DIR / ".cache"
    jobs_dir: Path = _BACKEND_DIR / ".cache" / "jobs"
    models_dir: Path = _BACKEND_DIR / ".cache" / "models"
    fonts_dir: Path = _BACKEND_DIR / ".cache" / "fonts"

    # Security & API surface
    api_key: SecretStr = Field(default=SecretStr(""))
    enable_docs: bool = False
    rate_limit_jobs_per_minute: int = Field(default=30, ge=1, le=10000)
    rate_limit_actions_per_minute: int = Field(default=120, ge=1, le=100000)

    # Ephemeral storage & batch limits
    job_ttl_seconds: int = Field(default=3600, ge=60, le=86400)  # 1 hour default TTL
    max_upload_mb: int = Field(default=100, ge=1, le=500)
    max_batch_images: int = Field(default=100, ge=1, le=500)
    max_active_jobs: int = Field(default=64, ge=1, le=1000)
    worker_concurrency: int = Field(default=2, ge=1, le=16)
    item_concurrency: int = Field(default=3, ge=1, le=16)  # Parallel images in flight per batch job
    bubble_ocr_concurrency: int = Field(default=4, ge=1, le=16)  # Parallel speech-bubble OCR workers per image

    # Models (local ONNX for comic/manga detection, Japanese manga OCR, and LaMa inpainting)
    lama_hf_repo: str = "Carve/LaMa-ONNX"
    lama_hf_filename: str = "lama_fp32.onnx"

    comic_detector_hf_repo: str = "mayocream/comic-text-detector-onnx"
    comic_detector_hf_filename: str = "comic-text-detector.onnx"
    comic_detector_conf: float = Field(default=0.25, ge=0.01, le=1.0)
    comic_detector_iou: float = Field(default=0.45, ge=0.01, le=1.0)

    manga_ocr_hf_repo: str = "onnx-community/manga-ocr-base-ONNX"
    manga_ocr_vocab_repo: str = "mayocream/manga-ocr-onnx"
    manga_ocr_encoder_filename: str = "onnx/encoder_model_int8.onnx"
    manga_ocr_decoder_filename: str = "onnx/decoder_model_int8.onnx"

    # Translation API configs
    default_translator: str = "google"  # deepseek | addis | google | deepl
    deepseek_api_key: SecretStr = Field(default=SecretStr(""))
    deepseek_api_url: str = "https://api.deepseek.com/chat/completions"
    deepseek_model: str = "deepseek-chat"
    deepl_api_key: SecretStr = Field(default=SecretStr(""))
    deepl_api_url: str = "https://api-free.deepl.com/v2/translate"
    addis_api_key: SecretStr = Field(default=SecretStr(""))
    addis_api_url: str = "https://api.addisassistant.com/api/v1/translate"

    # OCR tuning
    ocr_min_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    ocr_min_box_height: int = Field(default=8, ge=1, le=256)

    # CORS
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:9000",
        "http://127.0.0.1:9000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    @field_validator("deepseek_api_url", "deepl_api_url", "addis_api_url")
    @classmethod
    def _check_api_urls(cls, v: str) -> str:
        return _validate_https_outbound_url(v)

    def get_secret(self, field_name: str) -> str:
        """Read a secret setting safely whether stored as SecretStr or patched as plain str."""
        val = getattr(self, field_name, "")
        if isinstance(val, SecretStr):
            return val.get_secret_value().strip()
        if hasattr(val, "get_secret_value"):
            return str(val.get_secret_value() or "").strip()
        return str(val or "").strip()


settings = Settings()
_ensure_private_dir(settings.cache_dir)
_ensure_private_dir(settings.jobs_dir)
_ensure_private_dir(settings.models_dir)
_ensure_private_dir(settings.fonts_dir)
