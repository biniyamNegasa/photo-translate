from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


_BACKEND_DIR = Path(__file__).resolve().parents[1]


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

    # Ephemeral storage & batch limits
    job_ttl_seconds: int = 3600  # 1 hour default TTL
    max_upload_mb: int = 100
    max_batch_images: int = 100
    worker_concurrency: int = 2
    item_concurrency: int = 3          # Parallel images in flight per batch job (memory-safe on 8GB RAM)
    bubble_ocr_concurrency: int = 4    # Parallel speech-bubble OCR workers per image

    # Models (local ONNX for comic/manga detection, Japanese manga OCR, and LaMa inpainting)
    lama_hf_repo: str = "Carve/LaMa-ONNX"
    lama_hf_filename: str = "lama_fp32.onnx"

    comic_detector_hf_repo: str = "mayocream/comic-text-detector-onnx"
    comic_detector_hf_filename: str = "comic-text-detector.onnx"
    comic_detector_conf: float = 0.25
    comic_detector_iou: float = 0.45

    manga_ocr_hf_repo: str = "onnx-community/manga-ocr-base-ONNX"
    manga_ocr_vocab_repo: str = "mayocream/manga-ocr-onnx"
    manga_ocr_encoder_filename: str = "onnx/encoder_model_int8.onnx"
    manga_ocr_decoder_filename: str = "onnx/decoder_model_int8.onnx"

    # Translation API configs
    default_translator: str = "google"  # deepseek | addis | google | deepl
    deepseek_api_key: str = ""
    deepseek_api_url: str = "https://api.deepseek.com/chat/completions"
    deepseek_model: str = "deepseek-chat"
    deepl_api_key: str = ""
    deepl_api_url: str = "https://api-free.deepl.com/v2/translate"
    addis_api_key: str = ""
    addis_api_url: str = "https://api.addisassistant.com/api/v1/translate"

    # OCR tuning
    ocr_min_confidence: float = 0.45
    ocr_min_box_height: int = 8

    # CORS
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:9000",
        "http://127.0.0.1:9000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


settings = Settings()
settings.cache_dir.mkdir(parents=True, exist_ok=True)
settings.jobs_dir.mkdir(parents=True, exist_ok=True)
settings.models_dir.mkdir(parents=True, exist_ok=True)
settings.fonts_dir.mkdir(parents=True, exist_ok=True)
