import logging
import os
import re
import threading
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from app.config import settings
from app.pipeline.inpainter import get_onnx_providers

logger = logging.getLogger(__name__)

# Special token IDs from kha-white/manga-ocr-base generation_config.json
START_TOKEN_ID = 2  # [CLS]
EOS_TOKEN_ID = 3    # [SEP]
SPECIAL_MAX_ID = 4  # [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4
SPECIAL_TOKENS = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}


def preprocess_manga_crop(crop_rgb: np.ndarray) -> np.ndarray:
    """
    Preprocess a speech-bubble RGB crop to match `kha-white/manga-ocr-base`:
      grayscale -> 3-channel RGB -> resize 224x224 bilinear -> normalize (x/255 - 0.5)/0.5 -> NCHW float32.
    """
    pil_img = Image.fromarray(crop_rgb, mode="RGB").convert("L").convert("RGB")
    pil_img = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
    arr = (np.array(pil_img, dtype=np.float32) / 255.0 - 0.5) / 0.5
    return arr.transpose(2, 0, 1)[None, ...]


def decode_manga_tokens(token_ids: Sequence[int], vocab: Sequence[str]) -> str:
    """Decode greedy token IDs from manga-ocr BERT decoder into clean Japanese text."""
    pieces: List[str] = []
    vocab_len = len(vocab)
    for tid in token_ids:
        if tid <= SPECIAL_MAX_ID or tid >= vocab_len:
            continue
        tok = vocab[tid]
        if tok in SPECIAL_TOKENS or (tok.startswith("<unused") and tok.endswith(">")):
            continue
        if tok.startswith("##"):
            tok = tok[2:]
        pieces.append(tok)

    text = "".join(pieces)
    text = text.replace(" ", "").replace("\u3000", "")
    # Normalize manga ellipses cleanly
    text = re.sub(r"…+", "…", text)
    text = re.sub(r"・{3,}", "…", text)
    text = re.sub(r"\.{3,}", "…", text)
    return text.strip()


def _has_repetition_loop(ids: List[int]) -> bool:
    """Guard against infinite repetition loops on non-Japanese or blank crops."""
    n = len(ids)
    if n < 10:
        return False
    # Check period 1, 2, or 3 repeating at least 4 times in a row
    for period in (1, 2, 3):
        if n >= period * 4:
            unit = ids[-period:]
            if ids[-period * 4 :] == unit * 4:
                return True
    return False


class MangaOCROnnx:
    """
    Pure ONNXRuntime + NumPy implementation of `kha-white/manga-ocr-base` (INT8 quantized, ~116 MB total).
    Zero PyTorch dependency. Purpose-built for vertical & horizontal Japanese manga text with furigana stripping.
    """

    def __init__(self) -> None:
        self._enc_sess = None
        self._dec_sess = None
        self._vocab: List[str] = []
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._load_failed = False

    def model_dir(self) -> Path:
        return settings.models_dir / "manga-ocr"

    def encoder_path(self) -> Path:
        return self.model_dir() / settings.manga_ocr_encoder_filename

    def decoder_path(self) -> Path:
        return self.model_dir() / settings.manga_ocr_decoder_filename

    def vocab_path(self) -> Path:
        return self.model_dir() / "vocab.txt"

    def is_downloaded(self) -> bool:
        return (
            self.encoder_path().is_file()
            and self.encoder_path().stat().st_size > 10_000_000
            and self.decoder_path().is_file()
            and self.decoder_path().stat().st_size > 5_000_000
            and self.vocab_path().is_file()
        )

    def _ensure_loaded(self) -> bool:
        if self._enc_sess is not None and self._dec_sess is not None:
            return True
        if self._load_failed:
            return False

        with self._lock:
            if self._enc_sess is not None and self._dec_sess is not None:
                return True

            if not self.is_downloaded():
                try:
                    from huggingface_hub import hf_hub_download

                    out_dir = self.model_dir()
                    out_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(
                        "Downloading quantized MangaOCR ONNX models (%s)...",
                        settings.manga_ocr_hf_repo,
                    )
                    hf_hub_download(
                        repo_id=settings.manga_ocr_hf_repo,
                        filename=settings.manga_ocr_encoder_filename,
                        local_dir=str(out_dir),
                    )
                    hf_hub_download(
                        repo_id=settings.manga_ocr_hf_repo,
                        filename=settings.manga_ocr_decoder_filename,
                        local_dir=str(out_dir),
                    )
                    hf_hub_download(
                        repo_id=settings.manga_ocr_vocab_repo,
                        filename="vocab.txt",
                        local_dir=str(out_dir),
                    )
                except Exception as exc:
                    logger.warning("Could not download MangaOCR ONNX models (%s).", exc)
                    self._load_failed = True
                    return False

            try:
                import onnxruntime as ort

                sess_opts = ort.SessionOptions()
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
                sess_opts.inter_op_num_threads = 1
                providers = get_onnx_providers()
                logger.info("Loading quantized MangaOCR ONNX sessions on %s...", providers[0])
                self._enc_sess = ort.InferenceSession(
                    str(self.encoder_path()),
                    sess_options=sess_opts,
                    providers=providers,
                )
                self._dec_sess = ort.InferenceSession(
                    str(self.decoder_path()),
                    sess_options=sess_opts,
                    providers=providers,
                )
                self._vocab = self.vocab_path().read_text(encoding="utf-8").splitlines()
                return True
            except Exception as exc:
                logger.warning("Failed initializing MangaOCR ONNX sessions (%s).", exc)
                self._load_failed = True
                return False

    def recognize(self, crop_rgb: np.ndarray, max_length: int = 96) -> Optional[str]:
        """
        Recognize Japanese manga text from an RGB speech-bubble crop.
        Returns `None` if MangaOCR is unavailable, or the recognized string (possibly empty).
        """
        if crop_rgb is None or crop_rgb.size == 0:
            return ""
        if not self._ensure_loaded():
            return None

        pixel_values = preprocess_manga_crop(crop_rgb)
        with self._infer_lock:
            enc_hidden = self._enc_sess.run(None, {"pixel_values": pixel_values})[0]

            ids: List[int] = [START_TOKEN_ID]
            for _ in range(max_length):
                logits = self._dec_sess.run(
                    None,
                    {
                        "input_ids": np.array([ids], dtype=np.int64),
                        "encoder_hidden_states": enc_hidden,
                    },
                )[0]
                next_id = int(np.argmax(logits[0, -1]))
                if next_id == EOS_TOKEN_ID:
                    break
                ids.append(next_id)
                if _has_repetition_loop(ids):
                    # Trim trailing repeated tail cleanly
                    ids = ids[:-6]
                    break

        return decode_manga_tokens(ids, self._vocab)


manga_ocr = MangaOCROnnx()
