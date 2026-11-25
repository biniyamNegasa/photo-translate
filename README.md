# Photo Translate

Open-source web application that translates text inside **manga, comics, memes, drawings, and photos** while keeping all non-text artwork bit-for-bit untouched—no closed image-generation APIs and no censorship flags.

Upload a single image, multiple pages, or a `.zip` chapter archive, pick your target language and translation API, and inspect the result on a zoomable Before/After proofing desk with instant text editing and batch export.

## Features

- **Comic & Manga-First Detection (`comic-text-detector` ONNX):** Speech-bubble grouping, vertical/horizontal DBNet line orientation, manga right-to-left reading order, and learned UNet pixel-exact ink segmentation masks.
- **Specialized Manga + Multilingual OCR:**
  - Pure-ONNX quantized **`manga-ocr`** (zero PyTorch) for vertical & stylized Japanese manga bubbles with automatic furigana stripping.
  - Fast recognition-only **`RapidOCR`** directly on detected text lines for English, Chinese, Korean, Arabic, Amharic, Cyrillic, and European scripts (with automatic full-bubble fallback).
- **Pixel-Exact Erasure & Inpainting:**
  - Clean solid background fill inside uniform speech bubbles (zero blur or smudge artifacts).
  - Patch-based full-resolution **LaMa ONNX** (`Carve/LaMa-ONNX`) for screentones, halftones, outlined meme text, and photographic backgrounds.
  - Untranslated / disabled blocks restore original pixels bit-for-bit.
- **Pluggable Translation APIs:**
  - **Google Translate** (default, zero API key required)
  - **DeepSeek** (`deepseek-chat` context-aware page-batch JSON translation—great for Japanese/Chinese manga & idioms)
  - **Addis AI** (specialized for Amharic, Afaan Oromo, Tigrinya, Swahili & English)
  - **DeepL API** (high-nuance European & CJK translation)
- **Multilingual Comic Typesetting:** Auto-cached open-source **Comic Neue** for Latin speech bubbles + **Pillow + HarfBuzz/FriBidi (`libraqm`)** + Noto fonts with multi-font `cmap` fallback for Amharic/Ethiopic, Arabic/Hebrew RTL, CJK, Devanagari, Thai, and Cyrillic.
- **Proofing Desk UI ([Farm.js](https://farmjs.dev/) + React 19):**
  - Drag & drop, multi-file, `.zip` batch upload, and clipboard paste (`Ctrl+V`)
  - Parallel batch processing with live SSE progress, completion summary bar, desktop notifications, tab-title badge, and persistent toast
  - Zoomable (`100%–800%`), keyboard-accessible Before/After split slider & fullscreen `<dialog>` viewer
  - Instant (`<200ms`) Text Blocks inspector to edit translations or uncheck watermarks/SFX

## Prerequisites

- **Python 3.10+** (tested on Python 3.12)
- **Node.js 22.13+** and `npm`
- *(Linux recommended for complex scripts)* System **HarfBuzz / FriBidi (`libraqm`)** and **Noto fonts** (e.g., `fonts-noto`, `fonts-noto-cjk`, `fonts-noto-core`) for Amharic, Arabic RTL, and CJK rendering.

## Quick Start

### 1. Clone & Configure (Optional API Keys)
```bash
git clone <your-repo-url>
cd photo-translate
cp backend/.env.example backend/.env
chmod 600 backend/.env
# Edit backend/.env to add DeepSeek, Addis AI, or DeepL API keys (Google Translate works with no key).
```

### 2. Backend Setup
```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# Note: On Debian/Ubuntu without python3-venv ensurepip, run:
#   python3 -m venv --without-pip .venv
#   pip3 --python .venv/bin/python install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd frontend
npm install
```

### 4. Run Both Servers
From the repository root:
```bash
chmod +x start-dev.sh
./start-dev.sh
```
Open **http://localhost:3000** (FastAPI runs on **http://127.0.0.1:8000**).

> **First-run model cache:** Local ONNX models (`comic-text-detector` ~95 MB, quantized `manga-ocr` ~116 MB, and `LaMa` ~208 MB) plus `Comic Neue` are automatically downloaded into `backend/.cache/` on first use and cached locally.

## Configuration

See [backend/.env.example](backend/.env.example) for all options:

| Variable | Default | Description |
| --- | --- | --- |
| `PHOTO_TRANSLATE_API_KEY` | *(empty)* | Optional API key required on backend requests (`X-API-Key` / `Authorization: Bearer`) |
| `PHOTO_TRANSLATE_CORS_ORIGINS` | `["http://localhost:3000","http://127.0.0.1:3000"]` | Allowed browser origins for CORS |
| `PHOTO_TRANSLATE_ENABLE_DOCS` | `false` | Expose `/docs`, `/redoc`, and `/openapi.json` |
| `PHOTO_TRANSLATE_DEEPSEEK_API_KEY` | *(empty)* | Enables DeepSeek context-aware batch translator |
| `PHOTO_TRANSLATE_DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek model name |
| `PHOTO_TRANSLATE_ADDIS_API_KEY` | *(empty)* | Enables Addis AI (`am`, `om`, `ti`, `en`, `sw`, `fr`) |
| `PHOTO_TRANSLATE_DEEPL_API_KEY` | *(empty)* | Enables DeepL API translator |
| `PHOTO_TRANSLATE_JOB_TTL_SECONDS` | `3600` | Auto-expiry time for temporary job files |
| `PHOTO_TRANSLATE_WORKER_CONCURRENCY` | `2` | Maximum concurrent translation jobs |
| `PHOTO_TRANSLATE_ITEM_CONCURRENCY` | `3` | Concurrent images processed per batch job |
| `PHOTO_TRANSLATE_BUBBLE_OCR_CONCURRENCY` | `4` | Concurrent bubble OCR workers per image |
| `VITE_API_BASE` *(frontend)* | `http://127.0.0.1:8000` | Backend API base URL |

## Running Tests

Install development/test dependencies first (`requirements-dev.txt` includes `pytest`):

```bash
backend/.venv/bin/pip install -r backend/requirements-dev.txt

# Backend unit & pipeline tests
backend/.venv/bin/pytest backend/tests/test_pipeline.py -v

# Frontend type-check & production build
npm --prefix frontend run type-check
npm --prefix frontend run build
```

## Production Deployment & Security Hardening

Photo Translate defaults to a **localhost-first** workstation setup. Before exposing it on a shared network or public host, see [SECURITY.md](SECURITY.md) and apply the following hardening steps:

- **Never use `--reload` or `./start-dev.sh` in production:** Run Uvicorn/Gunicorn without `--reload` and build the frontend with `npm --prefix frontend run build && npm --prefix frontend run start`.
- **Require authentication & disable API docs:** Set a strong random `PHOTO_TRANSLATE_API_KEY` in `backend/.env` (`chmod 600 backend/.env`) and keep `PHOTO_TRANSLATE_ENABLE_DOCS=false`.
- **Pin explicit CORS origins:** Set `PHOTO_TRANSLATE_CORS_ORIGINS` strictly to your deployed frontend origin(s); never use `*`.
- **Place behind a TLS reverse proxy:** Terminate HTTPS at Nginx/Caddy/ALB and enforce request body limits, authentication, and per-IP rate limiting.
- **Tighten upload & retention limits:** Lower `PHOTO_TRANSLATE_MAX_UPLOAD_MB`, `PHOTO_TRANSLATE_MAX_BATCH_IMAGES`, `PHOTO_TRANSLATE_WORKER_CONCURRENCY`, and `PHOTO_TRANSLATE_JOB_TTL_SECONDS` to match your host resources and retention policy.

## Hardware & Performance Notes

- Runs out of the box on **CPU** (no PyTorch required). Typical speed on a 2-core laptop CPU is ~6–8s per comic page for detection + OCR + solid-bubble fill (plus ~10s per LaMa patch on textured panels).
- For fast production/cloud speed (<1s neural processing per page), deploy the FastAPI backend on a CUDA GPU (such as serverless NVIDIA T4/L4) with `onnxruntime-gpu`.

## License & Third-Party Models

- This project's source code is licensed under the [MIT License](LICENSE).
- Pretrained ONNX models and fonts downloaded at runtime into `backend/.cache/` remain subject to their respective upstream licenses:
  - [`comic-text-detector`](https://github.com/dmMaze/comic-text-detector) / `mayocream/comic-text-detector-onnx`
  - [`manga-ocr`](https://github.com/kha-white/manga-ocr) / `onnx-community/manga-ocr-base-ONNX` (Apache-2.0)
  - [`LaMa`](https://github.com/advimman/lama) / `Carve/LaMa-ONNX` (Apache-2.0)
  - [`RapidOCR`](https://github.com/RapidAI/RapidOCR) (Apache-2.0)
  - [`Comic Neue`](https://fonts.google.com/specimen/Comic+Neue) (SIL OFL 1.1)
