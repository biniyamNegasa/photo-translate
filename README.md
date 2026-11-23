# Photo Translate

Seamless, 100% local image text translation for **memes, drawings, comics, and photos**.

Upload a single image, multiple images, or a `.zip` batch — pick your target language — and get back the exact same image with only the text cleanly replaced in the target language (zero closed image-gen APIs, zero censorship flags, bit-for-bit untouched non-text pixels).

## Architecture

- **Backend (`backend/`)**: Python **FastAPI** + async batch worker + SSE live progress
  - **Comic & Manga Detection (`comic-text-detector` ONNX)**: Speech-bubble block grouping, vertical/horizontal DBNet line orientation, manga right-to-left reading order, and **learned UNet pixel-exact ink segmentation masks** (running via OpenCV DNN on CPU in ~3s)
  - **Specialized Manga + Multilingual OCR**: Pure-ONNX quantized **`manga-ocr`** (zero PyTorch) for vertical & stylized Japanese manga bubbles (automatically stripping furigana), paired with **`rapidocr-onnxruntime`** for English, Chinese, Korean, Arabic, Amharic, and general-image fallback
  - **Stroke-Complete Masking & Style Analysis**: Fuses learned UNet ink masks with typography & speech-bubble interior geometry analysis
  - **Hybrid Erasure / Inpainting**: Clean solid fill inside uniform speech bubbles + patch-based full-resolution **LaMa ONNX** (`Carve/LaMa-ONNX`) for screentones, halftones, memes, and photographic backgrounds
  - **Pluggable API Translation**:
    - `deepseek`: **DeepSeek API** context-aware batch translator — ideal for Japanese/Chinese → English comics & memes (`PHOTO_TRANSLATE_DEEPSEEK_API_KEY`)
    - `addis`: **Addis AI** adapter specialized for Amharic, Afaan Oromo, Tigrinya & English (`PHOTO_TRANSLATE_ADDIS_API_KEY`)
    - `google`: Zero-config **Google Translate** adapter (default fallback, no API key required)
    - `deepl`: **DeepL API** adapter (`PHOTO_TRANSLATE_DEEPL_API_KEY`)
  - **Comic Lettering & Multilingual Rendering**: Auto-cached open-source **`Comic Neue`** lettering for comic/manga speech bubbles + Pillow + **HarfBuzz/FriBidi (`libraqm`)** + Noto fonts with multi-font `cmap` fallback for Ethiopic/Amharic, Arabic/Hebrew RTL, CJK, Cyrillic, Devanagari, Thai, etc.
  - **Ephemeral Storage**: Stateless temporary jobs with configurable TTL (`PHOTO_TRANSLATE_JOB_TTL_SECONDS`) and cached inpainted/mask layers for instant `<200ms` Advanced re-rendering.
- **Frontend (`frontend/`)**: **[Farm.js](https://farmjs.dev/) (`@farm.js/core`) + React 19 (TypeScript)**
  - File-system `src/app/` routing with SSR + `"use client"` interactive studio
  - Drag & drop, multi-select, `.zip` batch upload, and clipboard paste (`Ctrl+V`)
  - Live per-image progress via SSE & interactive Before/After split slider
  - Optional collapsible **Advanced** drawer to edit text or toggle off watermarks/logos (restoring original pixels bit-for-bit)

## Quick Start (Local Dev)

### 1. Backend
```bash
cd backend
python3 -m venv --without-pip .venv
pip3 --python .venv/bin/python install -r requirements.txt
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Frontend (Farm.js + React)
```bash
cd frontend
npm install
npm run dev
```

Or run both together:
```bash
chmod +x start-dev.sh
./start-dev.sh
```

Open **http://localhost:3000** in your browser.

> **First run note:** Local ONNX models (`comic-text-detector` ~95 MB, quantized `manga-ocr` ~116 MB, and `LaMa` ~208 MB) plus the `Comic Neue` comic lettering font are automatically downloaded into `backend/.cache/` on first use and cached locally.

## Environment Variables (Optional)

| Variable | Default | Description |
| --- | --- | --- |
| `PHOTO_TRANSLATE_JOB_TTL_SECONDS` | `3600` | Auto-expiry time for temporary job files |
| `PHOTO_TRANSLATE_ITEM_CONCURRENCY` | `3` | Number of images processed concurrently per batch job |
| `PHOTO_TRANSLATE_DEEPSEEK_API_KEY` | *(empty)* | Enables DeepSeek context-aware batch translator when set |
| `PHOTO_TRANSLATE_DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek model name |
| `PHOTO_TRANSLATE_ADDIS_API_KEY` | *(empty)* | Enables Addis AI adapter (`am`, `om`, `ti`, `en`, `sw`) when set |
| `PHOTO_TRANSLATE_ADDIS_API_URL` | `https://api.addisassistant.com/api/v1/translate` | Addis AI translation endpoint |
| `PHOTO_TRANSLATE_DEEPL_API_KEY` | *(empty)* | Enables DeepL adapter when set |
