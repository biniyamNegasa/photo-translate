# Photo Translate

Seamless, 100% local image text translation for **memes, drawings, comics, and photos**.

Upload a single image, multiple images, or a `.zip` batch — pick your target language — and get back the exact same image with only the text cleanly replaced in the target language (zero closed image-gen APIs, zero censorship flags, bit-for-bit untouched non-text pixels).

## Architecture

- **Backend (`backend/`)**: Python **FastAPI** + async batch worker + SSE live progress
  - **OCR & Block Grouping**: `rapidocr-onnxruntime` (ONNX PaddleOCR) with multi-line grouping for meme captions & speech bubbles
  - **Tight Stroke Masking & Style Analysis**: Samples foreground color, meme stroke/outline, rotation angle, alignment, and uniform speech-bubble expansion bounds
  - **Inpainting**: Patch-based full-resolution **LaMa ONNX** (`Carve/LaMa-ONNX`, auto-downloaded & cached in `backend/.cache/models/`)
  - **Pluggable API Translation**:
    - `deepseek`: **DeepSeek API** context-aware batch translator — ideal for Japanese/Chinese → English comics & memes (`PHOTO_TRANSLATE_DEEPSEEK_API_KEY`)
    - `addis`: **Addis AI** adapter specialized for Amharic, Afaan Oromo, Tigrinya & English (`PHOTO_TRANSLATE_ADDIS_API_KEY`)
    - `google`: Zero-config **Google Translate** adapter (default fallback, no API key required)
    - `deepl`: **DeepL API** adapter (`PHOTO_TRANSLATE_DEEPL_API_KEY`)
  - **Typography Rendering**: Pillow + **HarfBuzz/FriBidi (`libraqm`)** + Noto fonts for Latin, Ethiopic/Amharic, Arabic/Hebrew RTL, CJK, Cyrillic, Devanagari, Thai, etc.
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

> **First run note:** The LaMa ONNX inpainting model (~208 MB) is automatically downloaded into `backend/.cache/models/` on your first translation and cached locally.

## Environment Variables (Optional)

| Variable | Default | Description |
| --- | --- | --- |
| `PHOTO_TRANSLATE_JOB_TTL_SECONDS` | `3600` | Auto-expiry time for temporary job files |
| `PHOTO_TRANSLATE_DEEPSEEK_API_KEY` | *(empty)* | Enables DeepSeek context-aware batch translator when set |
| `PHOTO_TRANSLATE_DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek model name |
| `PHOTO_TRANSLATE_ADDIS_API_KEY` | *(empty)* | Enables Addis AI adapter (`am`, `om`, `ti`, `en`, `sw`) when set |
| `PHOTO_TRANSLATE_ADDIS_API_URL` | `https://api.addisassistant.com/api/v1/translate` | Addis AI translation endpoint |
| `PHOTO_TRANSLATE_DEEPL_API_KEY` | *(empty)* | Enables DeepL adapter when set |
