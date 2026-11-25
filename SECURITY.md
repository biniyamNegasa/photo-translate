# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in **Photo Translate**, please **do not** open a public issue.

1. Use GitHub's private **Security Advisories** ("Report a vulnerability") feature for this repository, or contact the maintainer directly via private channels listed on the repository profile.
2. Include a clear description of the issue, reproduction steps or proof-of-concept, affected files/versions, and potential impact.
3. We aim to acknowledge reports within 72 hours and coordinate remediation and disclosure once a fix is verified.

## Threat Model & Intended Deployment

Photo Translate is designed as a **localhost-first** desktop/workstation web application:

- The FastAPI backend binds to `127.0.0.1:8000` by default and processes user-supplied images and `.zip` archives locally using ONNX models and Pillow/OpenCV.
- Uploaded images and rendered outputs are stored ephemerally in `backend/.cache/jobs/` and expire automatically after `PHOTO_TRANSLATE_JOB_TTL_SECONDS` (default: 1 hour).
- Optional external translation APIs (Google Translate, DeepSeek, Addis AI, DeepL) receive only extracted OCR text snippets—never raw uploaded images.

## Security Controls & Hardening

### Local Workstation Defaults
- Keep `backend/.env` restricted to owner-only permissions (`chmod 600 backend/.env`) and `backend/.cache/` restricted to `chmod 700`.
- Keep the backend bound to `127.0.0.1` unless explicitly deploying behind an authenticated reverse proxy.
- Client-side and server-side validation enforce allowlisted image/ZIP formats, maximum upload size, image pixel/decompression limits, ZIP bomb bounds, and strict job/item identifier validation.
- The frontend enforces a strict Content Security Policy (with per-response nonces), `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, and `Permissions-Policy`.

### Production / Networked Deployments
If you expose Photo Translate beyond `localhost`, you **must** apply all of the following controls:
1. **Disable dev reload:** Never run `uvicorn --reload` or `./start-dev.sh` in production.
2. **Require authentication:** Set a high-entropy `PHOTO_TRANSLATE_API_KEY` (and/or enforce authentication at your reverse proxy) and disable API docs (`PHOTO_TRANSLATE_ENABLE_DOCS=false`).
3. **Restrict CORS origins:** Configure `PHOTO_TRANSLATE_CORS_ORIGINS` explicitly to your production frontend origin(s) only—never `*`.
4. **Terminate TLS & rate-limit at a reverse proxy:** Serve both frontend and backend over HTTPS behind Nginx, Caddy, or a cloud load balancer with request body size caps, per-IP rate limiting, and connection timeouts.
5. **Tighten resource limits & retention:** Lower `PHOTO_TRANSLATE_MAX_UPLOAD_MB`, `PHOTO_TRANSLATE_MAX_BATCH_IMAGES`, `PHOTO_TRANSLATE_WORKER_CONCURRENCY`, and `PHOTO_TRANSLATE_JOB_TTL_SECONDS` to fit your host capacity and privacy requirements.
6. **Keep dependencies updated:** Regularly update Python dependencies (`backend/requirements.txt`) and Node dependencies (`frontend/package.json`) and scan lockfiles for known advisories.
