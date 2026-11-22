#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

trap 'kill 0' EXIT

echo "Starting FastAPI backend on http://127.0.0.1:8000 ..."
(
  cd "$ROOT_DIR/backend"
  "$ROOT_DIR/backend/.venv/bin/python" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
) &

echo "Starting Farm.js frontend on http://localhost:3000 ..."
(
  cd "$ROOT_DIR/frontend"
  npm run dev
) &

wait
