#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

free_port() {
  local port="$1"
  local pids
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "Stopping stale process on port $port (PID: ${pids//$'\n'/, }) ..."
    kill $pids 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      sleep 0.3
      pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
      [[ -z "$pids" ]] && return 0
    done
    kill -9 $pids 2>/dev/null || true
    sleep 0.2
  fi
}

cleanup() {
  trap - INT TERM EXIT
  kill 0 2>/dev/null || true
}
trap cleanup INT TERM EXIT

free_port 8000
free_port 3000

echo "Starting FastAPI backend on http://127.0.0.1:8000 ..."
(
  cd "$ROOT_DIR/backend"
  exec "$ROOT_DIR/backend/.venv/bin/python" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
) &

echo "Starting Farm.js frontend on http://localhost:3000 ..."
(
  cd "$ROOT_DIR/frontend"
  exec npm run dev
) &

wait
