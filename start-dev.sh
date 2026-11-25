#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PYTHON="$ROOT_DIR/backend/.venv/bin/python"
BACKEND_UVICORN="$ROOT_DIR/backend/.venv/bin/uvicorn"

if [[ ! -x "$BACKEND_PYTHON" || ! -x "$BACKEND_UVICORN" ]]; then
  echo "Error: Backend virtualenv not found at $ROOT_DIR/backend/.venv" >&2
  echo "Set it up first with:" >&2
  echo "  cd \"$ROOT_DIR/backend\" && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "Error: 'npm' is not installed or not on PATH. Install Node.js 22.13+ first." >&2
  exit 1
fi

if [[ ! -d "$ROOT_DIR/frontend/node_modules" ]]; then
  echo "Error: Frontend dependencies are not installed in $ROOT_DIR/frontend/node_modules" >&2
  echo "Install them first with:" >&2
  echo "  cd \"$ROOT_DIR/frontend\" && npm install" >&2
  exit 1
fi

is_project_process() {
  local pid="$1"
  local cwd=""
  local cmdline=""

  if [[ -r "/proc/$pid/cwd" ]]; then
    cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
  fi
  if [[ -r "/proc/$pid/cmdline" ]]; then
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  fi

  if [[ -n "$cwd" && ( "$cwd" == "$ROOT_DIR" || "$cwd" == "$ROOT_DIR"/* ) ]]; then
    return 0
  fi
  if [[ -n "$cmdline" && "$cmdline" == *"$ROOT_DIR"* ]]; then
    return 0
  fi
  return 1
}

free_port() {
  local port="$1"
  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi

  local -a pids=()
  mapfile -t pids < <(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
  if (( ${#pids[@]} == 0 )); then
    return 0
  fi

  local -a owned_pids=()
  local pid cmdline
  for pid in "${pids[@]}"; do
    if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
      continue
    fi
    if is_project_process "$pid"; then
      owned_pids+=("$pid")
    else
      cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || echo "<unknown>")"
      echo "Error: Port $port is already in use by an unrelated process (PID $pid: ${cmdline:-<unknown>})." >&2
      echo "Stop that process manually or free port $port before running start-dev.sh." >&2
      exit 1
    fi
  done

  if (( ${#owned_pids[@]} == 0 )); then
    return 0
  fi

  echo "Stopping stale project process on port $port (PID: ${owned_pids[*]}) ..."
  for pid in "${owned_pids[@]}"; do
    kill -- "$pid" 2>/dev/null || true
  done

  local still_running=0
  for _ in 1 2 3 4 5; do
    sleep 0.3
    still_running=0
    for pid in "${owned_pids[@]}"; do
      if kill -0 -- "$pid" 2>/dev/null; then
        still_running=1
        break
      fi
    done
    (( still_running == 0 )) && return 0
  done

  for pid in "${owned_pids[@]}"; do
    if kill -0 -- "$pid" 2>/dev/null && is_project_process "$pid"; then
      kill -9 -- "$pid" 2>/dev/null || true
    fi
  done
  sleep 0.2
}

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  trap - INT TERM EXIT
  local pid
  for pid in "$BACKEND_PID" "$FRONTEND_PID"; do
    if [[ -n "$pid" && "$pid" =~ ^[0-9]+$ ]] && kill -0 -- "$pid" 2>/dev/null; then
      kill -- "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

free_port 8000
free_port 3000

echo "Starting FastAPI backend on http://127.0.0.1:8000 ..."
(
  cd "$ROOT_DIR/backend"
  exec "$BACKEND_PYTHON" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
) &
BACKEND_PID=$!

echo "Starting Farm.js frontend on http://localhost:3000 ..."
(
  cd "$ROOT_DIR/frontend"
  exec npm run dev
) &
FRONTEND_PID=$!

wait
