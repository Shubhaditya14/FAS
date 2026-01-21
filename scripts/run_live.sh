#!/usr/bin/env bash
set -euo pipefail

# Launch FastAPI service for live anti-spoofing and open the UI.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

# Load .env if present (SMTP/OTP, behavior flags, DB, etc.)
if [ -f ".env" ]; then
  set -o allexport
  # shellcheck disable=SC1091
  source .env
  set +o allexport
fi

VENV_DIR="${ROOT_DIR}/venv"
PID_FILE="${ROOT_DIR}/.fas_web.pid"
PORT="${PORT:-8000}"
export FAS_PROB_MODE="${FAS_PROB_MODE:-spoof}"
export MONGO_URI="${MONGO_URI:-mongodb://localhost:27017}"
export MONGO_DB="${MONGO_DB:-face}"

if [ ! -d "$VENV_DIR" ]; then
  echo "Virtualenv missing. Run scripts/setup_env.sh first."
  exit 1
fi

source "$VENV_DIR/bin/activate"

# Quick Mongo reachability check (warn only)
python3 - <<'PY' || echo "⚠️ Mongo check failed (continuing). Set MONGO_URI if needed."
import os
from pymongo import MongoClient
uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
try:
    client = MongoClient(uri, serverSelectionTimeoutMS=1000)
    client.admin.command("ping")
    print("✅ Mongo reachable at", uri)
except Exception as exc:  # noqa: BLE001
    print("⚠️ Mongo not reachable:", exc)
PY

if [ -f "$PID_FILE" ] && ps -p "$(cat "$PID_FILE")" >/dev/null 2>&1; then
  echo "Service already running with PID $(cat "$PID_FILE"). Stop it first."
  exit 0
fi

echo "Starting FastAPI service on port ${PORT}..."
nohup uvicorn web_service:app --host 0.0.0.0 --port "$PORT" > fas_web.log 2>&1 &
echo $! > "$PID_FILE"

echo "Waiting for service to become ready..."
for _ in $(seq 1 30); do
  if curl -fs "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "✔ Service started (pid $(cat "$PID_FILE")). Logs: fas_web.log"
    echo "Open http://localhost:${PORT}/ui/ for detector or http://localhost:${PORT}/auth for login."
    exit 0
  fi
  sleep 1
done

echo "⚠️ Service did not respond on http://localhost:${PORT}/health within 30s."
echo "Last 40 log lines:"
tail -n 40 fas_web.log || true
exit 1
