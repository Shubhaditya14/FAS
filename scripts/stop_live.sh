#!/usr/bin/env bash
set -euo pipefail
# Stop the FastAPI service started by run_live.sh.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PID_FILE="${ROOT_DIR}/.fas_web.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "No PID file found. If the service is running, stop it manually."
  exit 0
fi

PID="$(cat "$PID_FILE")"

if ps -p "$PID" >/dev/null 2>&1; then
  echo "Stopping service (pid $PID)..."
  kill "$PID" || true
  rm -f "$PID_FILE"
  echo "✔ Service stopped."
else
  echo "Process $PID not running. Cleaning up PID file."
  rm -f "$PID_FILE"
fi
