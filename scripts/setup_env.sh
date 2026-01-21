#!/usr/bin/env bash
set -euo pipefail

# Set up virtualenv and install dependencies.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${ROOT_DIR}/venv"

select_python() {
  local candidates=("python3.12" "python3.11" "python3")
  for candidate in "${candidates[@]}"; do
    if command -v "$candidate" >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

PYTHON_BIN="$(select_python || true)"
if [ -z "$PYTHON_BIN" ]; then
  echo "No suitable python found. Please install Python 3.11 or 3.12."
  exit 1
fi

PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
case "$PY_VERSION" in
  3.11|3.12) ;;
  *)
    echo "Detected Python $PY_VERSION which is unsupported for this project."
    echo "Install Python 3.11 or 3.12 and rerun this script."
    exit 1
    ;;
esac

if [ -x "$VENV_DIR/bin/python" ]; then
  ACTIVE_VENV_VERSION="$("$VENV_DIR/bin/python" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
  if [ "$ACTIVE_VENV_VERSION" != "$PY_VERSION" ]; then
    echo "Recreating virtual environment with Python $PY_VERSION (was $ACTIVE_VENV_VERSION)..."
    rm -rf "$VENV_DIR"
  fi
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment with $PYTHON_BIN..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Installing requirements..."
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

echo "✔ Environment ready."
