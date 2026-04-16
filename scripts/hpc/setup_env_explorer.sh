#!/usr/bin/env bash
set -euo pipefail

# Run from repository root on Northeastern Explorer login node.
# Usage: bash scripts/hpc/setup_env_explorer.sh

cd "$(dirname "$0")/../.."

if command -v module >/dev/null 2>&1; then
  PYTHON_MODULE="${PYTHON_MODULE:-python/3.13.5}"
  module load "$PYTHON_MODULE" || true
fi

choose_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "$PYTHON_BIN"
    return 0
  fi

  for cand in python3.11 python3.10 python3; do
    if ! command -v "$cand" >/dev/null 2>&1; then
      continue
    fi
    if "$cand" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
    then
      echo "$cand"
      return 0
    fi
  done

  return 1
}

if ! PYTHON_BIN="$(choose_python_bin)"; then
  echo "[ERROR] Python >= 3.10 is required. Set PYTHON_BIN to a compatible interpreter." >&2
  echo "[ERROR] Example: PYTHON_BIN=python3.10 bash scripts/hpc/setup_env_explorer.sh" >&2
  exit 1
fi

echo "[INFO] Using Python interpreter: $PYTHON_BIN"
VENV_DIR="${VENV_DIR:-.venv}"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements-hpc.txt

# Print quick environment summary.
python - <<'PY'
import sys
import torch
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY

echo "[OK] Explorer environment is ready in $VENV_DIR"
