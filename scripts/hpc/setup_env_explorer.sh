#!/usr/bin/env bash
set -euo pipefail

# Run from repository root on Northeastern Explorer login node.
# Usage: bash scripts/hpc/setup_env_explorer.sh

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Print quick environment summary.
python - <<'PY'
import sys
import torch
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY

echo "[OK] Explorer environment is ready in $VENV_DIR"
