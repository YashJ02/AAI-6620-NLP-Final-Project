#!/usr/bin/env bash
#SBATCH --job-name=ner_pubmedbert
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Submit from repository root:
#   sbatch scripts/hpc/train_ner_explorer.slurm

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs artifacts/models artifacts/metrics

if command -v module >/dev/null 2>&1; then
  PYTHON_MODULE="${PYTHON_MODULE:-python/3.13.5}"
  module load "$PYTHON_MODULE" || true
fi

# Activate virtual environment created by setup_env_explorer.sh
source .venv/bin/activate

# Ensure src package is importable
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

# Helpful runtime info
echo "=== GPU Info ==="
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU memory:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
PY

echo ""
echo "=== Training Config ==="
cat configs/ner_train.yaml
echo ""

# Count training data
echo "=== Data Stats ==="
wc -l data/processed/train.jsonl data/processed/val.jsonl data/processed/test.jsonl
echo ""

# Train NER model
echo "=== Starting Training ==="
python scripts/run_ner_training.py \
  --config configs/ner_train.yaml \
  --data-dir data/processed \
  --output-dir artifacts/models/pubmedbert_ner

# Evaluate NER on test set
echo ""
echo "=== Running Evaluation ==="
python scripts/run_evaluation.py \
  --data-dir data/processed \
  --model-dir artifacts/models/pubmedbert_ner/model \
  --retrieval-benchmark data/processed/retrieval_eval.jsonl \
  --top-k 5 \
  --output artifacts/metrics/evaluation_metrics.json

echo ""
echo "=== Evaluation Results ==="
cat artifacts/metrics/evaluation_metrics.json

echo ""
echo "[OK] Training + evaluation finished"
