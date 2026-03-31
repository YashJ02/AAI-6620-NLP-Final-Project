#!/usr/bin/env bash
#SBATCH --job-name=ner_pubmedbert
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
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

# Activate virtual environment created by setup_env_explorer.sh
source .venv/bin/activate

# Helpful runtime info
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
PY

# Train NER model
python scripts/run_ner_training.py \
  --config configs/ner_train.yaml \
  --data-dir data/processed \
  --output-dir artifacts/models/pubmedbert_ner

# Evaluate NER + retrieval after training
python scripts/run_evaluation.py \
  --data-dir data/processed \
  --model-dir artifacts/models/pubmedbert_ner/model \
  --retrieval-benchmark data/processed/retrieval_eval.jsonl \
  --top-k 5 \
  --output artifacts/metrics/evaluation_metrics.json

echo "[OK] Training + evaluation finished"
