#!/usr/bin/env bash
set -euo pipefail

# Run from repository root on Explorer after environment setup.
# Usage:
#   bash scripts/hpc/prepare_data_explorer.sh

cd "$(dirname "$0")/../.."
source .venv/bin/activate
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "=== Step 1-2: Downloading datasets from manifest ==="
python scripts/download_sources_from_manifest.py \
  --manifest data/raw/sources_manifest.csv \
  --output-root datasets/external \
  --report artifacts/metrics/download_sources_report.json

echo "=== Step 3-6: Normalize + clean + regenerate NER splits ==="
python scripts/normalize_datasets.py --datasets-dir datasets --project-root .
python scripts/clean_observations.py
python scripts/generate_synthetic_ner_data.py \
  --observations-csv data/interim/normalized_records/biomarker_observations_clean.csv \
  --ranges-csv knowledge_base/reference_ranges/biomarker_reference_ranges.csv \
  --output-dir data/processed \
  --max-examples 20000 \
  --seed 42 \
  --train-ratio 0.80 \
  --val-ratio 0.10 \
  --test-ratio 0.10

echo "=== Readiness refresh ==="
python scripts/synthesize_readiness_assets.py

echo "[OK] Data preparation complete on Explorer"
