# Retraining Playbook (Data-First)

This playbook follows a strict sequence to avoid noisy training.

## 1) Find correct datasets

Use only sources with clear license + schema relevance to:
- CBC / blood chemistry biomarkers
- units + reference ranges
- optional report text/PDFs for extraction

Track all candidates in data/raw/sources_manifest.csv.

Required columns:
- source_name
- source_url
- license
- notes

Optional columns:
- destination_relpath
- sha256

## 2) Download datasets

```bash
python scripts/download_sources_from_manifest.py --manifest data/raw/sources_manifest.csv --output-root datasets/external --report artifacts/metrics/download_sources_report.json
```

## 3-4) Analyze and understand data

```bash
python scripts/normalize_datasets.py --datasets-dir datasets --project-root .
```

Review:
- artifacts/metrics/data_position_report.md
- data/interim/normalized_records/dataset_inventory.csv

## 5) Clean data

```bash
python scripts/clean_observations.py
```

Review:
- data/interim/normalized_records/cleaning_report.json

## 6) Analyze again

```bash
python scripts/synthesize_readiness_assets.py
```

## 7-8) Hyperparameters + split/seed control

Generate split with explicit ratios:

```bash
python scripts/generate_synthetic_ner_data.py --observations-csv data/interim/normalized_records/biomarker_observations_clean.csv --ranges-csv knowledge_base/reference_ranges/biomarker_reference_ranges.csv --output-dir data/processed --max-examples 20000 --seed 42 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

Tune in configs/ner_train.yaml:
- seed
- learning_rate
- batch_size + gradient_accumulation_steps
- warmup_ratio / warmup_steps
- early_stopping.patience

## 9-10) Early stopping + warmup

Already enabled via configs/ner_train.yaml and src/ner/train_pubmedbert.py:
- warmup schedule before full learning rate
- early stopping on best validation metric

## 11) Method

Recommended current method:
- PubMedBERT token classification
- model selected by validation entity F1 (`entity_f1`)

Experiment pattern:
1. keep split fixed, tune hyperparameters
2. keep best hyperparameters, sweep split ratios + seed

## 12) HPC run (Explorer)

```bash
bash scripts/hpc/setup_env_explorer.sh
bash scripts/hpc/prepare_data_explorer.sh
sbatch scripts/hpc/train_ner_explorer.slurm
```

## 13) Code changes

All changes should be committed with reason in commit message and reflected in docs.

## 14) Push updates

```bash
git add .
git commit -m "<message>"
git push origin main
```

## 15) Safety checks

Before each training run:
- verify manifest licenses
- check cleaning_report.json removal reasons
- verify split counts and label distribution
- verify no leakage across train/val/test

## 16) Legitimate accuracy target

Use validation/test metrics that are robust against noisy positives:
- token_accuracy
- entity_token_precision
- entity_token_recall
- entity_token_f1

Pick model by validation entity F1, not by train loss alone.
