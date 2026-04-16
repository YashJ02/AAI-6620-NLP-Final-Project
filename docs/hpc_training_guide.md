# Northeastern Explorer HPC Training Guide

This guide covers end-to-end retraining on Explorer, including data refresh and Slurm training.

## 1) Copy project to Explorer

From local machine:

rsync -av --progress --exclude ".git" --exclude ".venv" /Users/ruthvikbandari/Desktop/NLP/AAI-6620-NLP-Final-Project/ bandari.ru@login.explorer.northeastern.edu:~/AAI-6620-NLP-Final-Project/

## 2) SSH to Explorer

ssh bandari.ru@login.explorer.northeastern.edu

## 3) Prepare Python environment (first time)

cd ~/AAI-6620-NLP-Final-Project
bash scripts/hpc/setup_env_explorer.sh

## 4) Prepare training data on Explorer

Populate `data/raw/sources_manifest.csv` first, then run:

cd ~/AAI-6620-NLP-Final-Project
bash scripts/hpc/prepare_data_explorer.sh

This runs:
- dataset download from manifest
- normalization + cleaning
- synthetic NER split regeneration
- readiness synthesis

## 5) Submit H200 training job

cd ~/AAI-6620-NLP-Final-Project
sbatch scripts/hpc/train_ner_explorer.slurm

## 6) Monitor status

squeue -u bandari.ru

## 7) Inspect outputs

- Slurm logs: logs/
- Model: artifacts/models/pubmedbert_ner/model/
- Label mapping: artifacts/models/pubmedbert_ner/label_mapping.json
- Metrics: artifacts/metrics/evaluation_metrics.json
- Data prep report: artifacts/metrics/download_sources_report.json

## 8) Copy artifacts back to local machine

From local machine:

rsync -av --progress bandari.ru@login.explorer.northeastern.edu:~/AAI-6620-NLP-Final-Project/artifacts/ /Users/ruthvikbandari/Desktop/NLP/AAI-6620-NLP-Final-Project/artifacts/

## Notes

- If Explorer uses a different partition/QoS/account, update Slurm directives in `scripts/hpc/train_ner_explorer.slurm`.
- Training now supports warmup, early stopping, and deterministic `seed` from `configs/ner_train.yaml`.
