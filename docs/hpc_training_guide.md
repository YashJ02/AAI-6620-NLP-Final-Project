# Northeastern Explorer HPC Training Guide

This guide runs NER training for this project on Explorer with Slurm.

## 1) Copy project to Explorer
From your local machine, sync project once:

rsync -av --progress --exclude ".git" --exclude ".venv" /Users/ruthvikbandari/Desktop/NLP/nlp-final/ Bandari.ru@login.explorer.northeastern.edu:~/nlp-final/

## 2) SSH to Explorer
ssh Bandari.ru@login.explorer.northeastern.edu

## 3) Prepare Python environment (first time only)

cd ~/nlp-final
bash scripts/hpc/setup_env_explorer.sh

## 4) Submit GPU training job

cd ~/nlp-final
sbatch scripts/hpc/train_ner_explorer.slurm

## 5) Monitor status

squeue -u Bandari.ru

## 6) Inspect logs and outputs

- Slurm logs: `logs/`
- Model: `artifacts/models/pubmedbert_ner/model/`
- Label mapping: `artifacts/models/pubmedbert_ner/label_mapping.json`
- Metrics: `artifacts/metrics/evaluation_metrics.json`

## 7) Copy artifacts back to local machine
From your local machine:

rsync -av --progress Bandari.ru@login.explorer.northeastern.edu:~/nlp-final/artifacts/ /Users/ruthvikbandari/Desktop/NLP/nlp-final/artifacts/

## Notes
- Current dataset sizes in this repo are:
  - `data/processed/train.jsonl`: 9600 rows
  - `data/processed/val.jsonl`: 1200 rows
  - `data/processed/test.jsonl`: 1200 rows
- If your account uses a different GPU partition name, update `--partition=gpu` in the Slurm script.
