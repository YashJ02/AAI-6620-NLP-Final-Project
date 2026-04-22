# Health Blood Report Analyzer and Recommendation Engine

NLP Final Project (AAI 6620)

Team: Om Patel, Yash Jain, Ruthvik Bandari

An end-to-end clinical NLP system that ingests blood reports, extracts biomarker entities, interprets findings against reference ranges, and produces explainable food and lifestyle recommendations.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Repository Layout](#repository-layout)
- [Installation and Setup](#installation-and-setup)
- [Quickstart Commands](#quickstart-commands)
- [Model Architecture and Mathematical Algorithms](#model-architecture-and-mathematical-algorithms)
  - [NER Model (Clinical Entity Extraction)](#1-ner-model-clinical-entity-extraction)
  - [Lexical Retrieval (TF-IDF)](#2-lexical-retrieval-tf-idf-baseline)
  - [Semantic Retrieval (Embeddings and FAISS)](#3-semantic-retrieval-embeddings-and-faiss)
  - [Weighted Fusion Ranking](#4-score-fusion-and-global-ranking)
  - [Evaluation Metrics Formulation](#5-evaluation-metrics-and-formulas)
- [Current Project Status](#current-project-status)
- [Metrics and Validation Summary](#metrics-and-validation-summary)
- [Model Artifacts and Git LFS](#model-artifacts-and-git-lfs)
- [Notes](#notes)

## Overview

This project provides a practical pipeline for medical report understanding with the following goals:

- Robust extraction from both digital PDFs and scanned reports (OCR fallback)
- Biomedical named-entity recognition using PubMedBERT
- Clinical interpretation from extracted observations plus reference ranges
- Retrieval-augmented recommendation generation
- API-first deployment with a frontend integration path

## Key Features

- Hybrid extraction stack: direct text extraction with OCR fallback when needed
- Domain-adapted NER: PubMedBERT token classification with BIO labels
- Multi-stage interpretation: rule-based and classifier-assisted decision flow
- Retrieval options: TF-IDF and semantic retrieval pipelines
- Service interfaces: FastAPI endpoints plus scriptable CLI workflows
- Evaluation utilities: repeatable benchmarks for NER and retrieval quality

## System Architecture

```text
Input Report (PDF/Image)
  |
  v
Document Extraction (digital parser + OCR fallback)
  |
  v
NER (PubMedBERT token classification)
  |
  v
Observation Normalization and Interpretation
  |
  v
Recommendation Retrieval (TF-IDF / Semantic / Hybrid Fusion)
  |
  v
Template Generation + API Response
```

## Repository Layout

| Path | Purpose |
|---|---|
| `src/` | Core implementation modules (`api`, `extraction`, `ner`, `interpretation`, `recommendation`) |
| `scripts/` | CLI entry points for training, inference, evaluation, and pipeline orchestration |
| `configs/` | YAML configuration for training/evaluation workflows |
| `data/` | Raw, interim, annotation, and processed datasets |
| `artifacts/` | Model checkpoints, run outputs, metrics, and samples |
| `frontend-next/` | Next.js + TypeScript frontend |
| `tests/` | Regression and integration tests |
| `docs/` | Supporting technical docs and project playbooks |

## Installation and Setup

### Backend (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend-next
bun install
bun run dev
```

### Frontend validation checks

```bash
cd frontend-next
bun run check
bun run build
```

## Quickstart Commands

### 1) Run extraction

```bash
python scripts/run_extraction.py --input data/raw/pdfs_digital/sample.pdf
```

### 2) Build NER train/val/test splits

```bash
python scripts/run_annotation_export.py --input-dir data/annotations/label_studio_exports --output-dir data/processed
```

### 3) Train PubMedBERT NER

```bash
python scripts/run_ner_training.py --config configs/ner_train.yaml --data-dir data/processed --output-dir artifacts/models/pubmedbert_ner
```

### 4) Run end-to-end pipeline

```bash
python scripts/run_pipeline.py --input data/raw/pdfs_digital/sample.pdf --model-dir artifacts/models/pubmedbert_ner/model --output artifacts/sample_outputs/pipeline_output.json
```

### 5) Build retrieval indexes

```bash
python scripts/build_indexes.py
```

### 6) Run evaluation

```bash
python scripts/run_evaluation.py --data-dir data/processed --model-dir artifacts/models/pubmedbert_ner/model --retrieval-benchmark data/processed/retrieval_eval.jsonl --top-k 5 --output artifacts/metrics/evaluation_metrics.json
```

## Model Architecture and Mathematical Algorithms

This section documents the implemented model stack and the mathematical logic used during training, retrieval, and evaluation.

### 1) NER Model (Clinical Entity Extraction)

- Base encoder: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- Task head: `AutoModelForTokenClassification`
- Labeling scheme: BIO tags, including entities like `BIOMARKER`, `VALUE`, `UNIT`, `REFERENCE_RANGE`
- Inference strategy: chunked inference with overlap and span aggregation

For each token representation $h_t$ from PubMedBERT, logits are produced by a linear layer and then converted to a class distribution. This formulation is logically correct.

$$
z_t = W h_t + b, \quad p_t = \mathrm{softmax}(z_t)
$$

Training objective (masked token-level cross-entropy, ignoring labels set to `-100`):

$$
\mathcal{L}_{\mathrm{NER}} = -\frac{1}{N} \sum_{t=1}^{T} \mathbf{1}[y_t \neq -100] \log p_t(y_t)
$$

Configured optimization schedule:

- Learning rate: `2e-5`
- Weight decay: `0.02`
- Scheduler: linear
- Warmup steps: `500`
- Early stopping: enabled with patience and threshold

### 2) Lexical Retrieval (TF-IDF Baseline)

The lexical retriever uses `TfidfVectorizer` with `ngram_range=(1,2)` and English stop-word filtering.

Documents and query vectors are ranked by cosine similarity:

$$
\mathrm{score}_{\mathrm{tfidf}}(q, d) = \frac{v_q \cdot v_d}{\lVert v_q \rVert_2 \lVert v_d \rVert_2}
$$

### 3) Semantic Retrieval (Embeddings and FAISS)

- Embedding model: `all-MiniLM-L6-v2` (Sentence-Transformers)
- ANN backend: FAISS `IndexFlatIP`
- Embeddings are L2-normalized before indexing and search

With normalized vectors, inner product equals cosine similarity:

$$
\mathrm{score}_{\mathrm{semantic}}(q, d) = q^\top d = \cos(\theta_{q,d})
$$

### 4) Score Fusion and Global Ranking

Semantic and lexical candidates are merged and re-ranked with weighted score fusion:

- Semantic weight: `0.6`
- TF-IDF weight: `0.4`

$$
\mathrm{combined}(d) = \sum_{m \in \{\mathrm{semantic},\mathrm{tfidf}\}} \alpha_m \cdot s_m(d)
$$

where $\alpha_{\mathrm{semantic}} = 0.6$ and $\alpha_{\mathrm{tfidf}} = 0.4$.

### 5) Evaluation Metrics and Formulas

NER metrics:

- Token accuracy
- Entity-token precision
- Entity-token recall
- Entity-token F1

$$
\mathrm{Precision} = \frac{TP}{TP+FP}, \quad
\mathrm{Recall} = \frac{TP}{TP+FN}, \quad
F1 = \frac{2PR}{P+R}
$$

Retrieval metrics at rank $k$:

$$
\mathrm{Precision@k} = \frac{|\mathrm{relevant} \cap \mathrm{topk}|}{k}, \quad
\mathrm{Recall@k} = \frac{|\mathrm{relevant} \cap \mathrm{topk}|}{|\mathrm{relevant}|}
$$

Mean Reciprocal Rank:

$$
\mathrm{MRR@k} = \frac{1}{Q} \sum_{i=1}^{Q} \frac{1}{\mathrm{rank}_i}
$$

where $\mathrm{rank}_i$ is the first relevant-hit position for query $i$ (or 0 if no relevant item appears in top-$k$).

## Current Project Status

Completed:

- End-to-end extraction to recommendation pipeline
- PubMedBERT NER training and inference utilities
- FastAPI endpoints for extraction, NER, interpretation, recommendation, and pipeline orchestration
- Frontend integration plus successful local build checks

In progress:

- OCR quality improvements for noisy scanned reports
- Retrieval benchmarking and relevance tuning
- Expanded production-oriented smoke testing

## Metrics and Validation Summary

This section intentionally consolidates metrics into one canonical view to avoid duplication.

### Latest verified local evaluation

Source: `artifacts/metrics/evaluation_metrics.json`

| Metric | Value |
|---|---:|
| NER token accuracy | `0.9237` |
| NER entity-token precision | `0.9615` |
| NER entity-token recall | `0.8254` |
| NER entity-token F1 | `0.8883` |
| Retrieval precision@5 | `0.5867` |
| Retrieval recall@5 | `0.9854` |
| Retrieval MRR@5 | `0.9958` |

### 5-fold cross-validation summary

Source: `artifacts/runs/cv5/summary.json`

| Metric | Value |
|---|---:|
| Entity-token F1 mean | `0.8902` |
| Entity-token F1 std | `0.0034` |
| Precision mean | `0.9561` |
| Recall mean | `0.8329` |
| Token accuracy mean | `0.9235` |

### HPC NER log snapshot

Source logs: `logs/ner_pubmedbert_*.out`

| Run | Token Acc | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Best (`ner_pubmedbert_5985702.out`) | `0.9267` | `0.9535` | `0.8446` | `0.8958` |
| Most recent (`ner_pubmedbert_5986331.out`) | `0.9242` | `0.9547` | `0.8369` | `0.8919` |

Note: Retrieval values in older HPC logs (`0.0417/0.0771/0.1833` for precision/recall/MRR@5) were computed under a previous benchmark setup. Prefer the local verified retrieval metrics above for current comparisons.

## Model Artifacts and Git LFS

Model weights are versioned with Git LFS under:

- `artifacts/models/pubmedbert_ner/model/`

Fetch model files after clone:

```bash
git lfs install
git lfs pull
```

## Notes

- CLI scripts are runnable from repository root (manual `PYTHONPATH` export is not required).
- Semantic retrieval gracefully falls back to TF-IDF if semantic dependencies are unavailable at runtime.

