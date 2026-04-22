# Health Blood Report Analyzer and Recommendation Engine

NLP Final Project (AAI 6620)

Team: Om Patel, Yash Jain, Ruthvik Bandari

## Overview

This project provides an end-to-end pipeline that reads blood report documents, extracts clinical entities, interprets biomarker values against reference ranges, and generates explainable food and lifestyle recommendations.

The stack includes:

- Hybrid document extraction (digital PDF and OCR fallback)
- PubMedBERT-based NER with biomedical BIO labels
- Rule-based and classifier-assisted interpretation
- Recommendation retrieval (TF-IDF and semantic options)
- FastAPI backend and Next.js frontend

## Current Status

Completed:

- End-to-end pipeline from extraction to recommendations
- PubMedBERT training and inference scripts
- FastAPI endpoints for extraction, NER, interpretation, recommendation, and full pipeline
- Frontend integration and validated local build checks

In progress:

- OCR quality improvements for scanned reports
- Continued retrieval benchmarking and tuning
- Production smoke-test coverage expansion

## Repository Layout

Key folders:

- `src/` core modules (extraction, ner, interpretation, recommendation, api)
- `scripts/` runnable CLI entry points for each pipeline stage
- `configs/` training and pipeline configuration files
- `data/` raw, interim, annotations, and processed datasets
- `artifacts/` saved models, metrics, run outputs, sample outputs
- `frontend-next/` Next.js + TypeScript frontend
- `tests/` API and pipeline regression tests
- `docs/` design, API contract, and playbooks

## Setup

Backend (Windows):

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Frontend:

```bash
cd frontend-next
bun install
bun run dev
```

Frontend checks:

```bash
cd frontend-next
bun run check
bun run build
```

## Quickstart Commands

Run extraction:

```bash
python scripts/run_extraction.py --input data/raw/pdfs_digital/sample.pdf
```

Create NER train/val/test splits from Label Studio exports:

```bash
python scripts/run_annotation_export.py --input-dir data/annotations/label_studio_exports --output-dir data/processed
```

Train PubMedBERT NER:

```bash
python scripts/run_ner_training.py --config configs/ner_train.yaml --data-dir data/processed --output-dir artifacts/models/pubmedbert_ner
```

Run one-command pipeline:

```bash
python scripts/run_pipeline.py --input data/raw/pdfs_digital/sample.pdf --model-dir artifacts/models/pubmedbert_ner/model --output artifacts/sample_outputs/pipeline_output.json
```

Build retrieval indexes:

```bash
python scripts/build_indexes.py
```

Run evaluation:

```bash
python scripts/run_evaluation.py --data-dir data/processed --model-dir artifacts/models/pubmedbert_ner/model --retrieval-benchmark data/processed/retrieval_eval.jsonl --top-k 5 --output artifacts/metrics/evaluation_metrics.json
```

## Model Architecture and Mathematical Algorithms

This section documents the core modeling stack and mathematical methods implemented in this repository.

### 1) NER Model Architecture (Clinical Entity Extraction)

- Base encoder: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- Task head: token-classification head (`AutoModelForTokenClassification`)
- Labeling scheme: BIO tags with entity classes (for example `BIOMARKER`, `VALUE`, `UNIT`, `REFERENCE_RANGE`)
- Training split source: processed JSONL token-tag rows
- Inference strategy: chunked text inference with overlap and aggregated token spans

For each token representation `h_t` from PubMedBERT, logits are computed by a linear classifier and converted to probabilities:

$$
z_t = W h_t + b, \quad p_t = \operatorname{softmax}(z_t)
$$

Training minimizes token-level cross-entropy over valid tokens (special/padding/subword-ignored tokens use label `-100`):

$$
\mathcal{L}_{\mathrm{NER}} = -\frac{1}{N} \sum_{t=1}^{T} \mathbf{1}[y_t \neq -100] \log p_t(y_t)
$$

Optimization and schedule (from training config):

- Learning rate: `2e-5`
- Weight decay: `0.02`
- Scheduler: linear
- Warmup steps: `500`
- Early stopping with patience and threshold

### 2) Lexical Retrieval Algorithm (TF-IDF Baseline)

The lexical retriever uses `TfidfVectorizer` with uni-grams and bi-grams (`ngram_range=(1,2)`) and English stop-word filtering.

Each document and query are embedded in sparse TF-IDF space and ranked by cosine similarity:

$$
\mathrm{score}_{\mathrm{tfidf}}(q, d) = \frac{v_q \cdot v_d}{\lVert v_q \rVert_2 \lVert v_d \rVert_2}
$$

Top-k documents are selected by descending cosine score.

### 3) Semantic Retrieval Algorithm (Embeddings + FAISS)

- Embedding model: `all-MiniLM-L6-v2` (Sentence-Transformers)
- Index: FAISS `IndexFlatIP` (inner-product search)
- Embeddings are L2-normalized before indexing and querying

With normalized vectors, inner product equals cosine similarity:

$$
\mathrm{score}_{\mathrm{semantic}}(q, d) = q^\top d = \cos(\theta_{q,d})
$$

The retriever performs nearest-neighbor search in embedding space and returns top-k semantic matches.

### 4) Score Fusion and Global Ranking

TF-IDF and semantic candidates are merged and re-ranked with weighted score fusion:

- Semantic weight: `0.6`
- TF-IDF weight: `0.4`

For candidate `d`, combined score is:

$$
\mathrm{combined}(d) = \sum_{m \in \{\mathrm{semantic},\mathrm{tfidf}\}} \alpha_m \cdot s_m(d)
$$

where $\alpha_{\mathrm{semantic}} = 0.6$ and $\alpha_{\mathrm{tfidf}} = 0.4$.

### 5) Evaluation Metrics and Formulas

NER token/entity metrics:

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

Mean Reciprocal Rank (MRR@k):

$$
\mathrm{MRR@k} = \frac{1}{Q} \sum_{i=1}^{Q} \frac{1}{\mathrm{rank}_i}
$$

where $\mathrm{rank}_i$ is the first relevant-hit position for query $i$ (or 0 contribution if no hit in top-k).

## Metrics (Cleaned and De-duplicated)

This section replaces overlapping historical metric blocks and uses explicit sources.

### Latest Verified Evaluation (Local)

Source: `artifacts/metrics/evaluation_metrics.json`

- NER token accuracy: `0.9237`
- NER entity-token precision: `0.9615`
- NER entity-token recall: `0.8254`
- NER entity-token F1: `0.8883`
- Retrieval precision@5: `0.5867`
- Retrieval recall@5: `0.9854`
- Retrieval MRR@5: `0.9958`

### 5-Fold Cross-Validation Summary

Source: `artifacts/runs/cv5/summary.json`

- Entity-token F1 mean: `0.8902`
- Entity-token F1 std: `0.0034`
- Precision mean: `0.9561`
- Recall mean: `0.8329`
- Token accuracy mean: `0.9235`

### Training Log Snapshot (HPC NER Runs)

Source logs: `logs/ner_pubmedbert_*.out` (final metrics blocks)

- Best logged NER run: `ner_pubmedbert_5985702.out`
  - Token accuracy: `0.9267`
  - Entity-token precision: `0.9535`
  - Entity-token recall: `0.8446`
  - Entity-token F1: `0.8958`
- Most recent logged NER run: `ner_pubmedbert_5986331.out`
  - Token accuracy: `0.9242`
  - Entity-token precision: `0.9547`
  - Entity-token recall: `0.8369`
  - Entity-token F1: `0.8919`

Note: Retrieval values inside these HPC training logs are consistently `0.0417/0.0771/0.1833` for precision/recall/MRR@5 and represent an older benchmark setup. The local verified retrieval metrics above come from the latest unified evaluation artifact.

## Model Artifacts and Git LFS

Model weights are stored with Git LFS under:

- `artifacts/models/pubmedbert_ner/model/`

Fetch model files after clone:

```bash
git lfs install
git lfs pull
```

## Notes

- CLI scripts are runnable from repository root (manual `PYTHONPATH` setup is not required).
- Semantic retrieval degrades gracefully to TF-IDF when semantic runtime dependencies are unavailable.
