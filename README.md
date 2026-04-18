# Health Blood Report Analyzer & Recommendation Engine

NLP Final Project (AAI 6620)

Team: Om Patel, Yash Jain, Ruthvik Bandari  
Date: February 24, 2026

## Project Overview

This project builds an end-to-end NLP system that reads blood test report PDFs, extracts key medical information, interprets results using reference ranges, and generates practical nutrition and lifestyle recommendations.

Blood reports are hard for patients to understand because labs use different templates, naming conventions, units, and layouts. Our goal is to create one unified pipeline that can process these variations and produce clear, actionable summaries.

## What We Are Building

We are building a multi-stage system with the following components:

1. PDF Ingestion and Text Extraction

- Route digital PDFs to PyMuPDF.
- Route scanned/image-based PDFs to Surya OCR.
- Preserve document structure for rows, tables, and labels.

2. Biomedical Entity Extraction (NER)

- Train a custom NER model (PubMedBERT token classification) on our annotated corpus.
- Extract four entity types with BIO tagging:
  - BIOMARKER
  - VALUE
  - UNIT
  - REFERENCE_RANGE
- Use a SpaCy rule-based fallback for strongly structured report rows.

3. Clinical Value Interpretation

- Normalize biomarker names and units.
- Compare extracted values against curated reference ranges.
- Use rule-based logic for standard cases and an ML classifier for ambiguous edge cases.

4. Recommendation Engine

- Build a health knowledge base from reliable sources (USDA FoodData Central and medical references).
- Compare two retrieval approaches:
  - TF-IDF baseline (scikit-learn)
  - Semantic retrieval with Sentence-Transformers + FAISS
- Generate explainable recommendations using template-based NLG (Jinja2).

5. Product Interface

- FastAPI backend for pipeline services.
- Streamlit frontend for report upload and interpretation display.
- Plotly charts for intuitive result visualization.

## What We Are Achieving

By the end of this project, we aim to deliver:

- A custom annotated corpus of 90+ blood report documents.
- A fine-tuned biomedical NER model for blood report extraction.
- A robust extraction and normalization pipeline that handles multiple report formats.
- A comparison of lexical vs semantic recommendation retrieval quality.
- A functional demo app where users can upload reports and receive understandable health summaries.
- A final technical report and demo video.

## Corpus and Annotation Plan

- Sources:
  - Public lab report templates
  - Kaggle synthetic blood test datasets
  - Team-generated synthetic report PDFs
- Target size: 90+ reports (30 per team member)
- Annotation tool: Label Studio
- Annotation format: BIO tagging with finalized guidelines

## Development Roadmap (8 Weeks)

1. Data collection and corpus preparation
2. Annotation setup and quality checks
3. Extraction pipeline (PDF + OCR routing)
4. NER training and evaluation
5. Classification and interpretation logic
6. Recommendation retrieval and generation
7. Frontend/backend integration
8. Final testing, reporting, and demo delivery

## Phase 1 Quick Start (Extraction)

Run extraction on one PDF or image:

```bash
python scripts/run_extraction.py --input data/raw/pdfs_digital/sample.pdf
```

Run extraction on a folder of supported documents:

```bash
python scripts/run_extraction.py --input data/raw --output-dir data/interim/extracted_text
```

Output format:

- One JSON file per input document in `data/interim/extracted_text/`
- Includes routed engine (`pymupdf` or `surya`), page text, blocks, and metadata
- Supported inputs: `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.webp`

## Phase 2 Quick Start (Annotation to NER Dataset)

Convert Label Studio exports to token-classification splits:

```bash
python scripts/run_annotation_export.py --input-dir data/annotations/label_studio_exports --output-dir data/processed
```

Generated files:

- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/test.jsonl`
- `data/processed/metadata.json`

## Phase 3 Quick Start (PubMedBERT NER Training)

Train token-classification model from processed JSONL splits:

```bash
python scripts/run_ner_training.py --config configs/ner_train.yaml --data-dir data/processed --output-dir artifacts/models/pubmedbert_ner
```

Regenerate train/val/test with explicit split ratios and seed before training:

```bash
python scripts/generate_synthetic_ner_data.py --observations-csv data/interim/normalized_records/biomarker_observations_clean.csv --ranges-csv knowledge_base/reference_ranges/biomarker_reference_ranges.csv --output-dir data/processed --max-examples 20000 --seed 42 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

Training outputs:

- `artifacts/models/pubmedbert_ner/model/`
- `artifacts/models/pubmedbert_ner/label_mapping.json`

## Phase 4 Quick Start (NER Inference)

Run entity extraction on text or extraction JSON output:

```bash
python scripts/run_ner_inference.py --input data/interim/extracted_text/sample.json --model-dir artifacts/models/pubmedbert_ner/model --output artifacts/sample_outputs/ner_predictions.json
```

Inference output:

- `artifacts/sample_outputs/ner_predictions.json`
- Contains normalized entity objects with `label`, `text`, `start`, `end`, and `score`

## Phase 5 Quick Start (One-Command Pipeline)

Run extraction + NER + rule-based interpretation in one command:

```bash
python scripts/run_pipeline.py --input data/raw/pdfs_digital/sample.pdf --model-dir artifacts/models/pubmedbert_ner/model --output artifacts/sample_outputs/pipeline_output.json
```

Pipeline output:

- `artifacts/sample_outputs/pipeline_output.json`
- Includes extraction payload, NER entities, interpreted rows, and status summary

## Phase 6 Quick Start (Recommendation Indexes)

Build retrieval indexes for recommendation modules:

```bash
python scripts/build_indexes.py
```

Generated index artifacts:

- `knowledge_base/retrieval_index/tfidf/tfidf_index.npz`
- `knowledge_base/retrieval_index/faiss/faiss.index` (when semantic deps are available)

## Phase 7 Quick Start (Evaluation)

Run NER + retrieval evaluation:

```bash
python scripts/run_evaluation.py --data-dir data/processed --model-dir artifacts/models/pubmedbert_ner/model --retrieval-benchmark data/processed/retrieval_eval.jsonl --top-k 5 --output artifacts/metrics/evaluation_metrics.json
```

Evaluation output:

- `artifacts/metrics/evaluation_metrics.json`
- Contains NER token metrics and retrieval precision/recall/MRR

## Latest Training Results (April 2026)

Final single-run evaluation (HPC):

- NER token accuracy: `0.9233`
- NER entity precision: `0.9621`
- NER entity recall: `0.8236`
- NER entity F1: `0.8875`

5-fold cross-validation summary (robustness check):

- Entity-token F1 mean: `0.8902`
- Entity-token F1 std: `0.0034`
- Precision mean: `0.9561`
- Recall mean: `0.8329`
- Token accuracy mean: `0.9235`

Cross-validation artifacts:

- `artifacts/runs/cv5/fold_0.json`
- `artifacts/runs/cv5/fold_1.json`
- `artifacts/runs/cv5/fold_2.json`
- `artifacts/runs/cv5/fold_3.json`
- `artifacts/runs/cv5/fold_4.json`
- `artifacts/runs/cv5/summary.json`

## Model Artifacts and Git LFS

The final PubMedBERT model is committed with Git LFS under:

- `artifacts/models/pubmedbert_ner/model/`

To fetch model files after cloning:

```bash
git lfs install
git lfs pull
```

## Detailed Folder Structure (Implementation)

Use the following structure to keep data, models, services, and experiments cleanly separated:

```text
AAI-6620---NLP-Final-Project/
|-- README.md
|-- .gitignore
|-- requirements.txt
|-- pyproject.toml
|-- configs/
|   |-- extraction.yaml
|   |-- ner_train.yaml
|   |-- classification.yaml
|   `-- recommendation.yaml
|-- data/
|   |-- raw/
|   |   |-- pdfs_digital/
|   |   |-- pdfs_scanned/
|   |   `-- sources_manifest.csv
|   |-- interim/
|   |   |-- extracted_text/
|   |   |-- ocr_outputs/
|   |   `-- normalized_records/
|   |-- annotations/
|   |   |-- label_studio_exports/
|   |   |-- bio_tokens/
|   |   `-- guidelines/
|   `-- processed/
|       |-- train/
|       |-- val/
|       `-- test/
|-- knowledge_base/
|   |-- reference_ranges/
|   |   |-- biomarker_reference_ranges.csv
|   |   `-- unit_conversion_map.csv
|   |-- nutrition/
|   |   |-- usda_foods.csv
|   |   `-- lifestyle_advice.json
|   `-- retrieval_index/
|       |-- tfidf/
|       `-- faiss/
|-- src/
|   |-- extraction/
|   |   |-- router.py
|   |   |-- pymupdf_extractor.py
|   |   |-- surya_ocr_extractor.py
|   |   `-- table_parser.py
|   |-- ner/
|   |   |-- dataset_builder.py
|   |   |-- train_pubmedbert.py
|   |   `-- infer_pubmedbert.py
|   |-- interpretation/
|   |   |-- range_matcher.py
|   |   `-- rule_classifier.py
|   |-- recommendation/
|   |   |-- tfidf_retriever.py
|   |   |-- semantic_retriever.py
|   |   |-- ranker.py
|   |   `-- template_generator.py
|   |-- api/
|   |   |-- app.py
|   |   |-- routes.py
|   |   `-- models.py
|   `-- frontend/
|       `-- streamlit_app.py
|-- scripts/
|   |-- run_extraction.py
|   |-- run_annotation_export.py
|   |-- run_ner_training.py
|   |-- run_evaluation.py
|   `-- build_indexes.py
|-- tests/
|   |-- extraction/
|   |-- ner/
|   |-- interpretation/
|   |-- recommendation/
|   `-- api/
|-- notebooks/
|   |-- 01_data_audit.ipynb
|   |-- 02_annotation_qa.ipynb
|   |-- 03_ner_error_analysis.ipynb
|   `-- 04_retrieval_comparison.ipynb
|-- reports/
|   |-- figures/
|   |-- tables/
|   `-- final_report_draft.md
|-- artifacts/
|   |-- models/
|   |-- metrics/
|   `-- sample_outputs/
`-- docs/
	|-- api_contract.md
	|-- annotation_playbook.md
	`-- decisions_log.md
```

Recommended ownership mapping:

- `src/extraction`: PDF routing + PyMuPDF + Surya OCR components
- `src/ner`: PubMedBERT training/inference + SpaCy fallback
- `src/interpretation`: unit normalization and range-based classification
- `src/recommendation`: TF-IDF and FAISS semantic retrieval + template generation
- `src/api` and `src/frontend`: deployment-facing app layers
- `knowledge_base`: curated medical ranges and nutrition recommendations
- `artifacts`: saved models, experiment metrics, and demo-ready outputs

## Current Status

Completed:

- End-to-end digital PDF pipeline (extraction -> NER -> interpretation -> recommendation)
- FastAPI endpoints for extraction, NER, interpretation, recommendation, and full pipeline
- Label Studio dataset conversion and PubMedBERT training/inference scripts
- Retrieval and recommendation modules with TF-IDF + semantic retrieval

Next:

- Implement real OCR extraction for scanned PDFs in `surya_ocr_extractor.py`
- Improve retrieval quality (current retrieval benchmark remains lower than NER quality)
- Add production smoke tests and refresh docs/contracts for release

## Team Responsibilities (Primary Ownership)

- Om Patel:

  - Hybrid extraction router and preprocessing
  - PubMedBERT training pipeline
  - FAISS retrieval pipeline
  - FastAPI backend

- Yash Jain:

  - PyMuPDF extraction module
  - SpaCy rule-based fallback
  - Alert system/specialist mapping
  - Plotly visualizations and PDF export

- Ruthvik Bandari:
  - Surya OCR module
  - NER evaluation and error analysis
  - Rule-based classification and unit logic
  - Streamlit UI

All team members contribute to data collection, annotation, template design, integration testing, and final report writing.

## Why This Project Matters

This system is designed to reduce confusion around blood reports and make health information more accessible. It combines NLP, information extraction, retrieval, and explainable generation to transform raw lab reports into understandable guidance.

## Environment Setup (Windows)

Python backend setup:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- All project CLI scripts now self-bootstrap project imports, so commands in this README run directly from repository root without manually setting `PYTHONPATH`.
- Semantic retrieval gracefully degrades to TF-IDF if FAISS/Sentence-Transformers runtime pieces are unavailable.

Next.js frontend setup (Bun runtime):

```bash
cd frontend-next
bun install
bun run dev
```

Frontend quality checks:

```bash
cd frontend-next
bun run check
bun run build
```

## Next.js Frontend (Bun + TypeScript)

A modern App Router frontend is available under `frontend-next/` with:

- TypeScript-only setup (strict mode)
- Mantine UI components and theming
- Smooth light/dark theme toggle
- React Query for request caching and mutation orchestration
- Zustand for app-level state
- Zod + React Hook Form for validated input flows
- Framer Motion page/section transitions
- Loading skeletons and route-level loading/error boundaries
- `sharp`-powered image blur placeholder generation for optimized hero imagery

Default backend URL in UI: `http://localhost:8000`

## Latest Verified Evaluation (Local Run)

Command:

```bash
python scripts/run_evaluation.py --data-dir data/processed --model-dir artifacts/models/pubmedbert_ner/model --retrieval-benchmark data/processed/retrieval_eval.jsonl --top-k 5 --output artifacts/metrics/evaluation_metrics.json
```

Output snapshot:

- NER token accuracy: `0.9237`
- Entity-token precision: `0.9615`
- Entity-token recall: `0.8254`
- Entity-token F1: `0.8883`
- Retrieval precision@5: `0.5867`
- Retrieval recall@5: `0.9854`
- Retrieval MRR@5: `0.9958`
