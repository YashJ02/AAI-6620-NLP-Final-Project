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

Run extraction on one PDF:

```bash
python scripts/run_extraction.py --input data/raw/pdfs_digital/sample.pdf
```

Run extraction on a folder of PDFs:

```bash
python scripts/run_extraction.py --input data/raw --output-dir data/interim/extracted_text
```

Output format:
- One JSON file per PDF in `data/interim/extracted_text/`
- Includes routed engine (`pymupdf` or `surya`), page text, blocks, and metadata

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
|   |-- common/
|   |   |-- constants.py
|   |   |-- schema.py
|   |   `-- logging_utils.py
|   |-- extraction/
|   |   |-- router.py
|   |   |-- pymupdf_extractor.py
|   |   |-- surya_ocr_extractor.py
|   |   `-- table_parser.py
|   |-- ner/
|   |   |-- dataset_builder.py
|   |   |-- train_pubmedbert.py
|   |   |-- infer_pubmedbert.py
|   |   `-- spacy_fallback.py
|   |-- interpretation/
|   |   |-- unit_normalizer.py
|   |   |-- range_matcher.py
|   |   |-- rule_classifier.py
|   |   `-- ml_classifier.py
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
|       |-- streamlit_app.py
|       `-- plotly_components.py
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
- Problem scoping and architecture design
- Tech stack selection and research
- Annotation schema and guideline drafting
- Data source identification
- Team task planning

Next:
- Gather corpus files
- Start Label Studio annotation
- Implement Phase 1 extraction pipeline

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


