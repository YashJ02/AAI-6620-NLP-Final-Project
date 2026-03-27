# Final Report Draft

## Project

Health Blood Report Analyzer and Recommendation Engine (AAI 6620).

## Current Status (March 26, 2026)

The project has a working end-to-end software path:

1. PDF extraction with routing (digital and scanned)
2. NER train and inference scripts
3. Rule-based interpretation
4. Recommendation retrieval and templated summary generation
5. FastAPI routes and Streamlit frontend integration points

The implementation is functional for smoke usage, but not yet fully production-ready for clinical-grade quality guarantees.

## What Is Completed

### Extraction

- Digital extraction via PyMuPDF is implemented.
- Scanned extraction via Surya OCR is implemented.
- Extraction router and scanned OCR integration test are present.

### NER

- Dataset builder, trainer, and inference modules are implemented.
- Model artifact structure under artifacts/models/pubmedbert_ner is in place.

### Interpretation

- Rule-based classification and status summarization are implemented.
- Range matching and row-level interpretation pipeline exists.

### Recommendation

- TF-IDF and semantic retrieval modules are implemented.
- Candidate ranking and Jinja2 summary generation are implemented.

### API and Pipeline

- FastAPI app with health, extract, ner, interpret, recommend, and pipeline routes is implemented.
- One-command pipeline runner exists and writes integrated JSON output.

## What Is Not Finished

1. Final report content is still partial and needs full experiment results, analysis, and references.
2. API and end-to-end tests are still limited; some suites were previously placeholders.
3. Dataset quality and annotation conversion are the primary bottlenecks.
4. Production hardening items (dependency pinning, stricter validation, CI gates) are incomplete.

## Dataset Assessment

Current annotation exports are largely image-based with weak-label predictions and sparse finalized human text annotations.
This causes a mismatch with the text-span BIO dataset builder and can result in empty processed splits unless conversion is improved.

Assessment:

- Good enough for pipeline plumbing and smoke testing.
- Not sufficient yet for high-confidence biomedical NER performance claims.
- Requires a stronger gold-labeled set and annotation QA before final model evaluation.

## Risks

1. NER quality risk due to limited supervised signal in processed training data.
2. API contract risk if clients expect file-upload extract while current extract route uses JSON pdf_path.
3. Regression risk due to limited integration test coverage.

## Immediate Next Steps

1. Finalize annotation-to-BIO conversion for actual project annotation format.
2. Rebuild train/val/test from curated labels and retrain NER.
3. Expand API and pipeline regression tests and run in CI.
4. Complete report sections: methodology, experiments, error analysis, limitations, and conclusions.

