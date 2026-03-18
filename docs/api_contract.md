# API Contract

Base URL:

- Local development: `http://localhost:8000`

Versioning:

- Versioned routes live under `/v1`

## Health

Endpoint:

- `GET /health`

Response:

```json
{
  "status": "ok"
}
```

## Extract

Endpoint:

- `POST /v1/extract`

Request:

```json
{
  "pdf_path": "data/raw/pdfs_digital/sample.pdf"
}
```

Response fields:

- `document_id`
- `source_path`
- `engine`
- `pages`
- `full_text`
- `tables`
- `metadata`

Notes:

- `engine` is `pymupdf` for digital PDFs and `surya` for OCR fallback path.
- Current OCR module returns a schema-compatible placeholder payload.

## NER Inference

Endpoint:

- `POST /v1/ner`

Request:

```json
{
  "text": "Hemoglobin 12.5 g/dL 13.0-17.0",
  "model_dir": "artifacts/models/pubmedbert_ner/model"
}
```

Response:

```json
{
  "entity_count": 2,
  "entities": [
    {
      "label": "BIOMARKER",
      "text": "Hemoglobin",
      "start": 0,
      "end": 10,
      "score": 0.9876
    }
  ]
}
```

## Interpretation

Endpoint:

- `POST /v1/interpret`

Request:

```json
{
  "rows": [
    {
      "biomarker_normalized": "Hemoglobin",
      "value": "12.5",
      "unit": "g/dL",
      "reference_range": "13.0-17.0"
    }
  ]
}
```

## Recommend

Endpoint:

- `POST /v1/recommend`

Request:

```json
{
  "query": "low hemoglobin iron foods",
  "interpreted_rows": [],
  "ner_entities": [],
  "status_summary": {
    "low": 0,
    "normal": 0,
    "high": 0,
    "unknown": 0
  },
  "patient_id": "demo",
  "top_k": 5
}
```

Response:

```json
{
  "query": "low hemoglobin iron foods",
  "results": [
    {
      "id": "advice_iron_low_1",
      "text": "For low hemoglobin, include iron-rich foods...",
      "score": 0.81,
      "combined_score": 0.61,
      "method": "semantic"
    }
  ],
  "summary": "Report summary for demo: ..."
}
```

Response:

```json
{
  "row_count": 1,
  "status_summary": {
    "low": 1,
    "normal": 0,
    "high": 0,
    "unknown": 0
  },
  "rows": [
    {
      "biomarker": "Hemoglobin",
      "status": "low",
      "value": 12.5,
      "unit": "g/dL",
      "reference_range": "13.0-17.0"
    }
  ]
}
```

## Full Pipeline

Endpoint:

- `POST /v1/pipeline`

Request:

```json
{
  "pdf_path": "data/raw/pdfs_digital/sample.pdf",
  "model_dir": "artifacts/models/pubmedbert_ner/model"
}
```

Response:

- Top-level keys: `document_id`, `source_path`, `extraction`, `ner`, `interpretation`.
- This response mirrors the output schema produced by `scripts/run_pipeline.py`.

## Error Handling

Error format:

```json
{
  "detail": "Invalid PDF path."
}
```

Common HTTP status codes:

- `400`: invalid path or missing model directory
- `422`: request validation error
- `500`: unexpected server error
