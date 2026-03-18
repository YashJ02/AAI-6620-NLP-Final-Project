"""Single-command pipeline runner: extraction -> NER -> interpretation."""

import argparse
import json
from pathlib import Path

from src.extraction.pymupdf_extractor import extract_text_pymupdf
from src.extraction.router import route_pdf
from src.extraction.surya_ocr_extractor import extract_text_surya
from src.interpretation.rule_classifier import classify_records
from src.interpretation.rule_classifier import summarize_statuses
from src.ner.infer_pubmedbert import predict
from src.recommendation.service import generate_recommendations


def _extract(pdf_path: Path) -> dict:
    engine = route_pdf(str(pdf_path))
    if engine == "pymupdf":
        return extract_text_pymupdf(str(pdf_path))
    return extract_text_surya(str(pdf_path))


def _get_parsed_rows(extraction_output: dict) -> list[dict]:
    tables = extraction_output.get("tables", [])
    if not tables:
        return []
    first_table = tables[0] if isinstance(tables[0], dict) else {}
    rows = first_table.get("rows", []) if isinstance(first_table, dict) else []
    return rows if isinstance(rows, list) else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end blood report pipeline")
    parser.add_argument("--input", required=True, help="Path to a PDF file")
    parser.add_argument(
        "--model-dir",
        default="artifacts/models/pubmedbert_ner/model",
        help="Path to trained NER model directory",
    )
    parser.add_argument(
        "--output",
        default="artifacts/sample_outputs/pipeline_output.json",
        help="Path to write full pipeline output JSON",
    )
    args = parser.parse_args()

    pdf_path = Path(args.input)
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"PDF input not found: {args.input}")

    extraction_output = _extract(pdf_path)
    full_text = extraction_output.get("full_text", "")
    ner_entities = predict(text=full_text, model_dir=args.model_dir) if full_text else []

    parsed_rows = _get_parsed_rows(extraction_output)
    interpreted_rows = classify_records(parsed_rows)
    interpretation_summary = summarize_statuses(interpreted_rows)
    recommendation = generate_recommendations(
        interpreted_rows=interpreted_rows,
        ner_entities=ner_entities,
        status_summary=interpretation_summary,
        patient_id=extraction_output.get("document_id", pdf_path.stem),
        top_k=5,
    )

    output_payload = {
        "document_id": extraction_output.get("document_id", pdf_path.stem),
        "source_path": str(pdf_path),
        "extraction": extraction_output,
        "ner": {
            "entity_count": len(ner_entities),
            "entities": ner_entities,
        },
        "interpretation": {
            "row_count": len(interpreted_rows),
            "status_summary": interpretation_summary,
            "rows": interpreted_rows,
        },
        "recommendation": {
            "query": recommendation["query"],
            "results": recommendation["results"],
            "summary": recommendation["summary"],
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    print(f"[OK] Pipeline output saved to {output_path}")
    print(f"[INFO] NER entities: {len(ner_entities)}")
    print(f"[INFO] Interpreted rows: {len(interpreted_rows)}")


if __name__ == "__main__":
    main()
