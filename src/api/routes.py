"""API routes for extraction, NER, interpretation, and full pipeline."""

from pathlib import Path

from fastapi import APIRouter
from fastapi import HTTPException

from src.api.models import ExtractionRequest
from src.api.models import InterpretationRequest
from src.api.models import NerRequest
from src.api.models import PipelineRequest
from src.api.models import RecommendationRequest
from src.extraction.pymupdf_extractor import extract_text_pymupdf
from src.extraction.router import route_pdf
from src.extraction.surya_ocr_extractor import extract_text_surya
from src.interpretation.rule_classifier import classify_records
from src.interpretation.rule_classifier import summarize_statuses
from src.recommendation.service import generate_recommendations


router = APIRouter(prefix="/v1", tags=["pipeline"])


def _extract_from_pdf(pdf_path: Path) -> dict:
	engine = route_pdf(str(pdf_path))
	if engine == "pymupdf":
		return extract_text_pymupdf(str(pdf_path))
	return extract_text_surya(str(pdf_path))


def _predict_entities(text: str, model_dir: str) -> list[dict]:
	try:
		from src.ner.infer_pubmedbert import predict
	except Exception as exc:
		raise RuntimeError(f"NER runtime unavailable: {exc}") from exc

	return predict(text=text, model_dir=model_dir)


@router.post("/extract")
def extract_endpoint(payload: ExtractionRequest) -> dict:
	pdf_path = Path(payload.pdf_path)
	if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
		raise HTTPException(status_code=400, detail="Invalid PDF path.")
	return _extract_from_pdf(pdf_path)


@router.post("/ner")
def ner_endpoint(payload: NerRequest) -> dict:
	try:
		entities = _predict_entities(text=payload.text, model_dir=payload.model_dir)
	except FileNotFoundError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except RuntimeError as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc
	return {"entity_count": len(entities), "entities": entities}


@router.post("/interpret")
def interpret_endpoint(payload: InterpretationRequest) -> dict:
	rows = classify_records(payload.rows)
	summary = summarize_statuses(rows)
	return {"row_count": len(rows), "status_summary": summary, "rows": rows}


@router.post("/recommend")
def recommend_endpoint(payload: RecommendationRequest) -> dict:
	return generate_recommendations(
		interpreted_rows=payload.interpreted_rows,
		ner_entities=payload.ner_entities,
		status_summary=payload.status_summary,
		patient_id=payload.patient_id,
		query=payload.query,
		top_k=payload.top_k,
	)


@router.post("/pipeline")
def pipeline_endpoint(payload: PipelineRequest) -> dict:
	pdf_path = Path(payload.pdf_path)
	if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
		raise HTTPException(status_code=400, detail="Invalid PDF path.")

	extraction_output = _extract_from_pdf(pdf_path)
	full_text = extraction_output.get("full_text", "")

	try:
		ner_entities = _predict_entities(text=full_text, model_dir=payload.model_dir) if full_text else []
	except FileNotFoundError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except RuntimeError as exc:
		raise HTTPException(status_code=500, detail=str(exc)) from exc

	tables = extraction_output.get("tables", [])
	rows = []
	if tables and isinstance(tables[0], dict):
		rows = tables[0].get("rows", []) if isinstance(tables[0].get("rows", []), list) else []

	interpreted_rows = classify_records(rows)
	status_summary = summarize_statuses(interpreted_rows)
	recommendation = generate_recommendations(
		interpreted_rows=interpreted_rows,
		ner_entities=ner_entities,
		status_summary=status_summary,
		patient_id=extraction_output.get("document_id", pdf_path.stem),
		top_k=5,
	)

	return {
		"document_id": extraction_output.get("document_id", pdf_path.stem),
		"source_path": str(pdf_path),
		"extraction": extraction_output,
		"ner": {
			"entity_count": len(ner_entities),
			"entities": ner_entities,
		},
		"interpretation": {
			"row_count": len(interpreted_rows),
			"status_summary": status_summary,
			"rows": interpreted_rows,
		},
		"recommendation": {
			"query": recommendation["query"],
			"results": recommendation["results"],
			"summary": recommendation["summary"],
		},
	}

