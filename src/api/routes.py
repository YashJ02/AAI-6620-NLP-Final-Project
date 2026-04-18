"""API routes for extraction, NER, interpretation, and full pipeline."""

import tempfile
from pathlib import Path
from shutil import copyfileobj

from fastapi import APIRouter
from fastapi import File
from fastapi import Form
from fastapi import HTTPException
from fastapi import UploadFile

from src.api.models import ExtractionRequest
from src.api.models import InterpretationRequest
from src.api.models import NerRequest
from src.api.models import PipelineRequest
from src.api.models import RecommendationRequest
from src.extraction.pymupdf_extractor import extract_text_pymupdf
from src.extraction.router import is_supported_document
from src.extraction.router import route_document
from src.extraction.surya_ocr_extractor import extract_text_surya
from src.interpretation.rule_classifier import classify_records
from src.interpretation.rule_classifier import summarize_statuses
from src.recommendation.service import generate_recommendations


router = APIRouter(prefix="/v1", tags=["pipeline"])


def _extract_from_source(source_path: Path) -> dict:
	engine = route_document(str(source_path))
	if engine == "pymupdf":
		return extract_text_pymupdf(str(source_path))
	try:
		return extract_text_surya(str(source_path))
	except RuntimeError as exc:
		# If OCR runtime is unavailable for a PDF, fallback to text extraction instead of hard failing.
		if source_path.suffix.lower() == ".pdf" and "easyocr runtime is unavailable" in str(exc):
			return extract_text_pymupdf(str(source_path))
		raise


def _predict_entities(text: str, model_dir: str) -> list[dict]:
	try:
		from src.ner.infer_pubmedbert import predict
	except Exception as exc:
		raise RuntimeError(f"NER runtime unavailable: {exc}") from exc

	return predict(text=text, model_dir=model_dir)


def _run_pipeline_for_source(
	source_path: Path,
	*,
	model_dir: str,
	source_display_path: str | None = None,
	document_id_fallback: str | None = None,
) -> dict:
	try:
		extraction_output = _extract_from_source(source_path)
	except FileNotFoundError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc
	except RuntimeError as exc:
		raise HTTPException(status_code=503, detail=str(exc)) from exc

	full_text = extraction_output.get("full_text", "")

	try:
		ner_entities = _predict_entities(text=full_text, model_dir=model_dir) if full_text else []
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

	document_id = extraction_output.get("document_id")
	if not document_id:
		document_id = document_id_fallback or source_path.stem

	recommendation = generate_recommendations(
		interpreted_rows=interpreted_rows,
		ner_entities=ner_entities,
		status_summary=status_summary,
		patient_id=document_id,
		top_k=5,
	)

	return {
		"document_id": document_id,
		"source_path": source_display_path or str(source_path),
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


@router.post("/extract")
def extract_endpoint(payload: ExtractionRequest) -> dict:
	source_path = Path(payload.pdf_path)
	if not source_path.exists() or not is_supported_document(source_path):
		raise HTTPException(status_code=400, detail="Invalid PDF path or unsupported document input path.")
	return _extract_from_source(source_path)


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
	source_path = Path(payload.pdf_path)
	if not source_path.exists() or not is_supported_document(source_path):
		raise HTTPException(status_code=400, detail="Invalid PDF path or unsupported document input path.")
	return _run_pipeline_for_source(source_path, model_dir=payload.model_dir)



@router.post("/pipeline/upload")
async def pipeline_upload_endpoint(
	file: UploadFile = File(...),
	model_dir: str = Form("artifacts/models/pubmedbert_ner/model"),
) -> dict:
	filename = (file.filename or "uploaded_document").strip() or "uploaded_document"
	suffix = Path(filename).suffix.lower()
	temp_path: Path | None = None

	try:
		with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="upload_") as tmp:
			temp_path = Path(tmp.name)
			copyfileobj(file.file, tmp)

		if temp_path is None or not is_supported_document(temp_path):
			raise HTTPException(
				status_code=400,
				detail="Unsupported uploaded file type. Upload PDF or supported image format.",
			)

		return _run_pipeline_for_source(
			temp_path,
			model_dir=model_dir,
			source_display_path=filename,
			document_id_fallback=Path(filename).stem,
		)
	finally:
		await file.close()
		if temp_path is not None and temp_path.exists():
			temp_path.unlink(missing_ok=True)

