"""Pydantic request/response models."""

from pydantic import BaseModel
from pydantic import Field


class HealthResponse(BaseModel):
	status: str = "ok"


class ExtractionRequest(BaseModel):
	pdf_path: str = Field(..., description="Absolute or workspace-relative PDF path")


class NerRequest(BaseModel):
	text: str
	model_dir: str = "artifacts/models/pubmedbert_ner/model"


class InterpretationRequest(BaseModel):
	rows: list[dict]


class PipelineRequest(BaseModel):
	pdf_path: str
	model_dir: str = "artifacts/models/pubmedbert_ner/model"

