"""Pydantic request/response models."""

from pydantic import BaseModel
from pydantic import Field


class ExtractionRequest(BaseModel):
	pdf_path: str = Field(
		...,
		description="Absolute or workspace-relative extraction input path (.pdf/.png/.jpg/.jpeg/.tif/.tiff/.bmp/.webp)",
	)


class NerRequest(BaseModel):
	text: str
	model_dir: str = "artifacts/models/pubmedbert_ner/model"


class InterpretationRequest(BaseModel):
	rows: list[dict]


class PipelineRequest(BaseModel):
	pdf_path: str = Field(..., description="Path to extraction input (PDF or image)")
	model_dir: str = "artifacts/models/pubmedbert_ner/model"


class RecommendationRequest(BaseModel):
	query: str = ""
	interpreted_rows: list[dict] = Field(default_factory=list)
	ner_entities: list[dict] = Field(default_factory=list)
	status_summary: dict = Field(default_factory=lambda: {"low": 0, "normal": 0, "high": 0, "unknown": 0})
	patient_id: str = "unknown"
	top_k: int = 5

