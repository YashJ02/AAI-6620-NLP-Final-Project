"""Run inference with trained PubMedBERT NER model."""

from __future__ import annotations

import re
from pathlib import Path

from transformers import pipeline


CHUNK_CHAR_SIZE = 1000
CHUNK_OVERLAP = 120


def _split_text(text: str, chunk_size: int = CHUNK_CHAR_SIZE, overlap: int = CHUNK_OVERLAP) -> list[tuple[int, str]]:
    clean = text.strip()
    if not clean:
        return []

    chunks: list[tuple[int, str]] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        chunk = clean[start:end]
        chunks.append((start, chunk))
        if end == len(clean):
            break
        start = max(0, end - overlap)
    return chunks


def _normalize_entity_label(label: str) -> str:
    upper = (label or "").upper()
    if upper.startswith("B-") or upper.startswith("I-"):
        return upper[2:]
    return upper


def _dedupe_entities(entities: list[dict]) -> list[dict]:
    seen: set[tuple[str, int, int, str]] = set()
    output: list[dict] = []
    for ent in entities:
        key = (ent["label"], ent["start"], ent["end"], ent["text"])
        if key in seen:
            continue
        seen.add(key)
        output.append(ent)
    return output


def _cleanup_entity_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def predict(text: str, model_dir: str = "artifacts/models/pubmedbert_ner/model", max_length: int = 256) -> list[dict]:
    """Predict entities from free text using a trained token classification model."""
    if not text or not text.strip():
        return []

    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    ner_pipeline = pipeline(
        task="token-classification",
        model=str(model_path),
        tokenizer=str(model_path),
        aggregation_strategy="simple",
    )

    all_entities: list[dict] = []
    for chunk_start, chunk_text in _split_text(text):
        predictions = ner_pipeline(chunk_text, truncation=True, max_length=max_length)
        for pred in predictions:
            start = int(pred.get("start", 0)) + chunk_start
            end = int(pred.get("end", 0)) + chunk_start
            label = _normalize_entity_label(pred.get("entity_group", pred.get("entity", "")))

            entity_text = _cleanup_entity_text(text[start:end])
            all_entities.append(
                {
                    "label": label,
                    "text": entity_text,
                    "start": start,
                    "end": end,
                    "score": round(float(pred.get("score", 0.0)), 4),
                }
            )

    deduped = _dedupe_entities(all_entities)
    return sorted(deduped, key=lambda x: (x["start"], x["end"]))

