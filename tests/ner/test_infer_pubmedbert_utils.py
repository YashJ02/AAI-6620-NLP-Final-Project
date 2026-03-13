from src.ner.infer_pubmedbert import _dedupe_entities
from src.ner.infer_pubmedbert import _normalize_entity_label
from src.ner.infer_pubmedbert import _split_text


def test_split_text_returns_chunks_for_long_input():
    text = "A" * 2100
    chunks = _split_text(text, chunk_size=1000, overlap=100)

    assert len(chunks) >= 2
    assert chunks[0][0] == 0


def test_normalize_entity_label_strips_bio_prefix():
    assert _normalize_entity_label("B-BIOMARKER") == "BIOMARKER"
    assert _normalize_entity_label("I-VALUE") == "VALUE"
    assert _normalize_entity_label("UNIT") == "UNIT"


def test_dedupe_entities_removes_exact_duplicates():
    entities = [
        {"label": "BIOMARKER", "start": 0, "end": 10, "text": "Hemoglobin", "score": 0.9},
        {"label": "BIOMARKER", "start": 0, "end": 10, "text": "Hemoglobin", "score": 0.8},
    ]

    output = _dedupe_entities(entities)

    assert len(output) == 1
