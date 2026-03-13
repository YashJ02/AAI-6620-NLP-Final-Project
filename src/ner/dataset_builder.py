"""Build token classification datasets from annotation exports."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path


VALID_ENTITY_LABELS = {"BIOMARKER", "VALUE", "UNIT", "REFERENCE_RANGE"}
TOKEN_RE = re.compile(r"\S+")


def _read_label_studio_files(annotation_dir: Path) -> list[dict]:
    tasks: list[dict] = []
    for file_path in sorted(annotation_dir.glob("*.json")):
        content = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(content, list):
            tasks.extend(content)
        elif isinstance(content, dict):
            tasks.append(content)
    return tasks


def _extract_text(task: dict) -> str:
    data = task.get("data", {}) if isinstance(task, dict) else {}
    if not isinstance(data, dict):
        return ""

    for key in ("text", "raw_text", "transcription", "content"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extract_spans(task: dict) -> list[dict]:
    annotations = task.get("annotations", [])
    if not annotations:
        return []

    first_annotation = annotations[0]
    results = first_annotation.get("result", []) if isinstance(first_annotation, dict) else []
    spans: list[dict] = []

    for result in results:
        if not isinstance(result, dict):
            continue
        value = result.get("value", {})
        if not isinstance(value, dict):
            continue

        start = value.get("start")
        end = value.get("end")
        labels = value.get("labels", [])

        if not isinstance(start, int) or not isinstance(end, int) or start >= end:
            continue
        if not labels or not isinstance(labels, list):
            continue

        entity = str(labels[0]).strip().upper()
        if entity not in VALID_ENTITY_LABELS:
            continue

        spans.append({"start": start, "end": end, "label": entity})

    return spans


def _tokenize_with_offsets(text: str) -> list[dict]:
    tokens: list[dict] = []
    for match in TOKEN_RE.finditer(text):
        tokens.append(
            {
                "text": match.group(0),
                "start": match.start(),
                "end": match.end(),
            }
        )
    return tokens


def _token_label_for_span(token_start: int, token_end: int, span_start: int, span_end: int) -> bool:
    # Any overlap counts as token belonging to the entity span.
    return token_start < span_end and token_end > span_start


def _build_bio_labels(tokens: list[dict], spans: list[dict]) -> list[str]:
    tags = ["O"] * len(tokens)

    # Prefer shorter spans first only when needed; primary key is start index.
    sorted_spans = sorted(spans, key=lambda s: (s["start"], s["end"]))
    for span in sorted_spans:
        label = span["label"]
        token_indices = [
            idx
            for idx, token in enumerate(tokens)
            if _token_label_for_span(token["start"], token["end"], span["start"], span["end"])
        ]
        if not token_indices:
            continue

        first = True
        for idx in token_indices:
            prefix = "B" if first else "I"
            tags[idx] = f"{prefix}-{label}"
            first = False

    return tags


def _convert_task_to_example(task: dict, example_id: int) -> dict | None:
    text = _extract_text(task)
    if not text:
        return None

    tokens = _tokenize_with_offsets(text)
    if not tokens:
        return None

    spans = _extract_spans(task)
    tags = _build_bio_labels(tokens, spans)

    return {
        "id": example_id,
        "tokens": [token["text"] for token in tokens],
        "ner_tags": tags,
        "text": text,
    }


def _split_examples(examples: list[dict], seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    examples_copy = list(examples)
    random.Random(seed).shuffle(examples_copy)
    total = len(examples_copy)

    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    train = examples_copy[:train_end]
    val = examples_copy[train_end:val_end]
    test = examples_copy[val_end:]
    return train, val, test


def _write_jsonl(file_path: Path, rows: list[dict]) -> None:
    with file_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_dataset(annotation_dir: str, output_dir: str) -> None:
    """Convert Label Studio exports into BIO token classification JSONL splits."""
    annotation_path = Path(annotation_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tasks = _read_label_studio_files(annotation_path)
    examples: list[dict] = []
    for idx, task in enumerate(tasks):
        converted = _convert_task_to_example(task, example_id=idx)
        if converted is not None:
            examples.append(converted)

    train, val, test = _split_examples(examples)

    _write_jsonl(output_path / "train.jsonl", train)
    _write_jsonl(output_path / "val.jsonl", val)
    _write_jsonl(output_path / "test.jsonl", test)

    metadata = {
        "total_examples": len(examples),
        "train_examples": len(train),
        "val_examples": len(val),
        "test_examples": len(test),
        "labels": sorted(list(VALID_ENTITY_LABELS)),
    }
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

