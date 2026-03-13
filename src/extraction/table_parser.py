"""Helpers for parsing lab report tables into row-level records."""

import re


VALUE_RE = re.compile(r"(?<![A-Za-z])(-?\d+(?:\.\d+)?)")
RANGE_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:-|to)\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
UNIT_RE = re.compile(r"\b(?:g/dL|mg/dL|mmol/L|mIU/L|IU/L|U/L|10\^\d+/L|%|pg|fL|ng/mL)\b", re.IGNORECASE)

BIOMARKER_ALIASES = {
    "hb": "Hemoglobin",
    "hgb": "Hemoglobin",
    "hemoglobin": "Hemoglobin",
    "rbc": "RBC",
    "wbc": "WBC",
    "plt": "Platelets",
    "platelet": "Platelets",
    "platelets": "Platelets",
    "tsh": "TSH",
    "glucose": "Glucose",
}


def _normalize_spaces(text: str) -> str:
    return " ".join(text.split())


def _normalize_biomarker_name(name: str) -> str:
    normalized = _normalize_spaces(name).strip(" :-")
    if not normalized:
        return ""

    simple = re.sub(r"[^A-Za-z0-9 ]", "", normalized).lower().strip()
    return BIOMARKER_ALIASES.get(simple, normalized)


def _compute_confidence(*, biomarker: str, unit: str, reference_range: str, source_line: str) -> float:
    # Simple deterministic score based on expected row signals.
    score = 0.5
    if biomarker and re.search(r"[A-Za-z]", biomarker):
        score += 0.2
    if unit:
        score += 0.15
    if reference_range:
        score += 0.15
    if len(source_line.split()) < 3:
        score -= 0.2
    return max(0.0, min(1.0, round(score, 2)))


def _parse_line(line: str) -> dict | None:
    clean = _normalize_spaces(line)
    if not clean:
        return None

    value_match = VALUE_RE.search(clean)
    if not value_match:
        return None

    range_match = RANGE_RE.search(clean)
    unit_match = UNIT_RE.search(clean)

    value = value_match.group(1)
    reference_range = ""
    if range_match:
        reference_range = f"{range_match.group(1)}-{range_match.group(2)}"

    biomarker_candidate = clean[: value_match.start()].strip(" :-")
    if not biomarker_candidate:
        return None

    biomarker_normalized = _normalize_biomarker_name(biomarker_candidate)

    parsed = {
        "biomarker": biomarker_candidate,
        "biomarker_normalized": biomarker_normalized,
        "value": value,
        "unit": unit_match.group(0) if unit_match else "",
        "reference_range": reference_range,
        "source_line": clean,
    }

    # Avoid obvious false positives where the biomarker is just punctuation.
    if not re.search(r"[A-Za-z]", parsed["biomarker"]):
        return None

    parsed["confidence"] = _compute_confidence(
        biomarker=parsed["biomarker_normalized"],
        unit=parsed["unit"],
        reference_range=parsed["reference_range"],
        source_line=parsed["source_line"],
    )

    return parsed


def parse_table_rows(raw_table: list) -> list:
    """Parse candidate table rows from block-like text objects or strings."""
    if not raw_table:
        return []

    lines: list[str] = []
    for item in raw_table:
        if isinstance(item, dict):
            text = item.get("text", "")
            if text:
                lines.extend(text.splitlines())
        elif isinstance(item, str):
            lines.extend(item.splitlines())

    parsed_rows: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()
    for line in lines:
        parsed = _parse_line(line)
        if not parsed:
            continue

        key = (
            parsed["biomarker_normalized"].lower(),
            parsed["value"],
            parsed["unit"].lower(),
            parsed["reference_range"],
        )
        if key in seen:
            continue
        seen.add(key)
        parsed_rows.append(parsed)

    return parsed_rows

