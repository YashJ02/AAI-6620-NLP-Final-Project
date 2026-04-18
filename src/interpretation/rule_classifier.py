"""Rule-based interpretation for standard biomarkers."""

import re

from src.interpretation.range_matcher import classify_against_range


RANGE_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$")

DEFAULT_REFERENCE_RANGES: dict[str, tuple[float, float]] = {
    "fasting plasma glucose": (70.0, 99.0),
    "glucose": (70.0, 99.0),
    "cholesterol": (0.0, 200.0),
    "triglycerides": (0.0, 150.0),
    "hdl": (40.0, 60.0),
    "ldl": (0.0, 130.0),
    "vldl": (5.0, 40.0),
    "vitamin d": (30.0, 100.0),
}


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_reference_range(reference_range: str) -> tuple[float, float] | None:
    if not isinstance(reference_range, str):
        return None
    match = RANGE_RE.match(reference_range)
    if not match:
        return None
    low = _to_float(match.group(1))
    high = _to_float(match.group(2))
    if low is None or high is None:
        return None
    return low, high


def _display_biomarker(record: dict) -> str:
    normalized = _normalize_biomarker_text(str(record.get("biomarker_normalized", "") or ""))
    raw = _normalize_biomarker_text(str(record.get("biomarker", "") or ""))
    return normalized or raw


def _normalize_biomarker_text(text: str) -> str:
    value = text.strip()
    value = re.sub(r"^[A-Za-z0-9]+\)\s*", "", value)
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" :=<>-.")
    return value


def _lookup_reference_range(record: dict) -> tuple[float, float] | None:
    candidates = [
        _normalize_biomarker_text(str(record.get("biomarker_normalized", "") or "")),
        _normalize_biomarker_text(str(record.get("biomarker", "") or "")),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        key = re.sub(r"[^a-z0-9 ]", "", candidate.lower()).strip()
        key = re.sub(r"\s+", " ", key)
        if key in DEFAULT_REFERENCE_RANGES:
            return DEFAULT_REFERENCE_RANGES[key]
    return None


def classify_record(record: dict) -> dict:
    """Classify one parsed biomarker row as low/normal/high/unknown."""
    display_biomarker = _display_biomarker(record)
    normalized_biomarker = _normalize_biomarker_text(str(record.get("biomarker_normalized", "") or ""))
    raw_biomarker = _normalize_biomarker_text(str(record.get("biomarker", "") or ""))
    value = _to_float(record.get("value", ""))
    parsed_range = _parse_reference_range(record.get("reference_range", ""))

    inferred_range = None
    if parsed_range is None:
        inferred_range = _lookup_reference_range(record)
    effective_range = parsed_range or inferred_range

    if value is None or effective_range is None:
        return {
            "biomarker": display_biomarker,
            "biomarker_normalized": normalized_biomarker,
            "biomarker_raw": raw_biomarker,
            "status": "unknown",
            "reason": "missing_or_invalid_value_or_range",
            "value": record.get("value", ""),
            "unit": record.get("unit", ""),
            "reference_range": record.get("reference_range", ""),
        }

    low, high = effective_range
    status = classify_against_range(value=value, ref_low=low, ref_high=high)
    return {
        "biomarker": display_biomarker,
        "biomarker_normalized": normalized_biomarker,
        "biomarker_raw": raw_biomarker,
        "status": status,
        "value": value,
        "unit": record.get("unit", ""),
        "reference_range": f"{low}-{high}",
        "range_source": "parsed" if parsed_range is not None else "inferred",
    }


def classify_records(records: list[dict]) -> list[dict]:
    return [classify_record(record) for record in records]


def summarize_statuses(classifications: list[dict]) -> dict:
    summary = {"low": 0, "normal": 0, "high": 0, "unknown": 0}
    for row in classifications:
        status = row.get("status", "unknown")
        if status not in summary:
            status = "unknown"
        summary[status] += 1
    return summary

