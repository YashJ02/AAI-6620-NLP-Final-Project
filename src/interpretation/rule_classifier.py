"""Rule-based interpretation for standard biomarkers."""

import re

from src.interpretation.range_matcher import classify_against_range


RANGE_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$")


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


def classify_record(record: dict) -> dict:
    """Classify one parsed biomarker row as low/normal/high/unknown."""
    value = _to_float(record.get("value", ""))
    parsed_range = _parse_reference_range(record.get("reference_range", ""))

    if value is None or parsed_range is None:
        return {
            "biomarker": record.get("biomarker_normalized", record.get("biomarker", "")),
            "status": "unknown",
            "reason": "missing_or_invalid_value_or_range",
            "value": record.get("value", ""),
            "unit": record.get("unit", ""),
            "reference_range": record.get("reference_range", ""),
        }

    low, high = parsed_range
    status = classify_against_range(value=value, ref_low=low, ref_high=high)
    return {
        "biomarker": record.get("biomarker_normalized", record.get("biomarker", "")),
        "status": status,
        "value": value,
        "unit": record.get("unit", ""),
        "reference_range": f"{low}-{high}",
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

