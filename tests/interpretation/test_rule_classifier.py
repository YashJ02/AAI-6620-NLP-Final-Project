from src.interpretation.rule_classifier import classify_record
from src.interpretation.rule_classifier import summarize_statuses


def test_classify_record_normal():
    record = {
        "biomarker_normalized": "Hemoglobin",
        "value": "14.1",
        "unit": "g/dL",
        "reference_range": "13.0-17.0",
    }

    result = classify_record(record)

    assert result["status"] == "normal"


def test_classify_record_unknown_when_missing_range():
    record = {
        "biomarker_normalized": "TSH",
        "value": "6.2",
        "unit": "mIU/L",
        "reference_range": "",
    }

    result = classify_record(record)

    assert result["status"] == "unknown"


def test_summarize_statuses_counts_all_categories():
    summary = summarize_statuses(
        [
            {"status": "low"},
            {"status": "normal"},
            {"status": "high"},
            {"status": "unknown"},
        ]
    )

    assert summary == {"low": 1, "normal": 1, "high": 1, "unknown": 1}
