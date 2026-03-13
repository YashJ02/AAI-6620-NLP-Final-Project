"""Rule-based interpretation for standard biomarkers."""


def classify_record(record: dict) -> dict:
    return {"record": record, "status": "unknown"}

