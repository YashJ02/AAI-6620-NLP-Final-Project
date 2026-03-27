from src.api.app import health
from src.api.models import ExtractionRequest
from src.api.routes import extract_endpoint


def test_health_returns_ok_status():
    assert health() == {"status": "ok"}


def test_extract_endpoint_rejects_missing_pdf_path():
    payload = ExtractionRequest(pdf_path="does_not_exist.pdf")

    try:
        extract_endpoint(payload)
        assert False, "Expected extract_endpoint to raise for invalid path"
    except Exception as exc:  # FastAPI HTTPException
        assert getattr(exc, "status_code", None) == 400
        assert "Invalid PDF path" in str(getattr(exc, "detail", ""))

