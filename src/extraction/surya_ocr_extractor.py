"""Scanned PDF extraction using Surya OCR."""

from pathlib import Path


def extract_text_surya(file_path: str) -> dict:
    """Return schema-compatible placeholder until Surya integration is added."""
    pdf_path = Path(file_path)
    if pdf_path.suffix.lower() != ".pdf" or not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    return {
        "document_id": pdf_path.stem,
        "source_path": str(pdf_path),
        "engine": "surya",
        "pages": [],
        "full_text": "",
        "tables": [],
        "metadata": {
            "page_count": 0,
            "has_text": False,
            "status": "placeholder",
            "message": "Surya OCR integration pending.",
        },
    }

