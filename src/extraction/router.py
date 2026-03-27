"""Route extraction inputs to the best extraction engine."""

from pathlib import Path

import fitz


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", *SUPPORTED_IMAGE_EXTENSIONS}


def _decide_engine_from_text_lengths(text_lengths: list[int], threshold: int = 80) -> str:
    """Choose engine from early-page text density."""
    if not text_lengths:
        return "surya"
    if max(text_lengths) >= threshold:
        return "pymupdf"
    return "surya"


def is_supported_document(path: str | Path) -> bool:
    candidate = Path(path)
    return candidate.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS


def _route_pdf_path(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    try:
        page_scan = min(3, len(doc))
        text_lengths: list[int] = []
        for page_index in range(page_scan):
            page_text = doc[page_index].get_text("text").strip()
            text_lengths.append(len(page_text))
        return _decide_engine_from_text_lengths(text_lengths)
    finally:
        doc.close()


def route_document(file_path: str) -> str:
    """Route supported files to `pymupdf` or OCR (`surya`)."""
    path = Path(file_path)
    if not path.exists() or not is_supported_document(path):
        raise FileNotFoundError(f"Supported document not found: {file_path}")

    if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
        return "surya"
    return _route_pdf_path(path)


def route_pdf(file_path: str) -> str:
    """Route to `pymupdf` for digital PDFs, otherwise `surya` for OCR path."""
    pdf_path = Path(file_path)
    if pdf_path.suffix.lower() != ".pdf" or not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    return _route_pdf_path(pdf_path)

