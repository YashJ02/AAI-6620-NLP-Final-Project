"""Route PDFs to the best extraction engine."""

from pathlib import Path

import fitz


def _decide_engine_from_text_lengths(text_lengths: list[int], threshold: int = 80) -> str:
    """Choose engine from early-page text density."""
    if not text_lengths:
        return "surya"
    if max(text_lengths) >= threshold:
        return "pymupdf"
    return "surya"


def route_pdf(file_path: str) -> str:
    """Route to `pymupdf` for digital PDFs, otherwise `surya` for OCR path."""
    pdf_path = Path(file_path)
    if pdf_path.suffix.lower() != ".pdf" or not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

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

