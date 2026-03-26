"""Scanned PDF extraction using Surya OCR."""

from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np

from src.extraction.table_parser import parse_table_rows


_OCR_READER = None


def _get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is not None:
        return _OCR_READER

    try:
        import easyocr
    except Exception as exc:
        raise RuntimeError(
            "easyocr runtime is unavailable for scanned-PDF OCR extraction. "
            "Install dependencies from requirements.txt and ensure torch runtime is healthy."
        ) from exc

    # Prefer CPU by default for portability. If CUDA build is available, EasyOCR can still leverage it.
    _OCR_READER = easyocr.Reader(["en"], gpu=False)
    return _OCR_READER


def _page_pixmap_to_array(page: fitz.Page) -> tuple[np.ndarray, int, int]:
    # Render at 2x for better OCR readability while keeping runtime manageable.
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return arr, pix.width, pix.height


def _ocr_page(reader, page: fitz.Page, page_number: int) -> tuple[dict, list[float]]:
    image_array, width, height = _page_pixmap_to_array(page)
    predictions = reader.readtext(image_array, detail=1)

    block_items: list[dict] = []
    page_lines: list[str] = []
    confidences: list[float] = []
    for idx, item in enumerate(predictions):
        if len(item) < 3:
            continue

        bbox, text, confidence = item
        if not isinstance(text, str) or not text.strip():
            continue

        xs = [float(pt[0]) for pt in bbox]
        ys = [float(pt[1]) for pt in bbox]
        x0 = max(0.0, min(xs))
        y0 = max(0.0, min(ys))
        x1 = min(float(width), max(xs))
        y1 = min(float(height), max(ys))

        cleaned = text.strip()
        page_lines.append(cleaned)
        confidences.append(float(confidence))
        block_items.append(
            {
                "block_no": idx,
                "block_type": 0,
                "bbox": [x0, y0, x1, y1],
                "text": cleaned,
                "confidence": round(float(confidence), 4),
            }
        )

    page_payload = {
        "page_number": page_number,
        "text": "\n".join(page_lines),
        "blocks": block_items,
    }
    return page_payload, confidences


def extract_text_surya(file_path: str) -> dict:
    """Extract text from scanned PDF pages using OCR."""
    pdf_path = Path(file_path)
    if pdf_path.suffix.lower() != ".pdf" or not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    reader = _get_ocr_reader()

    doc = fitz.open(pdf_path)
    try:
        pages: list[dict] = []
        full_text_parts: list[str] = []
        all_confidences: list[float] = []

        for page_index, page in enumerate(doc, start=1):
            page_payload, confidences = _ocr_page(reader, page, page_index)
            pages.append(page_payload)
            all_confidences.extend(confidences)
            if page_payload["text"]:
                full_text_parts.append(page_payload["text"])

        full_text = "\n\n".join(full_text_parts)
        parsed_rows = parse_table_rows(
            [block for page in pages for block in page.get("blocks", [])]
        )

        avg_conf = round(sum(all_confidences) / len(all_confidences), 4) if all_confidences else 0.0
        return {
            "document_id": pdf_path.stem,
            "source_path": str(pdf_path),
            "engine": "surya",
            "pages": pages,
            "full_text": full_text,
            "tables": [
                {
                    "table_id": "ocr_heuristic_rows",
                    "parser": "regex_heuristic",
                    "rows": parsed_rows,
                }
            ],
            "metadata": {
                "page_count": len(doc),
                "has_text": bool(full_text.strip()),
                "ocr_blocks": sum(len(page.get("blocks", [])) for page in pages),
                "ocr_avg_confidence": avg_conf,
                "parsed_row_count": len(parsed_rows),
            },
        }
    finally:
        doc.close()

