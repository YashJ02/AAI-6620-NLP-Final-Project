"""Scanned PDF extraction using Surya OCR."""

from __future__ import annotations

from pathlib import Path

import fitz
import numpy as np

from src.extraction.table_parser import parse_table_rows


_OCR_READER = None
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


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


def _ocr_predictions_to_page_payload(
    predictions: list,
    page_number: int,
    width: float | None = None,
    height: float | None = None,
) -> tuple[dict, list[float]]:
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
        x0 = min(xs)
        y0 = min(ys)
        x1 = max(xs)
        y1 = max(ys)

        if width is not None and height is not None:
            x0 = max(0.0, x0)
            y0 = max(0.0, y0)
            x1 = min(float(width), x1)
            y1 = min(float(height), y1)

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


def _extract_from_pdf(reader, pdf_path: Path) -> tuple[list[dict], str, list[float], int]:
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
        return pages, full_text, all_confidences, len(doc)
    finally:
        doc.close()


def _extract_from_image(reader, image_path: Path) -> tuple[list[dict], str, list[float], int]:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(
            "Pillow is required for image OCR inputs. Install dependencies from requirements.txt."
        ) from exc

    image_array = np.array(Image.open(image_path).convert("RGB"))
    predictions = reader.readtext(image_array, detail=1)
    page_payload, confidences = _ocr_predictions_to_page_payload(predictions, page_number=1)
    full_text = page_payload["text"]
    return [page_payload], full_text, confidences, 1


def extract_text_surya(file_path: str) -> dict:
    """Extract text from scanned PDFs or image files using OCR."""
    source_path = Path(file_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    suffix = source_path.suffix.lower()
    if suffix != ".pdf" and suffix not in _IMAGE_EXTENSIONS:
        raise FileNotFoundError(f"Unsupported OCR input type: {file_path}")

    reader = _get_ocr_reader()

    if suffix == ".pdf":
        pages, full_text, all_confidences, page_count = _extract_from_pdf(reader, source_path)
        source_type = "pdf"
    else:
        pages, full_text, all_confidences, page_count = _extract_from_image(reader, source_path)
        source_type = "image"

    parsed_rows = parse_table_rows([block for page in pages for block in page.get("blocks", [])])
    avg_conf = round(sum(all_confidences) / len(all_confidences), 4) if all_confidences else 0.0

    return {
        "document_id": source_path.stem,
        "source_path": str(source_path),
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
            "page_count": page_count,
            "source_type": source_type,
            "has_text": bool(full_text.strip()),
            "ocr_blocks": sum(len(page.get("blocks", [])) for page in pages),
            "ocr_avg_confidence": avg_conf,
            "parsed_row_count": len(parsed_rows),
        },
    }

