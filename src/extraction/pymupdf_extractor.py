"""Digital PDF extraction using PyMuPDF."""

from pathlib import Path

import fitz

from src.extraction.table_parser import parse_table_rows


def extract_text_pymupdf(file_path: str) -> dict:
    """Extract page text and lightweight layout blocks from a digital PDF."""
    pdf_path = Path(file_path)
    if pdf_path.suffix.lower() != ".pdf" or not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc = fitz.open(pdf_path)
    try:
        pages: list[dict] = []
        full_text_parts: list[str] = []

        for page_index, page in enumerate(doc, start=1):
            page_text = page.get_text("text") or ""
            page_blocks = page.get_text("blocks") or []

            blocks = []
            for block in page_blocks:
                x0, y0, x1, y1, text, block_no, block_type = block
                blocks.append(
                    {
                        "block_no": int(block_no),
                        "block_type": int(block_type),
                        "bbox": [float(x0), float(y0), float(x1), float(y1)],
                        "text": (text or "").strip(),
                    }
                )

            pages.append(
                {
                    "page_number": page_index,
                    "text": page_text.strip(),
                    "blocks": blocks,
                }
            )
            full_text_parts.append(page_text.strip())

        full_text = "\n\n".join(part for part in full_text_parts if part)
        parsed_rows = parse_table_rows(
            [block for page in pages for block in page.get("blocks", [])]
        )

        return {
            "document_id": pdf_path.stem,
            "source_path": str(pdf_path),
            "engine": "pymupdf",
            "pages": pages,
            "full_text": full_text,
            "tables": [
                {
                    "table_id": "heuristic_rows",
                    "parser": "regex_heuristic",
                    "rows": parsed_rows,
                }
            ],
            "metadata": {
                "page_count": len(doc),
                "has_text": bool(full_text),
                "parsed_row_count": len(parsed_rows),
            },
        }
    finally:
        doc.close()

