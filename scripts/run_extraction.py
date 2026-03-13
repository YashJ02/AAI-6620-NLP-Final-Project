"""Command-line runner for extraction stage."""

import argparse
import json
from pathlib import Path

from src.extraction.pymupdf_extractor import extract_text_pymupdf
from src.extraction.router import route_pdf
from src.extraction.surya_ocr_extractor import extract_text_surya


def _collect_pdfs(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("*.pdf"))
    return []


def _extract_one(pdf_path: Path) -> dict:
    engine = route_pdf(str(pdf_path))
    if engine == "pymupdf":
        return extract_text_pymupdf(str(pdf_path))
    return extract_text_surya(str(pdf_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 PDF extraction")
    parser.add_argument("--input", required=True, help="PDF file or directory containing PDFs")
    parser.add_argument(
        "--output-dir",
        default="data/interim/extracted_text",
        help="Directory to write extracted JSON outputs",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = _collect_pdfs(input_path)
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found at: {input_path}")

    processed = 0
    for pdf_path in pdf_files:
        extracted = _extract_one(pdf_path)
        output_path = output_dir / f"{pdf_path.stem}.json"
        output_path.write_text(json.dumps(extracted, indent=2), encoding="utf-8")
        processed += 1
        print(f"[OK] {pdf_path.name} -> {output_path}")

    print(f"Completed extraction for {processed} file(s).")


if __name__ == "__main__":
    main()

