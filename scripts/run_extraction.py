"""Command-line runner for extraction stage."""

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_project_root_on_path


ensure_project_root_on_path()

from src.extraction.pymupdf_extractor import extract_text_pymupdf
from src.extraction.router import is_supported_document
from src.extraction.router import route_document
from src.extraction.surya_ocr_extractor import extract_text_surya


def _collect_documents(input_path: Path) -> list[Path]:
    if input_path.is_file() and is_supported_document(input_path):
        return [input_path]
    if input_path.is_dir():
        return sorted(
            [
                candidate
                for candidate in input_path.rglob("*")
                if candidate.is_file() and is_supported_document(candidate)
            ]
        )
    return []


def _extract_one(document_path: Path) -> dict:
    engine = route_document(str(document_path))
    if engine == "pymupdf":
        return extract_text_pymupdf(str(document_path))
    return extract_text_surya(str(document_path))


def _output_name(document_path: Path) -> str:
    if document_path.suffix.lower() == ".pdf":
        return f"{document_path.stem}.json"
    suffix_slug = document_path.suffix.lower().replace(".", "")
    return f"{document_path.stem}_{suffix_slug}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 document extraction")
    parser.add_argument(
        "--input",
        required=True,
        help="Supported document file or directory (.pdf, .png, .jpg, .jpeg, .tif, .tiff, .bmp, .webp)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/interim/extracted_text",
        help="Directory to write extracted JSON outputs",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = _collect_documents(input_path)
    if not input_files:
        raise FileNotFoundError(f"No supported document files found at: {input_path}")

    processed = 0
    for input_file in input_files:
        extracted = _extract_one(input_file)
        output_path = output_dir / _output_name(input_file)
        output_path.write_text(json.dumps(extracted, indent=2), encoding="utf-8")
        processed += 1
        print(f"[OK] {input_file.name} -> {output_path}")

    print(f"Completed extraction for {processed} document(s).")


if __name__ == "__main__":
    main()

