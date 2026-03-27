from pathlib import Path

import pytest

from src.extraction.router import _decide_engine_from_text_lengths
from src.extraction.router import is_supported_document
from src.extraction.router import route_document
from src.extraction.router import route_pdf


def test_route_prefers_pymupdf_for_text_rich_pages():
    assert _decide_engine_from_text_lengths([120, 40, 12]) == "pymupdf"


def test_route_prefers_surya_for_low_text_pages():
    assert _decide_engine_from_text_lengths([10, 6, 2]) == "surya"


def test_route_prefers_surya_when_empty_signal():
    assert _decide_engine_from_text_lengths([]) == "surya"


def test_is_supported_document_for_image_suffix():
    assert is_supported_document("sample_image.png")
    assert is_supported_document(Path("sample_scan.jpeg"))


def test_route_document_routes_images_to_surya(tmp_path: Path):
    image_path = tmp_path / "report_page.png"
    image_path.write_bytes(b"placeholder")

    assert route_document(str(image_path)) == "surya"


def test_route_pdf_rejects_non_pdf_file(tmp_path: Path):
    image_path = tmp_path / "not_pdf.jpg"
    image_path.write_bytes(b"placeholder")

    with pytest.raises(FileNotFoundError):
        route_pdf(str(image_path))

