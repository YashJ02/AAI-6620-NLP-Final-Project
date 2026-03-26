from pathlib import Path

import fitz
import pytest

from src.extraction.surya_ocr_extractor import extract_text_surya


def _create_scanned_style_pdf(out_path: Path) -> None:
    """Create an image-only PDF page to simulate scanned input."""
    pil = pytest.importorskip("PIL.Image")
    draw_mod = pytest.importorskip("PIL.ImageDraw")

    img = pil.new("RGB", (900, 1200), color="white")
    draw = draw_mod.Draw(img)
    draw.text((60, 120), "GLUCOSE 110 mg/dL 70-99", fill="black")
    draw.text((60, 170), "HEMOGLOBIN 12.5 g/dL 12.0-15.5", fill="black")

    png_path = out_path.with_suffix(".png")
    img.save(png_path)

    doc = fitz.open()
    page = doc.new_page(width=900, height=1200)
    page.insert_image(page.rect, filename=str(png_path))
    doc.save(out_path)
    doc.close()


def test_extract_text_surya_on_scanned_pdf(tmp_path: Path):
    try:
        import easyocr  # noqa: F401
    except Exception as exc:
        pytest.skip(f"easyocr runtime unavailable: {exc}")

    pdf_path = tmp_path / "scanned_sample.pdf"
    _create_scanned_style_pdf(pdf_path)

    result = extract_text_surya(str(pdf_path))

    assert result["engine"] == "surya"
    assert result["document_id"] == "scanned_sample"
    assert isinstance(result["pages"], list)
    assert len(result["pages"]) == 1
    assert "full_text" in result
    assert "metadata" in result
    assert "tables" in result
    assert result["metadata"]["page_count"] == 1
