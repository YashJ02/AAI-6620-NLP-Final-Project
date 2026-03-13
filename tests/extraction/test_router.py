from src.extraction.router import _decide_engine_from_text_lengths


def test_route_prefers_pymupdf_for_text_rich_pages():
    assert _decide_engine_from_text_lengths([120, 40, 12]) == "pymupdf"


def test_route_prefers_surya_for_low_text_pages():
    assert _decide_engine_from_text_lengths([10, 6, 2]) == "surya"


def test_route_prefers_surya_when_empty_signal():
    assert _decide_engine_from_text_lengths([]) == "surya"

