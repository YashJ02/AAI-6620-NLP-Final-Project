from src.extraction.table_parser import parse_table_rows


def test_parse_table_rows_from_text_blocks():
    blocks = [
        {"text": "Hemoglobin 12.5 g/dL 13.0-17.0"},
        {"text": "TSH: 6.2 mIU/L 0.4-4.0"},
        {"text": "Random heading line"},
    ]

    rows = parse_table_rows(blocks)

    assert len(rows) >= 2
    assert rows[0]["biomarker"]
    assert rows[0]["value"]
    assert "biomarker_normalized" in rows[0]
    assert "confidence" in rows[0]
    assert 0.0 <= rows[0]["confidence"] <= 1.0


def test_parse_table_rows_deduplicates_rows():
    blocks = [
        {"text": "Glucose 102 mg/dL 70-99"},
        {"text": "Glucose 102 mg/dL 70-99"},
    ]

    rows = parse_table_rows(blocks)

    assert len(rows) == 1


def test_parse_table_rows_normalizes_common_aliases():
    blocks = [
        {"text": "Hgb 11.9 g/dL 13.0-17.0"},
    ]

    rows = parse_table_rows(blocks)

    assert len(rows) == 1
    assert rows[0]["biomarker_normalized"] == "Hemoglobin"
