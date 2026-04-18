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


def test_parse_table_rows_ignores_document_noise_lines():
    blocks = [
        {"text": "Page 1"},
        {"text": "Registration on 20-06-2024"},
        {"text": "Male / 41"},
        {"text": "Hemoglobin 12.5 g/dL 13.0-17.0"},
    ]

    rows = parse_table_rows(blocks)

    assert len(rows) == 1
    assert rows[0]["biomarker_normalized"] == "Hemoglobin"


def test_parse_table_rows_ignores_reference_band_descriptors():
    blocks = [
        {"text": "Borderline High 200-239"},
        {"text": "Very High 500"},
        {"text": "LDL 160 mg/dL 0-130"},
    ]

    rows = parse_table_rows(blocks)

    assert len(rows) == 1
    assert rows[0]["biomarker_normalized"].lower() == "ldl"


def test_parse_table_rows_ignores_header_and_history_labels():
    blocks = [
        {"text": "MEDPLUS HEALTH SERVICES LIMITED H No 11 11.0-5.0 high"},
        {"text": "From previous 3 40.0-6700.0 low"},
        {"text": "Hemoglobin 12.5 g/dL 13.0-17.0"},
    ]

    rows = parse_table_rows(blocks)

    assert len(rows) == 1
    assert rows[0]["biomarker_normalized"] == "Hemoglobin"


def test_parse_table_rows_ignores_explanatory_sentence_fragments():
    blocks = [
        {"text": "Microalbuminuria is defined as an albumin:creatinine ratio of 17-299"},
        {"text": "A change in PSA of > 30 ng/mL"},
        {"text": "Vitamin D 34 ng/mL 30-100"},
    ]

    rows = parse_table_rows(blocks)

    assert len(rows) == 1
    assert rows[0]["biomarker_normalized"].lower() == "vitamin d"
