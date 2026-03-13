import json

from src.ner.dataset_builder import _build_bio_labels
from src.ner.dataset_builder import _tokenize_with_offsets
from src.ner.dataset_builder import build_dataset


def test_build_bio_labels_marks_span_tokens():
    text = "Hemoglobin 12.5 g/dL"
    tokens = _tokenize_with_offsets(text)
    start = text.index("Hemoglobin")
    end = start + len("Hemoglobin")
    spans = [{"start": start, "end": end, "label": "BIOMARKER"}]

    tags = _build_bio_labels(tokens, spans)

    assert tags[0] == "B-BIOMARKER"
    assert "O" in tags


def test_build_dataset_writes_split_files(tmp_path):
    export = [
        {
            "data": {"text": "Hemoglobin 12.5 g/dL 13.0-17.0"},
            "annotations": [
                {
                    "result": [
                        {
                            "value": {
                                "start": 0,
                                "end": 10,
                                "labels": ["BIOMARKER"],
                            }
                        }
                    ]
                }
            ],
        },
        {
            "data": {"text": "TSH 6.2 mIU/L 0.4-4.0"},
            "annotations": [
                {
                    "result": [
                        {
                            "value": {
                                "start": 0,
                                "end": 3,
                                "labels": ["BIOMARKER"],
                            }
                        }
                    ]
                }
            ],
        },
    ]

    input_dir = tmp_path / "in"
    output_dir = tmp_path / "out"
    input_dir.mkdir()

    (input_dir / "export.json").write_text(json.dumps(export), encoding="utf-8")

    build_dataset(str(input_dir), str(output_dir))

    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "val.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()
    assert (output_dir / "metadata.json").exists()
