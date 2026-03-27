import pytest

from src.ner.infer_pubmedbert import predict


def test_predict_returns_empty_for_blank_text():
    assert predict(text="") == []
    assert predict(text="   \n\t") == []


def test_predict_raises_for_missing_model_directory(tmp_path):
    missing_model_dir = tmp_path / "missing_model"
    with pytest.raises(FileNotFoundError):
        predict(text="hemoglobin 12.5", model_dir=str(missing_model_dir))

