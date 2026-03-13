from src.ner.train_pubmedbert import _build_label_list


def test_build_label_list_contains_o_and_sorted_tags():
    train_rows = [{"ner_tags": ["B-BIOMARKER", "O", "B-VALUE"]}]
    val_rows = [{"ner_tags": ["B-UNIT", "O"]}]

    labels = _build_label_list(train_rows, val_rows)

    assert "O" in labels
    assert labels == sorted(labels)
    assert "B-BIOMARKER" in labels
    assert "B-VALUE" in labels
    assert "B-UNIT" in labels
