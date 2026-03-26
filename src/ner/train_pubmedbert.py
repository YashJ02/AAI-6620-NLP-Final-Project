"""Train PubMedBERT token classification model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_label_list(train_rows: list[dict], val_rows: list[dict]) -> list[str]:
    labels = {"O"}
    for row in train_rows + val_rows:
        for tag in row.get("ner_tags", []):
            labels.add(tag)
    return sorted(labels)


def _tokenize_and_align_labels(rows: list[dict], tokenizer, label_to_id: dict[str, int], max_length: int) -> dict[str, list]:
    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    labels: list[list[int]] = []

    for row in rows:
        tokens = row["tokens"]
        tag_list = row["ner_tags"]

        encoded = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        word_ids = encoded.word_ids()
        aligned_label_ids: list[int] = []
        previous_word = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_label_ids.append(-100)
                continue

            if word_idx != previous_word:
                aligned_label_ids.append(label_to_id[tag_list[word_idx]])
            else:
                aligned_label_ids.append(-100)
            previous_word = word_idx

        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])
        labels.append(aligned_label_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }


class TokenDataset:
    def __init__(self, data: dict[str, list]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return {
            "input_ids": self.data["input_ids"][idx],
            "attention_mask": self.data["attention_mask"][idx],
            "labels": self.data["labels"][idx],
        }


def _compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    valid_mask = labels != -100
    if not np.any(valid_mask):
        return {"token_accuracy": 0.0}

    accuracy = float((preds[valid_mask] == labels[valid_mask]).mean())
    return {"token_accuracy": round(accuracy, 4)}


def train(config_path: str, data_dir: str, output_dir: str) -> None:
    """Fine-tune a token classification model on BIO-tagged JSONL data."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    model_name = config["model"]["name"]
    training_cfg = config["training"]

    train_path = Path(data_dir) / "train.jsonl"
    val_path = Path(data_dir) / "val.jsonl"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Expected train.jsonl and val.jsonl in data directory.")

    train_rows = _read_jsonl(train_path)
    val_rows = _read_jsonl(val_path)
    if not train_rows:
        raise ValueError("Training dataset is empty.")

    label_list = _build_label_list(train_rows, val_rows)
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = int(training_cfg.get("max_length", 256))

    train_encoded = _tokenize_and_align_labels(train_rows, tokenizer, label_to_id, max_length=max_length)
    val_encoded = _tokenize_and_align_labels(val_rows, tokenizer, label_to_id, max_length=max_length)

    train_dataset = TokenDataset(train_encoded)
    val_dataset = TokenDataset(val_encoded)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        learning_rate=float(training_cfg.get("learning_rate", 2e-5)),
        per_device_train_batch_size=int(training_cfg.get("batch_size", 8)),
        per_device_eval_batch_size=int(training_cfg.get("batch_size", 8)),
        num_train_epochs=int(training_cfg.get("epochs", 5)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=int(training_cfg.get("logging_steps", 20)),
        report_to=[],
    )

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "compute_metrics": _compute_metrics,
    }

    # transformers>=5 renamed `tokenizer` to `processing_class`.
    try:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)

    trainer.train()
    trainer.save_model(str(out_dir / "model"))
    tokenizer.save_pretrained(str(out_dir / "model"))

    mapping = {
        "labels": label_list,
        "label_to_id": label_to_id,
        "id_to_label": {str(k): v for k, v in id_to_label.items()},
        "model_name": model_name,
    }
    (out_dir / "label_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")

