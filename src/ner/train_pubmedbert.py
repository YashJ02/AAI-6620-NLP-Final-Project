"""Train PubMedBERT token classification model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers import Trainer
from transformers import TrainingArguments
from transformers import set_seed


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


def _build_compute_metrics(o_label_id: int):
    def _compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        valid_mask = labels != -100
        if not np.any(valid_mask):
            return {
                "token_accuracy": 0.0,
                "entity_token_precision": 0.0,
                "entity_token_recall": 0.0,
                "entity_token_f1": 0.0,
                "entity_f1": 0.0,
            }

        accuracy = float((preds[valid_mask] == labels[valid_mask]).mean())

        true_entity = (labels != o_label_id) & valid_mask
        pred_entity = (preds != o_label_id) & valid_mask
        exact_entity_match = (preds == labels) & true_entity
        entity_label_mismatch = pred_entity & true_entity & (preds != labels)

        tp = int(np.sum(exact_entity_match))
        fp = int(np.sum((pred_entity & ~true_entity) | entity_label_mismatch))
        fn = int(np.sum((true_entity & ~pred_entity) | entity_label_mismatch))

        precision = (tp / (tp + fp)) if (tp + fp) else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        return {
            "token_accuracy": round(accuracy, 4),
            "entity_token_precision": round(float(precision), 4),
            "entity_token_recall": round(float(recall), 4),
            "entity_token_f1": round(float(f1), 4),
            "entity_f1": round(float(f1), 4),
        }

    return _compute_metrics


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
    o_label_id = int(label_to_id.get("O", 0))

    seed = int(training_cfg.get("seed", 42))
    set_seed(seed)

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

    warmup_steps = int(training_cfg.get("warmup_steps", 0))
    args_kwargs = {
        "output_dir": str(out_dir),
        "learning_rate": float(training_cfg.get("learning_rate", 3e-5)),
        "per_device_train_batch_size": int(training_cfg.get("batch_size", 16)),
        "per_device_eval_batch_size": int(training_cfg.get("batch_size", 16)),
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 1)),
        "num_train_epochs": int(training_cfg.get("epochs", 8)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.01)),
        "lr_scheduler_type": str(training_cfg.get("lr_scheduler_type", "linear")),
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": str(training_cfg.get("metric_for_best_model", "entity_f1")),
        "greater_is_better": bool(training_cfg.get("greater_is_better", True)),
        "logging_steps": int(training_cfg.get("logging_steps", 50)),
        "seed": seed,
        "data_seed": seed,
        "save_total_limit": int(training_cfg.get("save_total_limit", 2)),
        "bf16": bool(training_cfg.get("bf16", False)),
        "fp16": bool(training_cfg.get("fp16", True)),
        "report_to": [],
    }

    if args_kwargs["bf16"]:
        args_kwargs["fp16"] = False

    if warmup_steps > 0:
        args_kwargs["warmup_steps"] = warmup_steps
    else:
        args_kwargs["warmup_ratio"] = float(training_cfg.get("warmup_ratio", 0.1))

    args = TrainingArguments(**args_kwargs)

    callbacks = []
    early_cfg = training_cfg.get("early_stopping", {}) or {}
    early_patience = int(early_cfg.get("patience", 0))
    if early_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_patience,
                early_stopping_threshold=float(early_cfg.get("threshold", 0.0)),
            )
        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_build_compute_metrics(o_label_id=o_label_id),
        callbacks=callbacks,
    )

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

