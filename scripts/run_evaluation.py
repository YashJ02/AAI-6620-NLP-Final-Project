"""Command-line runner for NER and retrieval evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _bootstrap import ensure_project_root_on_path


ensure_project_root_on_path()

import torch
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer

from src.recommendation.service import generate_recommendations


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _evaluate_ner(test_rows: list[dict[str, Any]], model_dir: Path, max_length: int = 256) -> dict[str, Any]:
    if not test_rows:
        return {"status": "skipped", "reason": "empty_test_set"}
    if not model_dir.exists():
        return {"status": "skipped", "reason": f"model_dir_missing:{model_dir}"}

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    model.eval()

    label2id = model.config.label2id or {"O": 0}
    id2label = {int(v): str(k) for k, v in label2id.items()}
    o_id = int(label2id.get("O", 0))

    total = 0
    correct = 0
    tp = 0
    fp = 0
    fn = 0

    for row in test_rows:
        tokens = row.get("tokens", [])
        tags = row.get("ner_tags", [])
        if not tokens or not tags:
            continue

        encoded = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        word_ids = encoded.word_ids(batch_index=0)
        true_ids: list[int] = []
        prev_word = None
        for word_idx in word_ids:
            if word_idx is None:
                true_ids.append(-100)
                continue
            if word_idx != prev_word:
                tag = tags[word_idx] if word_idx < len(tags) else "O"
                true_ids.append(int(label2id.get(tag, o_id)))
            else:
                true_ids.append(-100)
            prev_word = word_idx

        with torch.no_grad():
            outputs = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
            pred_ids = outputs.logits.argmax(dim=-1).squeeze(0).tolist()

        for t_id, p_id in zip(true_ids, pred_ids):
            if t_id == -100:
                continue
            total += 1
            if p_id == t_id:
                correct += 1

            true_is_entity = t_id != o_id
            pred_is_entity = p_id != o_id

            if true_is_entity and pred_is_entity and p_id == t_id:
                tp += 1
            elif not true_is_entity and pred_is_entity:
                fp += 1
            elif true_is_entity and not pred_is_entity:
                fn += 1
            elif true_is_entity and pred_is_entity and p_id != t_id:
                fp += 1
                fn += 1

    accuracy = (correct / total) if total else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "status": "ok",
        "token_count": total,
        "token_accuracy": round(accuracy, 4),
        "entity_token_precision": round(precision, 4),
        "entity_token_recall": round(recall, 4),
        "entity_token_f1": round(f1, 4),
        "label_space": sorted([id2label[k] for k in sorted(id2label.keys())]),
    }


def _evaluate_retrieval(benchmark_rows: list[dict[str, Any]], top_k: int = 5) -> dict[str, Any]:
    if not benchmark_rows:
        return {"status": "skipped", "reason": "missing_or_empty_benchmark"}

    precisions: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    skipped_rows = 0
    error_rows = 0

    for row in benchmark_rows:
        query = str(row.get("query", "")).strip()
        relevant_ids = [str(x) for x in row.get("relevant_ids", [])]
        if not query or not relevant_ids:
            skipped_rows += 1
            continue

        try:
            rec = generate_recommendations(
                interpreted_rows=[],
                ner_entities=[],
                status_summary={"low": 0, "normal": 0, "high": 0, "unknown": 0},
                patient_id="eval",
                query=query,
                top_k=top_k,
            )
        except Exception:
            error_rows += 1
            continue

        result_ids = [str(item.get("id", "")) for item in rec.get("results", [])[:top_k]]

        hits = [rid for rid in result_ids if rid in relevant_ids]
        precision = len(hits) / max(1, top_k)
        recall = len(hits) / max(1, len(relevant_ids))

        reciprocal_rank = 0.0
        for idx, rid in enumerate(result_ids, start=1):
            if rid in relevant_ids:
                reciprocal_rank = 1.0 / idx
                break

        precisions.append(precision)
        recalls.append(recall)
        mrrs.append(reciprocal_rank)

    if not precisions:
        return {"status": "skipped", "reason": "no_valid_benchmark_rows"}

    return {
        "status": "ok",
        "benchmark_row_count": len(benchmark_rows),
        "query_count": len(precisions),
        "skipped_rows": skipped_rows,
        "error_rows": error_rows,
        "precision_at_k": round(sum(precisions) / len(precisions), 4),
        "recall_at_k": round(sum(recalls) / len(recalls), 4),
        "mrr_at_k": round(sum(mrrs) / len(mrrs), 4),
        "k": int(top_k),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NER and recommendation retrieval")
    parser.add_argument("--data-dir", default="data/processed", help="Directory with test.jsonl")
    parser.add_argument("--model-dir", default="artifacts/models/pubmedbert_ner/model", help="NER model directory")
    parser.add_argument(
        "--retrieval-benchmark",
        default="data/processed/retrieval_eval.jsonl",
        help="JSONL with query and relevant_ids fields",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for retrieval metrics")
    parser.add_argument("--output", default="artifacts/metrics/evaluation_metrics.json", help="Output metrics JSON path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    top_k = max(1, int(args.top_k))

    test_rows = _read_jsonl(Path(args.data_dir) / "test.jsonl")
    benchmark_rows = _read_jsonl(Path(args.retrieval_benchmark))

    ner_metrics = _evaluate_ner(test_rows=test_rows, model_dir=Path(args.model_dir))
    retrieval_metrics = _evaluate_retrieval(benchmark_rows=benchmark_rows, top_k=top_k)

    report = {
        "ner": ner_metrics,
        "retrieval": retrieval_metrics,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[OK] Wrote evaluation report to {output_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

