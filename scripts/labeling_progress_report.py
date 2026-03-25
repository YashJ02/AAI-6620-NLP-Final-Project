"""Summarize labeling progress from image manifest CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _is_true(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def summarize(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    total = 0
    labeled = 0
    qc_pass = 0
    split_counts: Counter[str] = Counter()
    split_labeled: Counter[str] = Counter()

    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            split = str(row.get("split", "unknown") or "unknown")
            split_counts[split] += 1

            if _is_true(str(row.get("labeled", "0"))):
                labeled += 1
                split_labeled[split] += 1
            if _is_true(str(row.get("qc_pass", "0"))):
                qc_pass += 1

    def pct(part: int, whole: int) -> float:
        if whole <= 0:
            return 0.0
        return round((part / whole) * 100.0, 2)

    by_split = {}
    for split in sorted(split_counts.keys()):
        by_split[split] = {
            "total": split_counts[split],
            "labeled": split_labeled.get(split, 0),
            "labeled_pct": pct(split_labeled.get(split, 0), split_counts[split]),
        }

    return {
        "total_tasks": total,
        "labeled_tasks": labeled,
        "unlabeled_tasks": max(0, total - labeled),
        "labeling_pct": pct(labeled, total),
        "qc_pass_tasks": qc_pass,
        "qc_pass_pct": pct(qc_pass, total),
        "by_split": by_split,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Labeling progress report")
    parser.add_argument(
        "--manifest",
        default="data/annotations/label_studio_exports/image_manifest_lbmaske.csv",
        help="CSV manifest path",
    )
    args = parser.parse_args()

    summary = summarize(Path(args.manifest).resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
