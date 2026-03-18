"""Knowledge-base loading utilities for recommendation retrieval."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _load_lifestyle_docs(path: Path) -> list[dict]:
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    docs: list[dict] = []

    if isinstance(payload, list):
        for idx, item in enumerate(payload):
            if isinstance(item, str) and item.strip():
                docs.append(
                    {
                        "id": f"lifestyle_{idx}",
                        "text": item.strip(),
                        "source": "lifestyle_advice.json",
                    }
                )
            elif isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                if text:
                    docs.append(
                        {
                            "id": str(item.get("id", f"lifestyle_{idx}")),
                            "text": text,
                            "source": str(item.get("source", "lifestyle_advice.json")),
                        }
                    )

    return docs


def _load_usda_docs(path: Path) -> list[dict]:
    if not path.exists():
        return []

    docs: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            item = str(row.get("item", "")).strip()
            nutrient = str(row.get("nutrient", "")).strip()
            amount = str(row.get("amount", "")).strip()
            unit = str(row.get("unit", "")).strip()
            source = str(row.get("source", "usda_foods.csv")).strip() or "usda_foods.csv"

            if not item and not nutrient:
                continue

            text = f"Food: {item}. Nutrient: {nutrient}. Amount: {amount} {unit}."
            docs.append(
                {
                    "id": f"usda_{idx}",
                    "text": text.strip(),
                    "source": source,
                }
            )

    return docs


def load_recommendation_docs(base_dir: str = "knowledge_base/nutrition") -> list[dict]:
    """Load recommendation corpus docs from knowledge base files."""
    kb_dir = Path(base_dir)
    lifestyle = _load_lifestyle_docs(kb_dir / "lifestyle_advice.json")
    usda = _load_usda_docs(kb_dir / "usda_foods.csv")
    return lifestyle + usda
