"""Generate weak Label Studio pre-labels for blood report images using OCR."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image


RANGE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:-|to)\s*\d+(?:\.\d+)?\b", re.IGNORECASE)
VALUE_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
UNIT_RE = re.compile(
    r"^(?:mg/dL|mmol/L|g/dL|mIU/L|IU/L|U/L|ng/mL|pg|fL|%|10\^\d+/uL|10\^\d+/L|mL/min/1\.73\s*m2)$",
    re.IGNORECASE,
)

BIOMARKER_TERMS = {
    "hemoglobin",
    "hgb",
    "wbc",
    "rbc",
    "platelet",
    "platelets",
    "creatinine",
    "potassium",
    "sodium",
    "glucose",
    "cholesterol",
    "triglycerides",
    "hdl",
    "ldl",
    "tsh",
    "t3",
    "t4",
    "crp",
    "ferritin",
}


def _classify_text(text: str) -> str | None:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return None

    if RANGE_RE.search(cleaned):
        return "REFERENCE_RANGE"
    if UNIT_RE.match(cleaned):
        return "UNIT"
    if VALUE_RE.match(cleaned):
        return "VALUE"

    lowered = cleaned.lower()
    if any(term in lowered for term in BIOMARKER_TERMS):
        return "BIOMARKER"
    if lowered in {"high", "low", "normal", "abnormal"}:
        return "STATUS"
    return None


def _bbox_to_percent(bbox: list[list[float]], image_w: int, image_h: int) -> dict[str, float]:
    xs = [pt[0] for pt in bbox]
    ys = [pt[1] for pt in bbox]
    x_min = max(0.0, min(xs))
    y_min = max(0.0, min(ys))
    x_max = min(float(image_w), max(xs))
    y_max = min(float(image_h), max(ys))

    width = max(0.1, x_max - x_min)
    height = max(0.1, y_max - y_min)
    return {
        "x": round((x_min / image_w) * 100.0, 4),
        "y": round((y_min / image_h) * 100.0, 4),
        "width": round((width / image_w) * 100.0, 4),
        "height": round((height / image_h) * 100.0, 4),
    }


def _load_reader(languages: list[str]):
    import easyocr

    return easyocr.Reader(languages, gpu=False)


def generate(
    tasks_path: Path,
    output_path: Path,
    workspace_root: Path,
    limit: int | None,
    min_confidence: float,
    languages: list[str],
) -> tuple[int, int]:
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise ValueError("Expected task JSON list")

    reader = _load_reader(languages)
    used = tasks[:limit] if limit and limit > 0 else tasks

    emitted = 0
    with_predictions = 0
    for task in used:
        data = task.get("data", {})
        rel_image_path = data.get("image_path")
        if not rel_image_path:
            continue

        image_path = (workspace_root / rel_image_path).resolve()
        if not image_path.exists():
            continue

        with Image.open(image_path) as image:
            image_w, image_h = image.size

        ocr_results = reader.readtext(str(image_path), detail=1)
        pred_results: list[dict] = []
        for idx, ocr_item in enumerate(ocr_results, start=1):
            if len(ocr_item) < 3:
                continue
            bbox, text, conf = ocr_item
            if float(conf) < min_confidence:
                continue

            label = _classify_text(str(text))
            if not label:
                continue

            pct = _bbox_to_percent(bbox, image_w=image_w, image_h=image_h)
            pred_results.append(
                {
                    "id": f"pred_{task.get('id', 'x')}_{idx}",
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "original_width": image_w,
                    "original_height": image_h,
                    "image_rotation": 0,
                    "value": {
                        **pct,
                        "rotation": 0,
                        "rectanglelabels": [label],
                    },
                    "score": round(float(conf), 4),
                }
            )
            emitted += 1

        if pred_results:
            task["predictions"] = [{"model_version": "easyocr-weak-v1", "result": pred_results}]
            with_predictions += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(used, indent=2), encoding="utf-8")
    return with_predictions, emitted


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate weak OCR pre-labels for Label Studio tasks")
    parser.add_argument(
        "--tasks",
        default="data/annotations/label_studio_exports/image_tasks_lbmaske.json",
        help="Input Label Studio tasks JSON",
    )
    parser.add_argument(
        "--output",
        default="data/annotations/label_studio_exports/image_tasks_lbmaske_weaklabels.json",
        help="Output JSON with predictions",
    )
    parser.add_argument("--workspace-root", default=".", help="Workspace root path")
    parser.add_argument("--limit", type=int, default=25, help="Optional max number of tasks, 0 for all")
    parser.add_argument("--min-confidence", type=float, default=0.45, help="Minimum OCR confidence")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        help="EasyOCR language codes",
    )
    args = parser.parse_args()

    with_predictions, emitted = generate(
        tasks_path=Path(args.tasks).resolve(),
        output_path=Path(args.output).resolve(),
        workspace_root=Path(args.workspace_root).resolve(),
        limit=(args.limit if args.limit > 0 else None),
        min_confidence=float(args.min_confidence),
        languages=args.languages,
    )
    print(f"[OK] Tasks with weak predictions: {with_predictions}")
    print(f"[OK] Total predicted entities: {emitted}")
    print(f"[OK] Output file: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
