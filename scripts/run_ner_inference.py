"""Command-line runner for NER inference."""

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_project_root_on_path


ensure_project_root_on_path()

from src.ner.infer_pubmedbert import predict


def _read_text_from_input(input_path: Path) -> str:
    if input_path.suffix.lower() == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if isinstance(payload.get("full_text"), str):
                return payload["full_text"]
            if isinstance(payload.get("text"), str):
                return payload["text"]
    return input_path.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NER inference on text or extracted JSON")
    parser.add_argument("--input", required=True, help="Path to .txt or extraction .json file")
    parser.add_argument(
        "--model-dir",
        default="artifacts/models/pubmedbert_ner/model",
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--output",
        default="artifacts/sample_outputs/ner_predictions.json",
        help="Path to write entity predictions JSON",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    text = _read_text_from_input(input_path)
    entities = predict(text=text, model_dir=args.model_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"entities": entities}, indent=2), encoding="utf-8")

    print(f"[OK] Saved {len(entities)} entities to {output_path}")


if __name__ == "__main__":
    main()
