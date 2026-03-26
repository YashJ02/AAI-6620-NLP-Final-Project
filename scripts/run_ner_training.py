"""Command-line runner for NER training."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ner.train_pubmedbert import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PubMedBERT for token classification")
    parser.add_argument(
        "--config",
        default="configs/ner_train.yaml",
        help="Path to training configuration YAML",
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Directory containing train.jsonl and val.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/models/pubmedbert_ner",
        help="Directory for checkpoints and final model artifacts",
    )
    args = parser.parse_args()

    train(config_path=args.config, data_dir=args.data_dir, output_dir=args.output_dir)
    print(f"[OK] Training complete. Artifacts saved to {args.output_dir}")


if __name__ == "__main__":
    main()

