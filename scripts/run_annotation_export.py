"""Export and transform Label Studio annotations."""

import argparse

from src.ner.dataset_builder import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Label Studio exports to BIO JSONL")
    parser.add_argument(
        "--input-dir",
        default="data/annotations/label_studio_exports",
        help="Directory containing Label Studio JSON export files",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to write train/val/test JSONL outputs",
    )
    args = parser.parse_args()

    build_dataset(annotation_dir=args.input_dir, output_dir=args.output_dir)
    print(f"[OK] Converted Label Studio exports from {args.input_dir} to {args.output_dir}")


if __name__ == "__main__":
    main()

