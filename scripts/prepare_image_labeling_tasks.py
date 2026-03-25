"""Prepare Label Studio import tasks and split manifest for image labeling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def _extract_doc_key(file_name: str) -> str:
    marker = ".pdf_page_"
    if marker in file_name:
        return file_name.split(marker, maxsplit=1)[0]
    return Path(file_name).stem


def _stable_split(doc_key: str) -> str:
    digest = hashlib.md5(doc_key.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) % 100
    if value < 80:
        return "train"
    if value < 90:
        return "val"
    return "test"


def _build_local_files_url(relative_posix_path: str) -> str:
    return f"/data/local-files/?d={relative_posix_path}"


def prepare_tasks(
    image_dir: Path,
    output_json: Path,
    output_manifest_csv: Path,
    workspace_root: Path,
    limit: int | None = None,
) -> tuple[int, int]:
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = sorted(image_dir.glob("*.png"))
    if limit is not None and limit > 0:
        image_paths = image_paths[:limit]

    tasks: list[dict] = []
    manifest_rows: list[dict[str, str]] = []
    unique_docs: set[str] = set()

    for idx, image_path in enumerate(image_paths, start=1):
        rel_path = image_path.relative_to(workspace_root).as_posix()
        doc_key = _extract_doc_key(image_path.name)
        split = _stable_split(doc_key)
        unique_docs.add(doc_key)

        tasks.append(
            {
                "id": idx,
                "data": {
                    "image": _build_local_files_url(rel_path),
                    "image_path": rel_path,
                    "file_name": image_path.name,
                    "doc_key": doc_key,
                    "split": split,
                },
                "annotations": [],
            }
        )

        manifest_rows.append(
            {
                "id": str(idx),
                "image_path": rel_path,
                "file_name": image_path.name,
                "doc_key": doc_key,
                "split": split,
                "labeled": "0",
                "qc_pass": "0",
            }
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(tasks, indent=2), encoding="utf-8")

    with output_manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "image_path", "file_name", "doc_key", "split", "labeled", "qc_pass"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return len(tasks), len(unique_docs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare image tasks for manual labeling")
    parser.add_argument(
        "--image-dir",
        default="datasets/lbmaske",
        help="Directory with image files to label",
    )
    parser.add_argument(
        "--output-json",
        default="data/annotations/label_studio_exports/image_tasks_lbmaske.json",
        help="Path for Label Studio import JSON",
    )
    parser.add_argument(
        "--output-manifest",
        default="data/annotations/label_studio_exports/image_manifest_lbmaske.csv",
        help="Path for CSV manifest",
    )
    parser.add_argument(
        "--workspace-root",
        default=".",
        help="Workspace root used to compute relative paths",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional number of images to include; 0 means all",
    )
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    image_dir = Path(args.image_dir).resolve()
    output_json = Path(args.output_json).resolve()
    output_manifest = Path(args.output_manifest).resolve()

    count, doc_count = prepare_tasks(
        image_dir=image_dir,
        output_json=output_json,
        output_manifest_csv=output_manifest,
        workspace_root=workspace_root,
        limit=(args.limit if args.limit > 0 else None),
    )

    print(f"[OK] Prepared {count} image tasks across {doc_count} documents")
    print(f"[OK] Label Studio JSON: {output_json}")
    print(f"[OK] Manifest CSV: {output_manifest}")


if __name__ == "__main__":
    main()
