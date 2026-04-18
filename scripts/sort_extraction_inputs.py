"""Sort extraction-related files from datasets/ into data/raw ingestion folders."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from _bootstrap import ensure_project_root_on_path


ensure_project_root_on_path()

from src.extraction.router import route_document


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
MEDIA_EXTENSIONS = {".pdf", *IMAGE_EXTENSIONS}
SOURCE_DATASET_DIR = Path("datasets")


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 1
    while True:
        candidate = parent / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def _is_extraction_image(path: Path) -> bool:
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False

    lower_parts = {part.lower() for part in path.parts}
    if "lbmaske" in lower_parts:
        return True
    if path.parent == SOURCE_DATASET_DIR and "nurseslabs" in path.name.lower():
        return True
    return False


def _choose_image_target(path: Path, raw_dir: Path) -> Path:
    lower_parts = {part.lower() for part in path.parts}
    if "lbmaske" in lower_parts:
        return raw_dir / "images_scanned" / "lbmaske" / path.name
    return raw_dir / "images_reference" / "nurseslabs" / path.name


def _choose_pdf_target(path: Path, raw_dir: Path) -> Path:
    engine = route_document(str(path))
    folder = "pdfs_digital" if engine == "pymupdf" else "pdfs_scanned"
    return raw_dir / folder / path.name


def _collect_files(dataset_dir: Path) -> tuple[list[Path], list[Path]]:
    pdfs: list[Path] = []
    images: list[Path] = []

    for path in dataset_dir.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pdfs.append(path)
        elif _is_extraction_image(path):
            images.append(path)

    return sorted(pdfs), sorted(images)


def _collect_remaining_media(
    dataset_dir: Path,
    extraction_pdf_set: set[Path],
    extraction_image_set: set[Path],
    archive_dir: Path,
) -> list[Path]:
    remaining: list[Path] = []

    for path in dataset_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in MEDIA_EXTENSIONS:
            continue

        if path in extraction_pdf_set or path in extraction_image_set:
            continue

        # Skip files already inside archive target.
        if archive_dir in path.parents:
            continue

        remaining.append(path)

    return sorted(remaining)


def _move_file(source: Path, target: Path, apply_changes: bool) -> tuple[Path, Path]:
    destination = _unique_destination(target)
    if apply_changes:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    return source, destination


def sort_inputs(
    dataset_dir: Path,
    raw_dir: Path,
    apply_changes: bool,
    archive_remaining_media: bool,
    archive_dir: Path,
) -> dict:
    pdfs, images = _collect_files(dataset_dir)
    pdf_set = set(pdfs)
    image_set = set(images)
    remaining_media = (
        _collect_remaining_media(dataset_dir, pdf_set, image_set, archive_dir)
        if archive_remaining_media
        else []
    )

    moved_pdfs: list[dict] = []
    moved_images: list[dict] = []
    archived_media: list[dict] = []
    errors: list[dict] = []

    for pdf in pdfs:
        try:
            target = _choose_pdf_target(pdf, raw_dir)
            src, dst = _move_file(pdf, target, apply_changes=apply_changes)
            moved_pdfs.append({"from": str(src), "to": str(dst)})
        except Exception as exc:  # pragma: no cover - defensive logging
            errors.append({"file": str(pdf), "error": str(exc)})

    for image in images:
        try:
            target = _choose_image_target(image, raw_dir)
            src, dst = _move_file(image, target, apply_changes=apply_changes)
            moved_images.append({"from": str(src), "to": str(dst)})
        except Exception as exc:  # pragma: no cover - defensive logging
            errors.append({"file": str(image), "error": str(exc)})

    for media in remaining_media:
        try:
            rel_path = media.relative_to(dataset_dir)
            target = archive_dir / rel_path
            src, dst = _move_file(media, target, apply_changes=apply_changes)
            archived_media.append({"from": str(src), "to": str(dst)})
        except Exception as exc:  # pragma: no cover - defensive logging
            errors.append({"file": str(media), "error": str(exc)})

    return {
        "apply_changes": apply_changes,
        "dataset_dir": str(dataset_dir),
        "raw_dir": str(raw_dir),
        "archive_remaining_media": archive_remaining_media,
        "archive_dir": str(archive_dir),
        "counts": {
            "candidate_pdfs": len(pdfs),
            "candidate_images": len(images),
            "remaining_media_candidates": len(remaining_media),
            "moved_pdfs": len(moved_pdfs),
            "moved_images": len(moved_images),
            "archived_media": len(archived_media),
            "errors": len(errors),
        },
        "moved_pdfs": moved_pdfs,
        "moved_images": moved_images,
        "archived_media": archived_media,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sort extraction input files into data/raw folders")
    parser.add_argument("--dataset-dir", default="datasets", help="Source datasets directory")
    parser.add_argument("--raw-dir", default="data/raw", help="Target raw ingestion directory")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply filesystem moves. Without this flag, command runs in dry-run mode.",
    )
    parser.add_argument(
        "--archive-remaining-media",
        action="store_true",
        help="Archive media files not selected as extraction inputs into an archive folder under datasets.",
    )
    parser.add_argument(
        "--archive-dir",
        default="datasets/archive/non_extraction_media",
        help="Archive directory used with --archive-remaining-media.",
    )
    parser.add_argument(
        "--report",
        default="artifacts/metrics/sort_extraction_inputs_report.json",
        help="Path to write sorting summary report JSON",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    raw_dir = Path(args.raw_dir)
    archive_dir = Path(args.archive_dir)

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    summary = sort_inputs(
        dataset_dir=dataset_dir,
        raw_dir=raw_dir,
        apply_changes=args.apply,
        archive_remaining_media=args.archive_remaining_media,
        archive_dir=archive_dir,
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["counts"], indent=2))
    print(f"[INFO] Report written: {report_path}")


if __name__ == "__main__":
    main()
