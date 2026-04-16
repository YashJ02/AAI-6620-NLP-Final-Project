"""Download external datasets listed in data/raw/sources_manifest.csv.

Manifest columns (required):
- source_name
- source_url

Manifest columns (optional):
- license
- notes
- destination_relpath (path under output root)
- sha256 (hex digest for integrity check)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def _safe_filename(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return cleaned.strip("_") or "dataset"


def _hash_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


def _download(url: str, destination: Path, timeout: int, retries: int) -> tuple[bool, str]:
    last_err = "unknown_error"
    destination.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "AAI-6620-NLP-Final-Project/1.0"})
            with urlopen(req, timeout=timeout) as response, destination.open("wb") as out:
                out.write(response.read())
            return True, "ok"
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            if attempt < retries:
                time.sleep(min(5, attempt))

    return False, last_err


def _infer_filename(source_name: str, source_url: str) -> str:
    parsed = urlparse(source_url)
    tail = Path(parsed.path).name
    if tail:
        return _safe_filename(tail)
    return _safe_filename(source_name) + ".dat"


def download_from_manifest(
    manifest_csv: Path,
    output_root: Path,
    timeout: int,
    retries: int,
) -> dict:
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

    rows: list[dict] = []
    with manifest_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    results: list[dict] = []
    attempted = 0
    succeeded = 0

    for row in rows:
        source_name = str(row.get("source_name", "")).strip()
        source_url = str(row.get("source_url", "")).strip()
        if not source_url:
            continue

        if not source_url.startswith("http://") and not source_url.startswith("https://"):
            results.append(
                {
                    "source_name": source_name,
                    "source_url": source_url,
                    "status": "skipped",
                    "reason": "unsupported_url_scheme",
                }
            )
            continue

        attempted += 1
        dest_rel = str(row.get("destination_relpath", "")).strip()
        if dest_rel:
            dest_path = output_root / dest_rel
        else:
            dest_path = output_root / _infer_filename(source_name=source_name, source_url=source_url)

        ok, reason = _download(source_url, dest_path, timeout=timeout, retries=retries)

        expected_sha = str(row.get("sha256", "")).strip().lower()
        actual_sha = _hash_file(dest_path) if ok and dest_path.exists() else ""
        if ok and expected_sha and actual_sha != expected_sha:
            ok = False
            reason = "sha256_mismatch"

        if ok:
            succeeded += 1
            status = "downloaded"
            size_bytes = int(dest_path.stat().st_size)
        else:
            status = "failed"
            size_bytes = int(dest_path.stat().st_size) if dest_path.exists() else 0

        results.append(
            {
                "source_name": source_name,
                "source_url": source_url,
                "destination": str(dest_path),
                "license": str(row.get("license", "")).strip(),
                "status": status,
                "reason": reason,
                "size_bytes": size_bytes,
                "sha256": actual_sha,
            }
        )

    return {
        "manifest": str(manifest_csv),
        "output_root": str(output_root),
        "attempted": attempted,
        "succeeded": succeeded,
        "failed": max(0, attempted - succeeded),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets listed in a CSV manifest")
    parser.add_argument("--manifest", default="data/raw/sources_manifest.csv", help="CSV manifest path")
    parser.add_argument("--output-root", default="datasets/external", help="Download output directory")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Retry attempts per file")
    parser.add_argument(
        "--report",
        default="artifacts/metrics/download_sources_report.json",
        help="Path to write JSON download report",
    )
    args = parser.parse_args()

    report = download_from_manifest(
        manifest_csv=Path(args.manifest),
        output_root=Path(args.output_root),
        timeout=int(args.timeout),
        retries=int(args.retries),
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[OK] Download step complete")
    print(json.dumps({
        "attempted": report["attempted"],
        "succeeded": report["succeeded"],
        "failed": report["failed"],
        "report": str(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
