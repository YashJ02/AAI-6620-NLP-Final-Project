"""Generate starter NER JSONL splits from normalized tabular biomarker observations."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


TOKEN_RE = re.compile(r"\S+")


def _normalize_text(value: str) -> str:
    return str(value).strip().lower().replace("_", " ")


def _format_number(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _tokenize_with_offsets(text: str) -> list[dict]:
    tokens: list[dict] = []
    for match in TOKEN_RE.finditer(text):
        tokens.append(
            {
                "text": match.group(0),
                "start": match.start(),
                "end": match.end(),
            }
        )
    return tokens


def _token_overlaps(token_start: int, token_end: int, span_start: int, span_end: int) -> bool:
    return token_start < span_end and token_end > span_start


def _spans_to_bio_tags(tokens: list[dict], spans: list[dict]) -> list[str]:
    tags = ["O"] * len(tokens)

    for span in sorted(spans, key=lambda s: (s["start"], s["end"])):
        indices = [
            idx
            for idx, tok in enumerate(tokens)
            if _token_overlaps(tok["start"], tok["end"], span["start"], span["end"])
        ]
        if not indices:
            continue

        for pos, idx in enumerate(indices):
            prefix = "B" if pos == 0 else "I"
            tags[idx] = f"{prefix}-{span['label']}"

    return tags


def _build_range_lookup(ranges_df: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float]]:
    lookup: dict[tuple[str, str], tuple[float, float]] = {}
    for _, row in ranges_df.iterrows():
        biomarker = _normalize_text(row.get("biomarker", ""))
        unit = str(row.get("unit", "")).strip()
        low = row.get("ref_low")
        high = row.get("ref_high")
        if not biomarker or pd.isna(low) or pd.isna(high):
            continue
        lookup[(biomarker, unit)] = (float(low), float(high))
        lookup[(biomarker, "")] = (float(low), float(high))
    return lookup


def _build_fallback_ranges(obs_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    fallback: dict[str, tuple[float, float]] = {}
    for biomarker, group in obs_df.groupby("biomarker"):
        values = pd.to_numeric(group["value"], errors="coerce").dropna()
        if values.empty:
            continue

        q10 = float(values.quantile(0.10))
        q90 = float(values.quantile(0.90))
        if q10 == q90:
            q10 = q10 * 0.9
            q90 = q90 * 1.1

        low, high = (q10, q90) if q10 < q90 else (min(q10, q90), max(q10, q90))
        if low == high:
            low, high = low - 1.0, high + 1.0

        fallback[_normalize_text(biomarker)] = (low, high)

    return fallback


TEMPLATES = [
    # --- clinical report style ---
    "{biomarker} : {value} {unit} ( {range} )",
    "{biomarker} : {value} {unit} (Ref: {range} )",
    "{biomarker} : {value} {unit} [Normal: {range} ]",
    "{biomarker} ......... {value} {unit} Reference Range: {range}",
    "{biomarker} - {value} {unit} Ref Range {range}",
    "{biomarker} {value} {unit} ( {range} )",
    "{biomarker} {value} {unit} Normal Range: {range}",
    # --- sentence style ---
    "The patient's {biomarker} was {value} {unit} , with a reference range of {range} .",
    "Lab results show {biomarker} at {value} {unit} ( normal {range} ) .",
    "Test: {biomarker} Result: {value} {unit} Expected: {range}",
    "{biomarker} level measured at {value} {unit} against normal range {range} .",
    "Observed {biomarker} of {value} {unit} ; reference interval {range} .",
    "Report indicates {biomarker} is {value} {unit} , normal being {range} .",
    "{biomarker} was found to be {value} {unit} compared to the expected {range} .",
    # --- table-like / OCR style ---
    "{biomarker} | {value} | {unit} | {range}",
    "{biomarker} .. {value} {unit} .. {range}",
    "Investigation: {biomarker} Value: {value} Unit: {unit} Ref: {range}",
    "Parameter {biomarker} Observed {value} {unit} Bio Ref {range}",
    # --- discharge summary style ---
    "During admission , {biomarker} was recorded as {value} {unit} ( ref {range} ) .",
    "On investigation , {biomarker} came back {value} {unit} , within the reference of {range} .",
    "Blood work revealed {biomarker} at {value} {unit} , reference {range} .",
    "Complete blood count shows {biomarker} {value} {unit} ( {range} ) .",
    "Biochemistry panel: {biomarker} {value} {unit} Normal {range} .",
    # --- multi-entity context ---
    "The {biomarker} value of {value} {unit} falls within reference {range} for this age group .",
    "Patient presented with {biomarker} of {value} {unit} against expected {range} on day 3 of admission .",
    "{biomarker} reading was {value} {unit} . Normal reference range is {range} . No action required .",
    "Urgent: {biomarker} recorded {value} {unit} ( expected {range} ) - please review .",
]

NOISE_PREFIXES = [
    "", "", "",  # empty most of the time
    "Page 2 of 5 ",
    "Date: 2024-01-15 ",
    "Patient ID: XXXX ",
    "Dr. Smith Lab Report ",
    "CONFIDENTIAL ",
    "Hospital Lab Services ",
]

NOISE_SUFFIXES = [
    "", "", "",
    " Specimen: Blood",
    " Method: Automated",
    " Status: Final",
    " Collected: Morning",
    " Fasting: Yes",
]


def _make_example(
    *,
    example_id: int,
    biomarker: str,
    value: float,
    unit: str,
    ref_low: float,
    ref_high: float,
    rng: random.Random | None = None,
) -> dict:
    if rng is None:
        rng = random.Random(example_id)

    biomarker_text = str(biomarker).replace("_", " ").strip()
    # Random casing variations
    case_roll = rng.random()
    if case_roll < 0.15:
        biomarker_text = biomarker_text.upper()
    elif case_roll < 0.30:
        biomarker_text = biomarker_text.lower()

    value_text = _format_number(float(value))
    unit_text = str(unit).strip() if str(unit).strip() else "unitless"
    range_text = f"{_format_number(ref_low)}-{_format_number(ref_high)}"
    # Alternative range formats
    range_roll = rng.random()
    if range_roll < 0.2:
        range_text = f"{_format_number(ref_low)} - {_format_number(ref_high)}"
    elif range_roll < 0.35:
        range_text = f"{_format_number(ref_low)} to {_format_number(ref_high)}"

    template = rng.choice(TEMPLATES)
    prefix = rng.choice(NOISE_PREFIXES)
    suffix = rng.choice(NOISE_SUFFIXES)

    core = template.format(
        biomarker=biomarker_text,
        value=value_text,
        unit=unit_text,
        range=range_text,
    )
    text = prefix + core + suffix

    bio_start = text.find(biomarker_text)
    val_start = text.find(value_text, bio_start + len(biomarker_text))
    unit_start = text.find(unit_text, val_start + len(value_text))
    rr_start = text.find(range_text, unit_start + len(unit_text))

    spans = [
        {"start": bio_start, "end": bio_start + len(biomarker_text), "label": "BIOMARKER"},
        {"start": val_start, "end": val_start + len(value_text), "label": "VALUE"},
        {"start": unit_start, "end": unit_start + len(unit_text), "label": "UNIT"},
        {"start": rr_start, "end": rr_start + len(range_text), "label": "REFERENCE_RANGE"},
    ]

    tokens = _tokenize_with_offsets(text)
    tags = _spans_to_bio_tags(tokens, spans)

    return {
        "id": int(example_id),
        "tokens": [t["text"] for t in tokens],
        "ner_tags": tags,
        "text": text,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def generate_dataset(
    *,
    observations_csv: Path,
    ranges_csv: Path,
    output_dir: Path,
    max_examples: int,
    seed: int,
) -> dict:
    obs_df = pd.read_csv(observations_csv, low_memory=False)
    obs_df = obs_df.dropna(subset=["biomarker", "value"])
    obs_df["value"] = pd.to_numeric(obs_df["value"], errors="coerce")
    obs_df = obs_df.dropna(subset=["value"]).reset_index(drop=True)

    if obs_df.empty:
        raise ValueError("No usable rows found in biomarker observations.")

    if max_examples > 0 and len(obs_df) > max_examples:
        obs_df = obs_df.sample(n=max_examples, random_state=seed).reset_index(drop=True)

    ranges_df = pd.read_csv(ranges_csv)
    range_lookup = _build_range_lookup(ranges_df)
    fallback_ranges = _build_fallback_ranges(obs_df)

    examples: list[dict] = []
    used_fallback = 0
    rng = random.Random(seed)

    for idx, row in obs_df.iterrows():
        biomarker = str(row.get("biomarker", "")).strip()
        if not biomarker:
            continue

        unit = str(row.get("unit", "")).strip()
        value = float(row["value"])
        key = (_normalize_text(biomarker), unit)
        key_any_unit = (_normalize_text(biomarker), "")

        if key in range_lookup:
            ref_low, ref_high = range_lookup[key]
        elif key_any_unit in range_lookup:
            ref_low, ref_high = range_lookup[key_any_unit]
        elif _normalize_text(biomarker) in fallback_ranges:
            ref_low, ref_high = fallback_ranges[_normalize_text(biomarker)]
            used_fallback += 1
        else:
            ref_low, ref_high = value * 0.9, value * 1.1
            if ref_low == ref_high:
                ref_low, ref_high = value - 1.0, value + 1.0
            used_fallback += 1

        if ref_low >= ref_high:
            ref_low, ref_high = min(ref_low, ref_high), max(ref_low, ref_high)
            if ref_low == ref_high:
                ref_low, ref_high = ref_low - 1.0, ref_high + 1.0

        examples.append(
            _make_example(
                example_id=len(examples),
                biomarker=biomarker,
                value=value,
                unit=unit,
                ref_low=float(ref_low),
                ref_high=float(ref_high),
                rng=rng,
            )
        )

    if not examples:
        raise ValueError("Could not generate synthetic examples from provided data.")

    random.Random(seed).shuffle(examples)
    total = len(examples)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    train = examples[:train_end]
    val = examples[train_end:val_end]
    test = examples[val_end:]

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "train.jsonl", train)
    _write_jsonl(output_dir / "val.jsonl", val)
    _write_jsonl(output_dir / "test.jsonl", test)

    meta = {
        "source_observations": str(observations_csv),
        "source_ranges": str(ranges_csv),
        "total_examples": int(total),
        "train_examples": int(len(train)),
        "val_examples": int(len(val)),
        "test_examples": int(len(test)),
        "used_fallback_ranges": int(used_fallback),
        "labels": ["BIOMARKER", "VALUE", "UNIT", "REFERENCE_RANGE"],
        "seed": int(seed),
    }
    (output_dir / "synthetic_ner_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate starter NER JSONL splits from tabular biomarker data")
    parser.add_argument(
        "--observations-csv",
        default="data/interim/normalized_records/biomarker_observations.csv",
        help="Path to normalized biomarker observations CSV",
    )
    parser.add_argument(
        "--ranges-csv",
        default="knowledge_base/reference_ranges/biomarker_reference_ranges.csv",
        help="Path to biomarker reference ranges CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where train.jsonl/val.jsonl/test.jsonl will be written",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=12000,
        help="Maximum number of examples to generate (0 means use all rows)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    meta = generate_dataset(
        observations_csv=Path(args.observations_csv),
        ranges_csv=Path(args.ranges_csv),
        output_dir=Path(args.output_dir),
        max_examples=int(args.max_examples),
        seed=int(args.seed),
    )

    print("[OK] Synthetic starter NER dataset generated")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
