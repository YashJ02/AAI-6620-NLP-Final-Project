"""Generate realistic NER JSONL splits from normalized tabular biomarker observations.

Produces training data that mirrors real OCR-extracted blood report formats:
- Tabular layouts (test name | value | unit | reference range)
- Clinical sentence patterns
- OCR noise and formatting variations
- Multi-row contexts with multiple biomarkers
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


TOKEN_RE = re.compile(r"\S+")

# ---------------------------------------------------------------------------
# Medical biomarkers only (exclude demographic / outcome fields)
# ---------------------------------------------------------------------------
EXCLUDE_BIOMARKERS: set[str] = set()  # Already cleaned in clean_observations.py

# Canonical display names for biomarkers (how they appear in real reports)
BIOMARKER_ALIASES: dict[str, list[str]] = {
    # Uses canonical names from clean_observations.py
    "Blood_Glucose": ["Blood Glucose", "BLOOD SUGAR RANDOM", "Blood Sugar", "Glucose", "GLUCOSE", "RBS", "FBS", "Fasting Blood Sugar", "Random Blood Sugar", "Glu", "BLOOD SUGAR", "Blood Sugar Random", "Fasting Glucose"],
    "Hemoglobin": ["Hemoglobin", "HEMOGLOBIN", "Haemoglobin", "Hb", "HB", "HGB", "Hgb", "HAEMOGLOBIN"],
    "HDL_Cholesterol": ["HDL", "HDL Cholesterol", "HDL-C", "HDL CHOLESTEROL", "High Density Lipoprotein", "HDL-Cholesterol"],
    "LDL_Cholesterol": ["LDL", "LDL Cholesterol", "LDL-C", "LDL CHOLESTEROL", "Low Density Lipoprotein", "LDL-Cholesterol"],
    "Total_Cholesterol": ["Total Cholesterol", "TOTAL CHOLESTEROL", "Cholesterol", "CHOLESTEROL", "Serum Cholesterol", "T.Cholesterol", "TC"],
    "Triglycerides": ["Triglycerides", "TRIGLYCERIDES", "TG", "Triglyceride", "TRIGLYCERIDE", "Serum Triglycerides"],
    "HbA1c": ["HbA1c", "HbA1C", "GLYCOSYLATED HEMOGLOBIN", "Glycated Hemoglobin", "HBA1C", "A1C", "Glycated Hb"],
    "Creatinine": ["Creatinine", "CREATININE", "Serum Creatinine", "S.Creatinine", "S. Creatinine", "SERUM CREATININE", "Creat"],
    "Systolic_BP": ["Systolic BP", "Systolic Blood Pressure", "SYSTOLIC BP", "SBP", "Sys BP"],
    "Diastolic_BP": ["Diastolic BP", "Diastolic Blood Pressure", "DIASTOLIC BP", "DBP", "Dia BP"],
    "RBC": ["RBC", "Red Blood Cells", "RED BLOOD CELLS", "RBC COUNT", "Total RBC", "Erythrocyte Count", "RBC Count", "Erythrocytes"],
    "WBC": ["WBC", "White Blood Cells", "WHITE BLOOD CELLS", "WBC COUNT", "Total WBC", "Leukocyte Count", "Total W.B.C", "WBC Count", "Leucocytes"],
    "Platelet_Count": ["Platelet Count", "PLATELET COUNT", "Platelets", "PLATELETS", "PLT", "Plt Count", "Thrombocytes"],
    "MCV": ["MCV", "Mean Corpuscular Volume", "MEAN CORPUSCULAR VOLUME"],
    "MCH": ["MCH", "Mean Corpuscular Hemoglobin", "MEAN CORPUSCULAR HEMOGLOBIN", "Mean Corp. Hb"],
    "MCHC": ["MCHC", "Mean Corpuscular Hb Conc", "MEAN CORPUSCULAR HB CONC", "Mean Corp. Hb Conc."],
    "Hematocrit": ["Hematocrit", "HEMATOCRIT", "HCT", "Hct", "PCV", "Packed Cell Volume"],
    "RDW": ["RDW", "Red Cell Distribution Width", "RDW-CV"],
    "MPV": ["MPV", "Mean Platelet Volume", "MEAN PLATELET VOLUME"],
    "Insulin": ["Insulin", "INSULIN", "Fasting Insulin", "Serum Insulin"],
    "Albumin": ["Albumin", "ALBUMIN", "Serum Albumin", "S. Albumin"],
    "CRP": ["CRP", "C-Reactive Protein", "C Reactive Protein", "hs-CRP"],
    "Heart_Rate": ["Heart Rate", "HEART RATE", "Pulse", "HR", "Pulse Rate"],
    "Lymphocytes": ["Lymphocytes", "LYMPHOCYTES", "Lymph", "Lymphocyte Count"],
    "Monocytes": ["Monocytes", "MONOCYTES", "Mono", "Monocyte Count"],
    "Neutrophils": ["Neutrophils", "NEUTROPHILS", "Neut", "Neutrophil Count", "Polymorphs"],
    "Eosinophils": ["Eosinophils", "EOSINOPHILS", "Eos", "Eosinophil Count"],
    "Basophils": ["Basophils", "BASOPHILS", "Baso", "Basophil Count"],
    "IgA": ["IgA", "Immunoglobulin A", "IMMUNOGLOBULIN A"],
    "IgG": ["IgG", "Immunoglobulin G", "IMMUNOGLOBULIN G"],
    "IgM": ["IgM", "Immunoglobulin M", "IMMUNOGLOBULIN M"],
    "IgE": ["IgE", "Immunoglobulin E", "IMMUNOGLOBULIN E", "Total IgE"],
    "Homocysteine": ["Homocysteine", "HOMOCYSTEINE", "Serum Homocysteine"],
    "Prealbumin": ["Prealbumin", "PREALBUMIN", "Transthyretin"],
}

# Units with common OCR variations
UNIT_VARIANTS: dict[str, list[str]] = {
    "mg/dL": ["mg/dL", "mg/dl", "mg /dL", "mgldl", "mg/ dL", "mg/dl"],
    "g/dL": ["g/dL", "g/dl", "gldl", "g /dL", "gm/dl", "gm/dL"],
    "mmHg": ["mmHg", "mm Hg", "mmhg", "mm/Hg"],
    "%": ["%", "percent"],
    "fL": ["fL", "fl", "FL"],
    "pg": ["pg", "Pg", "PG"],
    "10^3/uL": ["10^3/uL", "x10^3/uL", "10^3/ul", "thou/uL", "K/uL", "cells/cumm"],
    "10^6/uL": ["10^6/uL", "x10^6/uL", "10^6/ul", "mill/cumm", "M/uL", "milllcumm"],
    "IU/L": ["IU/L", "IU/l", "IUIL", "U/L", "lUI/L"],
    "ug/L": ["ug/L", "ng/mL", "mcg/L"],
    "mU/L": ["mU/L", "uIU/mL", "mIU/L"],
    "ng/dL": ["ng/dL", "ng/dl"],
    "KU/L": ["KU/L", "kU/L"],
}

# ---------------------------------------------------------------------------
# Templates based on REAL extracted blood reports
# ---------------------------------------------------------------------------

# Table-row templates (most common in real reports ~60%)
TABLE_TEMPLATES = [
    # Standard lab table rows
    "{biomarker} {value} {unit} {range}",
    "{biomarker} {value} {unit} {range}",
    "{biomarker} {value} {unit} {range}",
    # With dots/alignment (common in printed reports)
    "{biomarker} .... {value} {unit} {range}",
    "{biomarker} ... {value} {unit} {range}",
    # With separators
    "{biomarker} | {value} | {unit} | {range}",
    "{biomarker} : {value} {unit} ( {range} )",
    "{biomarker} : {value} {unit} {range}",
    # Parenthesized ranges
    "{biomarker} {value} {unit} ( {range} )",
    "{biomarker} {value} {unit} ({range})",
    "{biomarker} {value} {unit} [ {range} ]",
    # Labeled columns (as seen in OCR output)
    "Test: {biomarker} Result: {value} {unit} Ref: {range}",
    "Investigation {biomarker} {value} {unit} Reference Range {range}",
    "Parameter {biomarker} {value} {unit} Biological Reference {range}",
    "{biomarker} Observed Value {value} {unit} Reference Range {range}",
    "{biomarker} Result {value} {unit} Normal Range {range}",
]

# Clinical sentence templates (~25%)
SENTENCE_TEMPLATES = [
    "The patient's {biomarker} was {value} {unit} , reference range {range} .",
    "Lab results show {biomarker} at {value} {unit} ( normal {range} ) .",
    "{biomarker} level measured at {value} {unit} against normal range {range} .",
    "Blood work revealed {biomarker} at {value} {unit} , reference {range} .",
    "Complete blood count shows {biomarker} {value} {unit} ( {range} ) .",
    "On investigation , {biomarker} came back {value} {unit} within the reference of {range} .",
    "Report indicates {biomarker} is {value} {unit} , normal being {range} .",
    "Biochemistry panel : {biomarker} {value} {unit} Normal {range} .",
    "During admission , {biomarker} was recorded as {value} {unit} ( ref {range} ) .",
    "{biomarker} reading was {value} {unit} . Normal reference range is {range} .",
]

# Hospital header / multi-row context templates (~15%)
CONTEXT_TEMPLATES = [
    "LABORATORY TEST RESULT {biomarker} {value} {unit} {range}",
    "HEMATOLOGY REPORT {biomarker} {value} {unit} {range}",
    "BIOCHEMISTRY {biomarker} {value} {unit} {range}",
    "LIPID PROFILE {biomarker} {value} {unit} {range}",
    "COMPLETE BLOOD COUNT {biomarker} {value} {unit} {range}",
    "RFT WITH EGFR {biomarker} {value} {unit} {range}",
    "BLOOD GAS ANALYSIS {biomarker} {value} {unit} {range}",
    "LIVER FUNCTION TEST {biomarker} {value} {unit} {range}",
    "KIDNEY FUNCTION TEST {biomarker} {value} {unit} {range}",
    "THYROID PROFILE {biomarker} {value} {unit} {range}",
]

# Hospital headers/noise seen in real reports
HOSPITAL_HEADERS = [
    "",
    "ADVANTA SUPER SPECIALITY HOSPITAL",
    "Sugam Hospitals Symbol of Health",
    "PARTH PATHOLOGY LABORATORY",
    "KIMS ICON Hospital",
    "Apollo Hospitals",
    "Good Life Diagnostic Centre",
    "Sterling Accuris Diagnostic",
    "Dr. Lal PathLabs",
    "SRL Diagnostics",
    "Metropolis Healthcare",
    "Thyrocare Technologies",
]

NOISE_PREFIXES = [
    "", "", "", "", "",  # empty most of the time
    "Page 1 of 3 ",
    "Page 2 of 5 ",
    "Date: 25-04-2025 ",
    "Date: 27/04/2025 ",
    "Patient ID: {pid} ",
    "UHID{pid} ",
    "Srl No {pid} ",
    "Dr: {doctor} ",
    "Ref. {doctor} ",
    "Consultant {doctor} ",
    "Specimen: Blood ",
    "Sample Type: Serum ",
    "Collected: 25-04-2025 08:30 AM ",
]

NOISE_SUFFIXES = [
    "", "", "", "", "",
    " Method: Automated",
    " Method: UV Kinetic",
    " Method: CALCULATED",
    " Method: Photometry",
    " Status: Final",
    " Specimen: Blood",
    " Fasting: Yes",
    " End of Report",
]

DOCTOR_NAMES = [
    "Dr. Sharma", "Dr. Patel", "Dr. Verma", "Dr. Singh",
    "Dr. Kumar", "Dr. Rajput", "Dr. Gupta", "Dr. Reddy",
    "Dr. Jain", "Dr. Rao", "Dr. Das", "Dr. Nair",
]

# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------


def _normalize_text(value: str) -> str:
    return str(value).strip().lower().replace("_", " ")


def _format_number(value: float, rng: random.Random) -> str:
    """Format number with realistic variations."""
    if abs(value) >= 1000:
        # Large numbers: sometimes no decimal
        if rng.random() < 0.5:
            return str(int(round(value)))
        return f"{value:.0f}"
    if abs(value) >= 100:
        if rng.random() < 0.3:
            return f"{value:.0f}"
        return f"{value:.1f}"
    if abs(value) >= 10:
        roll = rng.random()
        if roll < 0.3:
            return f"{value:.0f}"
        elif roll < 0.7:
            return f"{value:.1f}"
        return f"{value:.2f}"
    # Small numbers: more decimal places
    roll = rng.random()
    if roll < 0.4:
        return f"{value:.1f}"
    elif roll < 0.8:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _format_range(ref_low: float, ref_high: float, rng: random.Random) -> str:
    """Format reference range with realistic variations."""
    low_s = _format_number(ref_low, rng)
    high_s = _format_number(ref_high, rng)

    roll = rng.random()
    if roll < 0.55:
        return f"{low_s}-{high_s}"       # "12.0-15.5"
    elif roll < 0.70:
        return f"{low_s} - {high_s}"     # "12.0 - 15.5"
    elif roll < 0.80:
        return f"{low_s} to {high_s}"    # "12.0 to 15.5"
    elif roll < 0.88:
        return f"({low_s}-{high_s})"     # "(12.0-15.5)"
    elif roll < 0.94:
        return f"{low_s}-{high_s}"       # plain again
    else:
        return f"[{low_s}-{high_s}]"     # "[12.0-15.5]"


def _get_biomarker_display(biomarker_key: str, rng: random.Random) -> str:
    """Get a realistic display name for a biomarker."""
    key = _normalize_text(biomarker_key)
    if key in BIOMARKER_ALIASES:
        return rng.choice(BIOMARKER_ALIASES[key])
    # Fallback: clean up the raw name
    name = biomarker_key.replace("_", " ").strip()
    roll = rng.random()
    if roll < 0.3:
        return name.upper()
    elif roll < 0.5:
        return name.title()
    return name


def _get_unit_display(unit: str, rng: random.Random) -> str:
    """Get a realistic unit display with optional OCR noise."""
    unit = str(unit).strip()
    if unit in UNIT_VARIANTS:
        return rng.choice(UNIT_VARIANTS[unit])
    # Check reverse lookup
    for canonical, variants in UNIT_VARIANTS.items():
        if unit in variants or unit.lower() == canonical.lower():
            return rng.choice(variants)
    if not unit or unit == "nan" or unit == "NaN":
        return rng.choice(["", "units", "-"])
    return unit


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


# ---------------------------------------------------------------------------
# Example builders
# ---------------------------------------------------------------------------


def _make_example(
    *,
    example_id: int,
    biomarker: str,
    value: float,
    unit: str,
    ref_low: float,
    ref_high: float,
    rng: random.Random,
) -> dict | None:
    biomarker_text = _get_biomarker_display(biomarker, rng)
    value_text = _format_number(value, rng)
    unit_text = _get_unit_display(unit, rng)
    range_text = _format_range(ref_low, ref_high, rng)

    # Skip if unit is empty (produces ambiguous spans)
    if not unit_text or unit_text == "-":
        unit_text = rng.choice(["units", "%", "mg/dL", "g/dL", "mmHg"])

    # Pick template category
    roll = rng.random()
    if roll < 0.55:
        template = rng.choice(TABLE_TEMPLATES)
    elif roll < 0.80:
        template = rng.choice(SENTENCE_TEMPLATES)
    else:
        template = rng.choice(CONTEXT_TEMPLATES)

    core = template.format(
        biomarker=biomarker_text,
        value=value_text,
        unit=unit_text,
        range=range_text,
    )

    # Add realistic noise
    prefix = rng.choice(NOISE_PREFIXES)
    if "{pid}" in prefix:
        prefix = prefix.replace("{pid}", str(rng.randint(10000, 99999)))
    if "{doctor}" in prefix:
        prefix = prefix.replace("{doctor}", rng.choice(DOCTOR_NAMES))

    suffix = rng.choice(NOISE_SUFFIXES)

    # Optionally prepend hospital header
    header = ""
    if rng.random() < 0.15:
        header = rng.choice(HOSPITAL_HEADERS)
        if header:
            header = header + " "

    text = header + prefix + core + suffix

    # Find span offsets
    bio_start = text.find(biomarker_text)
    if bio_start == -1:
        return None
    val_start = text.find(value_text, bio_start + len(biomarker_text))
    if val_start == -1:
        return None
    unit_start = text.find(unit_text, val_start + len(value_text))
    if unit_start == -1:
        return None
    rr_start = text.find(range_text, unit_start + len(unit_text))
    if rr_start == -1:
        return None

    spans = [
        {"start": bio_start, "end": bio_start + len(biomarker_text), "label": "BIOMARKER"},
        {"start": val_start, "end": val_start + len(value_text), "label": "VALUE"},
        {"start": unit_start, "end": unit_start + len(unit_text), "label": "UNIT"},
        {"start": rr_start, "end": rr_start + len(range_text), "label": "REFERENCE_RANGE"},
    ]

    tokens = _tokenize_with_offsets(text)
    tags = _spans_to_bio_tags(tokens, spans)

    # Validate: ensure all entity types got at least one B- tag
    tag_set = set(tags)
    required = {"B-BIOMARKER", "B-VALUE", "B-UNIT", "B-REFERENCE_RANGE"}
    if not required.issubset(tag_set):
        return None

    return {
        "id": int(example_id),
        "tokens": [t["text"] for t in tokens],
        "ner_tags": tags,
        "text": text,
    }


def _make_multi_row_example(
    *,
    example_id: int,
    rows: list[dict],
    rng: random.Random,
) -> list[dict]:
    """Generate a multi-biomarker context (2-4 biomarkers in one block)."""
    examples = []
    # Pick a section header
    headers = [
        "HEMATOLOGY REPORT", "BIOCHEMISTRY", "LIPID PROFILE",
        "COMPLETE BLOOD COUNT", "LIVER FUNCTION TEST", "KIDNEY FUNCTION TEST",
        "Test Name Observed Value Unit Reference Range",
        "Investigation Result Unit Bio. Ref. Interval",
    ]
    header = rng.choice(headers)

    for i, row in enumerate(rows):
        biomarker_text = _get_biomarker_display(row["biomarker"], rng)
        value_text = _format_number(row["value"], rng)
        unit_text = _get_unit_display(row["unit"], rng)
        if not unit_text or unit_text == "-":
            unit_text = rng.choice(["units", "%", "mg/dL"])
        range_text = _format_range(row["ref_low"], row["ref_high"], rng)

        # Build multi-row text: header + preceding rows (as O-tagged context) + target row
        context_lines = [header]
        for j in range(i):
            prev = rows[j]
            prev_bio = _get_biomarker_display(prev["biomarker"], rng)
            prev_val = _format_number(prev["value"], rng)
            prev_unit = _get_unit_display(prev["unit"], rng) or "units"
            prev_range = _format_range(prev["ref_low"], prev["ref_high"], rng)
            context_lines.append(f"{prev_bio} {prev_val} {prev_unit} {prev_range}")

        # Target line
        target_line = f"{biomarker_text} {value_text} {unit_text} {range_text}"
        context_lines.append(target_line)

        # Add trailing rows as context
        for j in range(i + 1, len(rows)):
            nxt = rows[j]
            nxt_bio = _get_biomarker_display(nxt["biomarker"], rng)
            nxt_val = _format_number(nxt["value"], rng)
            nxt_unit = _get_unit_display(nxt["unit"], rng) or "units"
            nxt_range = _format_range(nxt["ref_low"], nxt["ref_high"], rng)
            context_lines.append(f"{nxt_bio} {nxt_val} {nxt_unit} {nxt_range}")

        text = " ".join(context_lines)

        # Find target spans in full text
        target_start = text.find(target_line)
        if target_start == -1:
            continue

        bio_start = target_start
        val_start = target_start + target_line.find(value_text)
        unit_start = target_start + target_line.find(unit_text, len(biomarker_text))
        rr_start = target_start + target_line.find(range_text, target_line.find(unit_text))

        if any(s < 0 for s in [bio_start, val_start, unit_start, rr_start]):
            continue

        spans = [
            {"start": bio_start + target_line.find(biomarker_text),
             "end": bio_start + target_line.find(biomarker_text) + len(biomarker_text),
             "label": "BIOMARKER"},
            {"start": val_start, "end": val_start + len(value_text), "label": "VALUE"},
            {"start": unit_start, "end": unit_start + len(unit_text), "label": "UNIT"},
            {"start": rr_start, "end": rr_start + len(range_text), "label": "REFERENCE_RANGE"},
        ]

        tokens = _tokenize_with_offsets(text)
        tags = _spans_to_bio_tags(tokens, spans)

        tag_set = set(tags)
        required = {"B-BIOMARKER", "B-VALUE", "B-UNIT", "B-REFERENCE_RANGE"}
        if not required.issubset(tag_set):
            continue

        examples.append({
            "id": example_id + i,
            "tokens": [t["text"] for t in tokens],
            "ner_tags": tags,
            "text": text,
        })

    return examples


# ---------------------------------------------------------------------------
# Data cleaning & generation
# ---------------------------------------------------------------------------


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


def _clean_observations(obs_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to medical biomarkers only, fix units."""
    # Drop non-biomarker fields
    obs_df = obs_df[~obs_df["biomarker"].str.lower().str.strip().str.replace("_", " ").isin(EXCLUDE_BIOMARKERS)]

    # Drop rows with missing values
    obs_df = obs_df.dropna(subset=["biomarker", "value"])
    obs_df["value"] = pd.to_numeric(obs_df["value"], errors="coerce")
    obs_df = obs_df.dropna(subset=["value"])

    # Drop zero/negative values that don't make sense for most biomarkers
    obs_df = obs_df[obs_df["value"] > 0]

    # Fix missing units using biomarker name heuristics
    unit_map = {
        "blood_glucose": "mg/dL", "glucose": "mg/dL",
        "haemoglobin": "g/dL", "hemoglobin": "g/dL",
        "hdl": "mg/dL", "ldl": "mg/dL",
        "cholesterol": "mg/dL", "triglycerides": "mg/dL",
        "creatinine": "mg/dL",
        "hba1c": "%",
        "systolic_bp": "mmHg", "diastolic_bp": "mmHg",
        "systolic_blood_pressure_mmhg": "mmHg",
        "dyastolic_blood_pressure_mmhg": "mmHg",
        "red_blood_cells": "10^6/uL", "eritrosit": "10^6/uL",
        "white_blood_cells": "10^3/uL",
        "platelet_count": "10^3/uL",
        "mcv": "fL", "mch": "pg", "mchc": "g/dL",
        "ferritin": "ug/L",
    }
    for idx, row in obs_df.iterrows():
        if pd.isna(row["unit"]) or str(row["unit"]).strip() in ("", "nan", "NaN"):
            key = _normalize_text(row["biomarker"])
            if key in unit_map:
                obs_df.at[idx, "unit"] = unit_map[key]

    # Drop remaining rows without units
    obs_df = obs_df[obs_df["unit"].notna() & (obs_df["unit"].astype(str).str.strip() != "")
                     & (obs_df["unit"].astype(str).str.strip() != "nan")]

    return obs_df.reset_index(drop=True)


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
    obs_df = _clean_observations(obs_df)

    if obs_df.empty:
        raise ValueError("No usable rows found in biomarker observations.")

    print(f"[INFO] After cleaning: {len(obs_df)} observations, "
          f"{obs_df['biomarker'].nunique()} unique biomarkers")
    print(f"[INFO] Biomarkers: {sorted(obs_df['biomarker'].unique())}")

    ranges_df = pd.read_csv(ranges_csv)
    range_lookup = _build_range_lookup(ranges_df)
    fallback_ranges = _build_fallback_ranges(obs_df)

    rng = random.Random(seed)

    # Sample observations for single-row examples
    if max_examples > 0:
        # Stratified sampling: ensure all biomarkers represented
        sampled_frames = []
        per_biomarker = max(10, max_examples // obs_df["biomarker"].nunique())
        for bio, group in obs_df.groupby("biomarker"):
            n = min(len(group), per_biomarker)
            sampled_frames.append(group.sample(n=n, random_state=seed))
        sampled = pd.concat(sampled_frames).reset_index(drop=True)
        if len(sampled) > max_examples:
            sampled = sampled.sample(n=max_examples, random_state=seed).reset_index(drop=True)
    else:
        sampled = obs_df

    # Generate single-row examples
    examples: list[dict] = []
    used_fallback = 0
    skipped = 0

    for idx, row in sampled.iterrows():
        biomarker = str(row["biomarker"]).strip()
        unit = str(row["unit"]).strip()
        value = float(row["value"])
        key = (_normalize_text(biomarker), unit)
        key_any = (_normalize_text(biomarker), "")

        if key in range_lookup:
            ref_low, ref_high = range_lookup[key]
        elif key_any in range_lookup:
            ref_low, ref_high = range_lookup[key_any]
        elif _normalize_text(biomarker) in fallback_ranges:
            ref_low, ref_high = fallback_ranges[_normalize_text(biomarker)]
            used_fallback += 1
        else:
            ref_low, ref_high = value * 0.8, value * 1.2
            if ref_low == ref_high:
                ref_low, ref_high = value - 1.0, value + 1.0
            used_fallback += 1

        if ref_low >= ref_high:
            ref_low, ref_high = min(ref_low, ref_high), max(ref_low, ref_high)
            if ref_low == ref_high:
                ref_low, ref_high = ref_low - 1.0, ref_high + 1.0

        example = _make_example(
            example_id=len(examples),
            biomarker=biomarker,
            value=value,
            unit=unit,
            ref_low=ref_low,
            ref_high=ref_high,
            rng=rng,
        )
        if example:
            examples.append(example)
        else:
            skipped += 1

    # Generate multi-row examples (~20% of total)
    multi_target = len(examples) // 4
    multi_generated = 0
    biomarker_groups = list(sampled.groupby("biomarker"))

    while multi_generated < multi_target and biomarker_groups:
        # Pick 2-4 random biomarkers for a multi-row block
        n_rows = rng.randint(2, min(4, len(biomarker_groups)))
        selected_groups = rng.sample(biomarker_groups, n_rows)
        block_rows = []
        for bio, group in selected_groups:
            row = group.sample(n=1, random_state=rng.randint(0, 100000)).iloc[0]
            unit = str(row["unit"]).strip()
            value = float(row["value"])
            key = (_normalize_text(bio), unit)
            key_any = (_normalize_text(bio), "")
            if key in range_lookup:
                rl, rh = range_lookup[key]
            elif key_any in range_lookup:
                rl, rh = range_lookup[key_any]
            elif _normalize_text(bio) in fallback_ranges:
                rl, rh = fallback_ranges[_normalize_text(bio)]
            else:
                rl, rh = value * 0.8, value * 1.2
            if rl >= rh:
                rl, rh = min(rl, rh) - 0.1, max(rl, rh) + 0.1
            block_rows.append({
                "biomarker": bio, "value": value, "unit": unit,
                "ref_low": rl, "ref_high": rh,
            })

        multi_examples = _make_multi_row_example(
            example_id=len(examples),
            rows=block_rows,
            rng=rng,
        )
        examples.extend(multi_examples)
        multi_generated += len(multi_examples)

    print(f"[INFO] Generated {len(examples)} examples ({skipped} skipped), "
          f"{multi_generated} multi-row, {used_fallback} used fallback ranges")

    rng.shuffle(examples)
    # Re-assign IDs
    for i, ex in enumerate(examples):
        ex["id"] = i

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

    # Collect label stats
    all_tags = set()
    for ex in examples:
        all_tags.update(ex["ner_tags"])
    labels = sorted([t for t in all_tags if t != "O"])

    meta = {
        "source_observations": str(observations_csv),
        "source_ranges": str(ranges_csv),
        "total_examples": total,
        "train_examples": len(train),
        "val_examples": len(val),
        "test_examples": len(test),
        "used_fallback_ranges": used_fallback,
        "labels": labels,
        "seed": seed,
    }
    (output_dir / "synthetic_ner_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate realistic NER JSONL splits from tabular biomarker data"
    )
    parser.add_argument(
        "--observations-csv",
        default="data/interim/normalized_records/biomarker_observations_clean.csv",
    )
    parser.add_argument(
        "--ranges-csv",
        default="knowledge_base/reference_ranges/biomarker_reference_ranges.csv",
    )
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--max-examples", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    meta = generate_dataset(
        observations_csv=Path(args.observations_csv),
        ranges_csv=Path(args.ranges_csv),
        output_dir=Path(args.output_dir),
        max_examples=int(args.max_examples),
        seed=int(args.seed),
    )

    print("[OK] Synthetic NER dataset generated")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
