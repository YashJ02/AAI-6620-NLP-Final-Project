"""Clean and standardize biomarker_observations.csv for NER training.

This script:
1. Removes non-medical fields (age, bmi, mortality, diabetes, etc.)
2. Merges duplicate biomarker names to canonical forms
3. Fixes/infers missing units from biomarker names
4. Normalizes Platelet_Count from raw counts to 10^3/uL
5. Drops rows with missing units or clinically implausible values
6. Validates all values against reference ranges
7. Produces a detailed cleaning report
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Non-medical fields to remove entirely
EXCLUDE_FIELDS = {
    "age", "bmi", "mortality", "readmission_30d", "diabetes",
    "hypertension", "visit", "sex", "diagnosis", "patient_id",
}

# Canonical name mapping: raw_name -> (canonical_name, unit)
# Merges duplicates and fixes names
CANONICAL_MAP = {
    # --- From health_markers_dataset ---
    "Blood_glucose": ("Blood_Glucose", "mg/dL"),
    "HbA1C": ("HbA1c", "%"),
    "Systolic_BP": ("Systolic_BP", "mmHg"),
    "Diastolic_BP": ("Diastolic_BP", "mmHg"),
    "LDL": ("LDL_Cholesterol", "mg/dL"),
    "HDL": ("HDL_Cholesterol", "mg/dL"),
    "Triglycerides": ("Triglycerides", "mg/dL"),
    "Haemoglobin": ("Hemoglobin", "g/dL"),
    "MCV": ("MCV", "fL"),
    # --- From blood_count_dataset ---
    "Hemoglobin": ("Hemoglobin", "g/dL"),
    "Platelet_Count": ("Platelet_Count", "10^3/uL"),
    "White_Blood_Cells": ("WBC", "10^3/uL"),
    "Red_Blood_Cells": ("RBC", "10^6/uL"),
    "MCH": ("MCH", "pg"),
    "MCHC": ("MCHC", "g/dL"),
    # --- From synthetic_clinical_dataset ---
    "glucose": ("Blood_Glucose", "mg/dL"),
    "cholesterol": ("Total_Cholesterol", "mg/dL"),
    "creatinine": ("Creatinine", "mg/dL"),
    "systolic_bp": ("Systolic_BP", "mmHg"),
    "diastolic_bp": ("Diastolic_BP", "mmHg"),
    # --- From AI4FoodDB biomarkers ---
    "hgb_g_dl": ("Hemoglobin", "g/dL"),
    "hematocrit_perc": ("Hematocrit", "%"),
    "mcv_fl": ("MCV", "fL"),
    "mch_pg": ("MCH", "pg"),
    "mchc_g_dl": ("MCHC", "g/dL"),
    "rdw_perc": ("RDW", "%"),
    "mpv_fl": ("MPV", "fL"),
    "plats_10e3_ul": ("Platelet_Count", "10^3/uL"),
    "leukocytes_10e3_ul": ("WBC", "10^3/uL"),
    "erythrocytes_10e6_ul": ("RBC", "10^6/uL"),
    "lympho_10e3_ul": ("Lymphocytes", "10^3/uL"),
    "mono_10e3_ul": ("Monocytes", "10^3/uL"),
    "seg_10e3_ul": ("Neutrophils", "10^3/uL"),
    "eos_10e3_ul": ("Eosinophils", "10^3/uL"),
    "baso_10e3_ul": ("Basophils", "10^3/uL"),
    "glu_mg_dl": ("Blood_Glucose", "mg/dL"),
    "chol_mg_dl": ("Total_Cholesterol", "mg/dL"),
    "tri_mg_dl": ("Triglycerides", "mg/dL"),
    "hdl_mg_dl": ("HDL_Cholesterol", "mg/dL"),
    "ldl_mg_dl": ("LDL_Cholesterol", "mg/dL"),
    "hba1c_perc": ("HbA1c", "%"),
    "insulin_uui_ml": ("Insulin", "mU/L"),
    "crp_mg_dl": ("CRP", "mg/dL"),
    "alb_g_dl": ("Albumin", "g/dL"),
    "adiponectin_ug_ml": ("Adiponectin", "ug/mL"),
    "homocysteine_umol_l": ("Homocysteine", "umol/L"),
    "prealbumin_mg_dl": ("Prealbumin", "mg/dL"),
    "tnf_a_ui_ml": ("TNF_Alpha", "UI/mL"),
    "iga_mg_dl": ("IgA", "mg/dL"),
    "igg_mg_dl": ("IgG", "mg/dL"),
    "igm_mg_dl": ("IgM", "mg/dL"),
    "ige_ui_ml": ("IgE", "UI/mL"),
    "homa": ("HOMA_IR", "index"),
    "hba1ifcc_mmol_mol": ("HbA1c_IFCC", "mmol/mol"),
    # --- From vital_signs ---
    "systolic_blood_pressure_mmhg": ("Systolic_BP", "mmHg"),
    "dyastolic_blood_pressure_mmhg": ("Diastolic_BP", "mmHg"),
    "heart_rate_bpm": ("Heart_Rate", "bpm"),
    # --- Percentage fields that need fixing ---
    "lympho_perc": ("Lymphocyte_Pct", "%"),
    "mono_perc": ("Monocyte_Pct", "%"),
    "seg_perc": ("Neutrophil_Pct", "%"),
    "eos_perc": ("Eosinophil_Pct", "%"),
    "baso_perc": ("Basophil_Pct", "%"),
}

# Clinical plausibility ranges (min, max) - values outside are dropped
CLINICAL_RANGES = {
    "Blood_Glucose": (20, 600),
    "HbA1c": (2, 20),
    "Systolic_BP": (50, 260),
    "Diastolic_BP": (20, 160),
    "LDL_Cholesterol": (10, 400),
    "HDL_Cholesterol": (5, 120),
    "Triglycerides": (20, 1000),
    "Total_Cholesterol": (50, 500),
    "Hemoglobin": (3, 25),
    "MCV": (50, 130),
    "MCH": (15, 45),
    "MCHC": (25, 40),
    "Platelet_Count": (10, 900),  # in 10^3/uL
    "WBC": (1, 50),               # in 10^3/uL
    "RBC": (1, 10),               # in 10^6/uL
    "Creatinine": (0.1, 20),
    "Hematocrit": (15, 65),       # percentage
    "RDW": (8, 30),
    "MPV": (4, 20),
    "Insulin": (0.5, 300),
    "Heart_Rate": (30, 200),
}


def clean_observations(input_csv: Path, output_csv: Path, report_json: Path) -> dict:
    """Clean biomarker observations and produce a report."""
    df = pd.read_csv(input_csv, low_memory=False)
    initial_rows = len(df)
    report = {"initial_rows": initial_rows, "steps": []}

    # --- Step 1: Remove non-medical fields ---
    mask_exclude = df["biomarker"].str.lower().str.strip().str.replace("_", " ").isin(
        {f.replace("_", " ") for f in EXCLUDE_FIELDS}
    )
    removed_nonmed = int(mask_exclude.sum())
    df = df[~mask_exclude].copy()
    report["steps"].append({
        "step": "Remove non-medical fields",
        "removed": removed_nonmed,
        "remaining": len(df),
        "fields_removed": sorted(EXCLUDE_FIELDS),
    })

    # --- Step 2: Drop rows with null values ---
    null_before = int(df["value"].isna().sum())
    df = df.dropna(subset=["biomarker", "value"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    null_after = int(df["value"].isna().sum())
    df = df.dropna(subset=["value"])
    report["steps"].append({
        "step": "Drop null values",
        "removed": null_before + null_after,
        "remaining": len(df),
    })

    # --- Step 3: Map to canonical names and fix units ---
    unmapped = set()
    canonical_names = []
    canonical_units = []
    for _, row in df.iterrows():
        raw_name = str(row["biomarker"]).strip()
        if raw_name in CANONICAL_MAP:
            cname, cunit = CANONICAL_MAP[raw_name]
            canonical_names.append(cname)
            canonical_units.append(cunit)
        else:
            unmapped.add(raw_name)
            canonical_names.append(raw_name)
            canonical_units.append(str(row.get("unit", "")) if pd.notna(row.get("unit")) else "")

    df["biomarker"] = canonical_names
    df["unit"] = canonical_units
    report["steps"].append({
        "step": "Canonicalize biomarker names and units",
        "mapped": len(df) - len(unmapped),
        "unmapped_biomarkers": sorted(unmapped),
    })

    # --- Step 4: Normalize Platelet_Count from raw counts to 10^3/uL ---
    platelet_mask = (df["biomarker"] == "Platelet_Count") & (df["value"] > 1000)
    platelet_converted = int(platelet_mask.sum())
    df.loc[platelet_mask, "value"] = df.loc[platelet_mask, "value"] / 1000.0
    report["steps"].append({
        "step": "Normalize Platelet_Count to 10^3/uL",
        "converted": platelet_converted,
    })

    # --- Step 5: Normalize WBC from raw counts to 10^3/uL ---
    wbc_mask = (df["biomarker"] == "WBC") & (df["value"] > 100)
    wbc_converted = int(wbc_mask.sum())
    df.loc[wbc_mask, "value"] = df.loc[wbc_mask, "value"] / 1000.0
    report["steps"].append({
        "step": "Normalize WBC to 10^3/uL",
        "converted": wbc_converted,
    })

    # --- Step 6: Normalize percentage fields stored as decimals ---
    pct_biomarkers = {"Hematocrit", "Lymphocyte_Pct", "Monocyte_Pct",
                      "Neutrophil_Pct", "Eosinophil_Pct", "Basophil_Pct", "RDW"}
    for bio in pct_biomarkers:
        mask = (df["biomarker"] == bio) & (df["value"] < 1) & (df["unit"] == "%")
        converted = int(mask.sum())
        if converted > 0:
            df.loc[mask, "value"] = df.loc[mask, "value"] * 100.0
            report["steps"].append({
                "step": f"Convert {bio} from decimal to percentage",
                "converted": converted,
            })

    # --- Step 7: Drop zero/negative values ---
    nonpositive = int((df["value"] <= 0).sum())
    df = df[df["value"] > 0]
    report["steps"].append({
        "step": "Remove zero/negative values",
        "removed": nonpositive,
        "remaining": len(df),
    })

    # --- Step 8: Drop rows with empty/missing units ---
    bad_units = df["unit"].isna() | df["unit"].astype(str).str.strip().isin(["", "nan", "NaN"])
    removed_no_unit = int(bad_units.sum())
    df = df[~bad_units]
    report["steps"].append({
        "step": "Remove rows with missing units",
        "removed": removed_no_unit,
        "remaining": len(df),
    })

    # --- Step 9: Clinical plausibility check ---
    implausible_total = 0
    for bio, (lo, hi) in CLINICAL_RANGES.items():
        mask = (df["biomarker"] == bio) & ((df["value"] < lo) | (df["value"] > hi))
        count = int(mask.sum())
        if count > 0:
            implausible_total += count
            df = df[~mask]
    report["steps"].append({
        "step": "Remove clinically implausible values",
        "removed": implausible_total,
        "remaining": len(df),
    })

    # --- Step 10: Remove duplicates ---
    dup_count = int(df.duplicated(subset=["sample_id", "biomarker", "value"]).sum())
    df = df.drop_duplicates(subset=["sample_id", "biomarker", "value"])
    report["steps"].append({
        "step": "Remove duplicate rows",
        "removed": dup_count,
        "remaining": len(df),
    })

    # --- Final summary ---
    df = df.sort_values(["biomarker", "sample_id"]).reset_index(drop=True)

    report["final_rows"] = len(df)
    report["final_biomarkers"] = int(df["biomarker"].nunique())
    report["final_biomarker_counts"] = df["biomarker"].value_counts().to_dict()
    report["removed_total"] = initial_rows - len(df)
    report["removal_pct"] = round((initial_rows - len(df)) / initial_rows * 100, 1)

    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    return report


def main():
    input_csv = Path("data/interim/normalized_records/biomarker_observations.csv")
    output_csv = Path("data/interim/normalized_records/biomarker_observations_clean.csv")
    report_json = Path("data/interim/normalized_records/cleaning_report.json")

    report = clean_observations(input_csv, output_csv, report_json)

    print("[OK] Cleaning complete")
    print(f"  Initial rows:  {report['initial_rows']:>10,}")
    print(f"  Final rows:    {report['final_rows']:>10,}")
    print(f"  Removed:       {report['removed_total']:>10,} ({report['removal_pct']}%)")
    print(f"  Biomarkers:    {report['final_biomarkers']:>10}")
    print(f"\nBiomarker counts:")
    for bio, count in sorted(report["final_biomarker_counts"].items(), key=lambda x: -x[1]):
        print(f"    {bio:30s} {count:>8,}")

    print(f"\n[OK] Saved to {output_csv}")
    print(f"[OK] Report at {report_json}")


if __name__ == "__main__":
    main()
