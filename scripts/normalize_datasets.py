"""Normalize collected datasets into project-ready assets and report data readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FOOD_UNIT_MAP = {
    "Caloric Value": "kcal",
    "Fat": "g",
    "Saturated Fats": "g",
    "Monounsaturated Fats": "g",
    "Polyunsaturated Fats": "g",
    "Carbohydrates": "g",
    "Sugars": "g",
    "Protein": "g",
    "Dietary Fiber": "g",
    "Cholesterol": "mg",
    "Sodium": "mg",
    "Water": "g",
    "Iron": "mg",
    "Magnesium": "mg",
    "Potassium": "mg",
    "Zinc": "mg",
    "Calcium": "mg",
    "Phosphorus": "mg",
    "Vitamin C": "mg",
}


OBSERVATION_UNIT_HINTS = {
    "Hemoglobin": "g/dL",
    "Platelet_Count": "10^3/uL",
    "White_Blood_Cells": "10^3/uL",
    "Red_Blood_Cells": "10^6/uL",
    "MCV": "fL",
    "MCH": "pg",
    "MCHC": "g/dL",
    "Blood_glucose": "mg/dL",
    "HbA1C": "%",
    "Systolic_BP": "mmHg",
    "Diastolic_BP": "mmHg",
    "LDL": "mg/dL",
    "HDL": "mg/dL",
    "Triglycerides": "mg/dL",
    "Haemoglobin": "g/dL",
    "glucose": "mg/dL",
    "cholesterol": "mg/dL",
    "creatinine": "mg/dL",
    "systolic_bp": "mmHg",
    "diastolic_bp": "mmHg",
    "leukocytes_10e3_ul": "10^3/uL",
    "plats_10e3_ul": "10^3/uL",
    "lympho_10e3_ul": "10^3/uL",
    "mono_10e3_ul": "10^3/uL",
    "seg_10e3_ul": "10^3/uL",
    "eos_10e3_ul": "10^3/uL",
    "baso_10e3_ul": "10^3/uL",
    "erythrocytes_10e6_ul": "10^6/uL",
    "hgb_g_dl": "g/dL",
    "hematocrit_perc": "%",
    "mcv_fl": "fL",
    "mch_pg": "pg",
    "mchc_g_dl": "g/dL",
    "hba1c_perc": "%",
    "glu_mg_dl": "mg/dL",
    "chol_mg_dl": "mg/dL",
    "tri_mg_dl": "mg/dL",
    "hdl_mg_dl": "mg/dL",
    "ldl_mg_dl": "mg/dL",
    "systolic_blood_pressure_mmhg": "mmHg",
    "dyastolic_blood_pressure_mmhg": "mmHg",
    "heart_rate_bpm": "bpm",
}


def _classify_dataset_file(rel_path: str, ext: str) -> str:
    p = rel_path.lower()
    if "final food dataset/food-data-group" in p and ext == ".csv":
        return "nutrition_food_composition"
    if "final food dataset/metadata" in p and ext == ".csv":
        return "nutrition_metadata"
    if "ai4fooddb-master" in p and ext == ".csv":
        return "ai4food_tabular"
    if "lbmaske" in p and ext in {".png", ".jpg", ".jpeg"}:
        return "report_page_images"
    if p.endswith("lab_test_results_public.csv"):
        return "lab_reference_ranges"
    if p.endswith("blood_count_dataset.csv"):
        return "cbc_tabular"
    if p.endswith("health_markers_dataset.csv"):
        return "health_markers_tabular"
    if p.endswith("synthetic_clinical_dataset.csv"):
        return "clinical_outcomes_tabular"
    if p.endswith("healthcare_dataset.csv"):
        return "hospital_admin_tabular"
    if p.endswith("blood.csv"):
        return "non_clinical_tabular"
    if p.endswith("train.json"):
        return "external_ner_json"
    if ext in {".jpg", ".jpeg", ".png"}:
        return "image_reference"
    if ext == ".pdf":
        return "pdf_reference"
    if ext == ".xlsx":
        return "spreadsheet_reference"
    if ext == ".csv":
        return "other_csv"
    if ext == ".md":
        return "documentation"
    return "other"


def build_inventory(datasets_dir: Path, output_csv: Path) -> dict:
    records: list[dict] = []
    for f in sorted(datasets_dir.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(datasets_dir).as_posix()
        ext = f.suffix.lower()
        category = _classify_dataset_file(rel, ext)
        records.append(
            {
                "relative_path": rel,
                "extension": ext,
                "size_bytes": int(f.stat().st_size),
                "category": category,
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(["category", "relative_path"]).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    category_counts = (
        df.groupby("category").size().sort_values(ascending=False).to_dict() if not df.empty else {}
    )
    return {
        "file_count": int(len(df)),
        "category_counts": {k: int(v) for k, v in category_counts.items()},
    }


def _load_existing_usda(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["item", "nutrient", "amount", "unit", "source"])
    df = pd.read_csv(path)
    wanted = ["item", "nutrient", "amount", "unit", "source"]
    for c in wanted:
        if c not in df.columns:
            df[c] = ""
    return df[wanted].copy()


def convert_food_to_usda_like(datasets_dir: Path, usda_path: Path) -> dict:
    food_glob = sorted((datasets_dir / "FINAL FOOD DATASET").glob("FOOD-DATA-GROUP*.csv"))
    rows: list[pd.DataFrame] = []

    for file_path in food_glob:
        df = pd.read_csv(file_path)
        if df.empty:
            continue

        drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        if "food" not in df.columns:
            continue

        nutrient_cols = [c for c in df.columns if c != "food"]
        if not nutrient_cols:
            continue

        long_df = df.melt(id_vars=["food"], value_vars=nutrient_cols, var_name="nutrient", value_name="amount")
        long_df["amount"] = pd.to_numeric(long_df["amount"], errors="coerce")
        long_df = long_df.dropna(subset=["amount"])
        long_df = long_df[long_df["amount"] != 0]

        if long_df.empty:
            continue

        out = pd.DataFrame(
            {
                "item": long_df["food"].astype(str).str.strip(),
                "nutrient": long_df["nutrient"].astype(str).str.strip(),
                "amount": long_df["amount"].astype(float),
                "unit": long_df["nutrient"].map(FOOD_UNIT_MAP).fillna("dataset_unit"),
                "source": file_path.name,
            }
        )
        rows.append(out)

    converted = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["item", "nutrient", "amount", "unit", "source"]
    )

    existing = _load_existing_usda(usda_path)
    existing_count = int(len(existing))

    merged = pd.concat([existing, converted], ignore_index=True)
    if not merged.empty:
        merged["item"] = merged["item"].astype(str).str.strip()
        merged["nutrient"] = merged["nutrient"].astype(str).str.strip()
        merged["unit"] = merged["unit"].astype(str).str.strip()
        merged["source"] = merged["source"].astype(str).str.strip()
        merged["amount"] = pd.to_numeric(merged["amount"], errors="coerce")
        merged = merged.dropna(subset=["item", "nutrient", "amount"])
        merged = merged.drop_duplicates(subset=["item", "nutrient", "amount", "unit", "source"])
        merged = merged.sort_values(["item", "nutrient", "source"]).reset_index(drop=True)

    usda_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(usda_path, index=False)

    return {
        "food_group_files_used": int(len(food_glob)),
        "existing_rows": existing_count,
        "converted_rows": int(len(converted)),
        "final_rows": int(len(merged)),
        "added_rows": int(max(0, len(merged) - existing_count)),
    }


def _load_existing_ranges(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["biomarker", "unit", "ref_low", "ref_high", "source"])
    df = pd.read_csv(path)
    wanted = ["biomarker", "unit", "ref_low", "ref_high", "source"]
    for c in wanted:
        if c not in df.columns:
            df[c] = ""
    return df[wanted].copy()


def convert_reference_ranges(datasets_dir: Path, ranges_csv: Path, unit_map_csv: Path) -> dict:
    source_csv = datasets_dir / "lab_test_results_public.csv"
    if not source_csv.exists():
        return {
            "source_found": False,
            "existing_rows": 0,
            "final_rows": 0,
            "added_rows": 0,
            "unit_map_rows": 0,
        }

    src = pd.read_csv(source_csv)
    for c in ["Min_Reference", "Max_Reference"]:
        if c not in src.columns:
            src[c] = pd.NA
        src[c] = pd.to_numeric(src[c], errors="coerce")

    valid = src.dropna(subset=["Min_Reference", "Max_Reference"]).copy()
    valid = valid[valid["Min_Reference"] < valid["Max_Reference"]]

    converted = pd.DataFrame(
        {
            "biomarker": valid.get("Test_Name", pd.Series(dtype=str)).astype(str).str.strip(),
            "unit": valid.get("Unit", pd.Series(dtype=str)).astype(str).str.strip(),
            "ref_low": valid["Min_Reference"].astype(float),
            "ref_high": valid["Max_Reference"].astype(float),
            "source": "lab_test_results_public.csv",
        }
    )
    converted = converted[converted["biomarker"] != ""]

    existing = _load_existing_ranges(ranges_csv)
    existing_count = int(len(existing))

    merged = pd.concat([existing, converted], ignore_index=True)
    if not merged.empty:
        merged["ref_low"] = pd.to_numeric(merged["ref_low"], errors="coerce")
        merged["ref_high"] = pd.to_numeric(merged["ref_high"], errors="coerce")
        merged = merged.dropna(subset=["biomarker", "ref_low", "ref_high"])
        merged = merged[merged["biomarker"].astype(str).str.strip() != ""]
        merged = merged.drop_duplicates(subset=["biomarker", "unit", "ref_low", "ref_high", "source"])
        merged = merged.sort_values(["biomarker", "unit", "source"]).reset_index(drop=True)

    ranges_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(ranges_csv, index=False)

    # Seed identity conversions so the map is not empty.
    units = sorted({u for u in merged["unit"].astype(str).tolist() if u.strip()})
    identity = pd.DataFrame(
        {
            "biomarker": ["*" for _ in units],
            "from_unit": units,
            "to_unit": units,
            "multiplier": [1.0 for _ in units],
            "offset": [0.0 for _ in units],
        }
    )
    unit_map_csv.parent.mkdir(parents=True, exist_ok=True)
    identity.to_csv(unit_map_csv, index=False)

    return {
        "source_found": True,
        "existing_rows": existing_count,
        "final_rows": int(len(merged)),
        "added_rows": int(max(0, len(merged) - existing_count)),
        "unit_map_rows": int(len(identity)),
    }


def convert_biomarker_observations(datasets_dir: Path, output_csv: Path) -> dict:
    sources = [
        datasets_dir / "blood_count_dataset.csv",
        datasets_dir / "health_markers_dataset.csv",
        datasets_dir / "synthetic_clinical_dataset.csv",
        datasets_dir / "AI4FoodDB-master" / "datasets" / "DS4_Biomarkers" / "biomarkers.csv",
        datasets_dir / "AI4FoodDB-master" / "datasets" / "DS6_VitalSigns" / "vital_signs.csv",
    ]

    frames: list[pd.DataFrame] = []
    files_used = 0

    for source_csv in sources:
        if not source_csv.exists():
            continue
        df = pd.read_csv(source_csv)
        if df.empty:
            continue

        files_used += 1

        id_col = None
        for candidate in ["patient_id", "id"]:
            if candidate in df.columns:
                id_col = candidate
                break
        if id_col is None:
            df = df.reset_index(drop=False).rename(columns={"index": "row_id"})
            id_col = "row_id"

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if id_col in numeric_cols:
            numeric_cols.remove(id_col)
        if not numeric_cols:
            continue

        long_df = df[[id_col] + numeric_cols].melt(
            id_vars=[id_col],
            value_vars=numeric_cols,
            var_name="biomarker",
            value_name="value",
        )
        long_df = long_df.dropna(subset=["value"])
        if long_df.empty:
            continue

        out = pd.DataFrame(
            {
                "sample_id": long_df[id_col].astype(str),
                "biomarker": long_df["biomarker"].astype(str),
                "value": pd.to_numeric(long_df["value"], errors="coerce"),
                "unit": long_df["biomarker"].map(OBSERVATION_UNIT_HINTS).fillna(""),
                "source_dataset": source_csv.name,
            }
        )
        out = out.dropna(subset=["value"])
        frames.append(out)

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["sample_id", "biomarker", "value", "unit", "source_dataset"]
    )
    if not merged.empty:
        merged = merged.drop_duplicates().sort_values(["source_dataset", "sample_id", "biomarker"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    return {
        "files_used": int(files_used),
        "rows": int(len(merged)),
    }


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _count_pdf_files(path: Path) -> int:
    if not path.exists():
        return 0
    return int(sum(1 for _ in path.rglob("*.pdf")))


def _readiness_level(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def build_data_position_report(
    project_root: Path,
    inventory_stats: dict,
    food_stats: dict,
    range_stats: dict,
    observation_stats: dict,
    output_json: Path,
    output_md: Path,
) -> dict:
    data_dir = project_root / "data"

    pdf_digital = _count_pdf_files(data_dir / "raw" / "pdfs_digital")
    pdf_scanned = _count_pdf_files(data_dir / "raw" / "pdfs_scanned")
    train_rows = _count_jsonl_rows(data_dir / "processed" / "train.jsonl")
    val_rows = _count_jsonl_rows(data_dir / "processed" / "val.jsonl")
    test_rows = _count_jsonl_rows(data_dir / "processed" / "test.jsonl")
    retrieval_eval_rows = _count_jsonl_rows(data_dir / "processed" / "retrieval_eval.jsonl")

    extraction_score = min(1.0, (pdf_digital + pdf_scanned) / 50.0)
    ner_score = min(1.0, (train_rows + val_rows + test_rows) / 300.0)
    interpretation_score = min(1.0, range_stats.get("final_rows", 0) / 40.0)
    recommendation_score = min(1.0, food_stats.get("final_rows", 0) / 5000.0)
    evaluation_score = min(1.0, retrieval_eval_rows / 100.0)

    overall_score = (extraction_score + ner_score + interpretation_score + recommendation_score + evaluation_score) / 5.0

    report = {
        "inventory": inventory_stats,
        "conversion": {
            "nutrition_usda_like": food_stats,
            "reference_ranges": range_stats,
            "biomarker_observations": observation_stats,
        },
        "position": {
            "extraction": {
                "pdf_digital": int(pdf_digital),
                "pdf_scanned": int(pdf_scanned),
                "score": round(float(extraction_score), 3),
                "level": _readiness_level(extraction_score),
            },
            "ner": {
                "train_rows": int(train_rows),
                "val_rows": int(val_rows),
                "test_rows": int(test_rows),
                "score": round(float(ner_score), 3),
                "level": _readiness_level(ner_score),
            },
            "interpretation": {
                "reference_range_rows": int(range_stats.get("final_rows", 0)),
                "score": round(float(interpretation_score), 3),
                "level": _readiness_level(interpretation_score),
            },
            "recommendation": {
                "nutrition_rows": int(food_stats.get("final_rows", 0)),
                "score": round(float(recommendation_score), 3),
                "level": _readiness_level(recommendation_score),
            },
            "evaluation": {
                "retrieval_eval_rows": int(retrieval_eval_rows),
                "score": round(float(evaluation_score), 3),
                "level": _readiness_level(evaluation_score),
            },
            "overall": {
                "score": round(float(overall_score), 3),
                "level": _readiness_level(overall_score),
            },
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Data Position Report",
        "",
        f"- Overall readiness: {report['position']['overall']['level']} ({report['position']['overall']['score']})",
        f"- Extraction readiness: {report['position']['extraction']['level']} ({report['position']['extraction']['score']})",
        f"- NER readiness: {report['position']['ner']['level']} ({report['position']['ner']['score']})",
        f"- Interpretation readiness: {report['position']['interpretation']['level']} ({report['position']['interpretation']['score']})",
        f"- Recommendation readiness: {report['position']['recommendation']['level']} ({report['position']['recommendation']['score']})",
        f"- Evaluation readiness: {report['position']['evaluation']['level']} ({report['position']['evaluation']['score']})",
        "",
        "## Conversion Summary",
        f"- Nutrition rows final: {food_stats.get('final_rows', 0)}",
        f"- Reference ranges final: {range_stats.get('final_rows', 0)}",
        f"- Biomarker observation rows: {observation_stats.get('rows', 0)}",
        "",
        "## Immediate Next Gaps",
        "- Add real PDFs into data/raw/pdfs_digital and data/raw/pdfs_scanned.",
        "- Create text-span Label Studio annotations and export train/val/test JSONL.",
        "- Expand retrieval_eval.jsonl with more benchmark queries.",
    ]
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize collected datasets and generate readiness report")
    parser.add_argument("--datasets-dir", default="datasets", help="Path to collected datasets directory")
    parser.add_argument("--project-root", default=".", help="Project root path")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    datasets_dir = (project_root / args.datasets_dir).resolve()

    if not datasets_dir.exists():
        raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")

    inventory_csv = project_root / "data" / "interim" / "normalized_records" / "dataset_inventory.csv"
    usda_csv = project_root / "knowledge_base" / "nutrition" / "usda_foods.csv"
    ranges_csv = project_root / "knowledge_base" / "reference_ranges" / "biomarker_reference_ranges.csv"
    unit_map_csv = project_root / "knowledge_base" / "reference_ranges" / "unit_conversion_map.csv"
    observations_csv = project_root / "data" / "interim" / "normalized_records" / "biomarker_observations.csv"
    report_json = project_root / "artifacts" / "metrics" / "data_position_report.json"
    report_md = project_root / "artifacts" / "metrics" / "data_position_report.md"

    inventory_stats = build_inventory(datasets_dir=datasets_dir, output_csv=inventory_csv)
    food_stats = convert_food_to_usda_like(datasets_dir=datasets_dir, usda_path=usda_csv)
    range_stats = convert_reference_ranges(
        datasets_dir=datasets_dir,
        ranges_csv=ranges_csv,
        unit_map_csv=unit_map_csv,
    )
    observation_stats = convert_biomarker_observations(datasets_dir=datasets_dir, output_csv=observations_csv)

    report = build_data_position_report(
        project_root=project_root,
        inventory_stats=inventory_stats,
        food_stats=food_stats,
        range_stats=range_stats,
        observation_stats=observation_stats,
        output_json=report_json,
        output_md=report_md,
    )

    print("[OK] Dataset normalization complete")
    print(f"[OK] Inventory: {inventory_csv}")
    print(f"[OK] Nutrition KB: {usda_csv} ({food_stats['final_rows']} rows)")
    print(f"[OK] Reference ranges: {ranges_csv} ({range_stats['final_rows']} rows)")
    print(f"[OK] Observations: {observations_csv} ({observation_stats['rows']} rows)")
    print(f"[OK] Position report: {report_md}")
    print(json.dumps(report["position"]["overall"], indent=2))


if __name__ == "__main__":
    main()
