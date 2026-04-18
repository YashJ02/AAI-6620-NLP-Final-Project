"""Synthesize missing readiness assets for interpretation and retrieval evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_project_root_on_path


ensure_project_root_on_path()

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.recommendation.kb_loader import load_recommendation_docs


RANGE_COLUMNS = ["biomarker", "unit", "ref_low", "ref_high", "source"]
UNIT_MAP_COLUMNS = ["biomarker", "from_unit", "to_unit", "multiplier", "offset"]


def _read_csv_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)

    df = pd.read_csv(path, low_memory=False)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df[columns].copy()


def _normalize_key(biomarker: str, unit: str) -> tuple[str, str]:
    return str(biomarker).strip().lower(), str(unit).strip().lower()


def synthesize_reference_ranges(
    ranges_csv: Path,
    observations_csv: Path,
    unit_map_csv: Path,
    target_rows: int,
    min_samples: int,
) -> dict:
    existing = _read_csv_or_empty(ranges_csv, RANGE_COLUMNS)
    observations = _read_csv_or_empty(observations_csv, ["biomarker", "value", "unit"])

    observations["biomarker"] = observations["biomarker"].astype(str).str.strip()
    observations["unit"] = observations["unit"].fillna("").astype(str).str.strip()
    observations["value"] = pd.to_numeric(observations["value"], errors="coerce")
    observations = observations.dropna(subset=["biomarker", "value"])
    observations = observations[observations["biomarker"] != ""]

    existing["biomarker"] = existing["biomarker"].astype(str).str.strip()
    existing["unit"] = existing["unit"].fillna("").astype(str).str.strip()
    existing["ref_low"] = pd.to_numeric(existing["ref_low"], errors="coerce")
    existing["ref_high"] = pd.to_numeric(existing["ref_high"], errors="coerce")
    existing = existing.dropna(subset=["biomarker", "ref_low", "ref_high"])

    existing_keys = {
        _normalize_key(row["biomarker"], row["unit"]) for _, row in existing.iterrows()
    }

    grouped = observations.groupby(["biomarker", "unit"], dropna=False)
    candidates: list[dict] = []

    grouped_items = sorted(grouped, key=lambda item: len(item[1]), reverse=True)
    for (biomarker, unit), frame in grouped_items:
        if len(existing) + len(candidates) >= target_rows:
            break

        key = _normalize_key(biomarker, unit)
        if key in existing_keys:
            continue

        values = frame["value"].dropna().to_numpy(dtype=float)
        if len(values) < min_samples:
            continue

        low = float(np.quantile(values, 0.10))
        high = float(np.quantile(values, 0.90))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            continue

        candidates.append(
            {
                "biomarker": str(biomarker).strip(),
                "unit": str(unit).strip(),
                "ref_low": round(low, 4),
                "ref_high": round(high, 4),
                "source": "synthetic_quantile_estimate",
            }
        )
        existing_keys.add(key)

    candidate_df = pd.DataFrame(candidates, columns=RANGE_COLUMNS)
    merged = pd.concat([existing, candidate_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=["biomarker", "unit"], keep="first")
    merged = merged.sort_values(["biomarker", "unit"]).reset_index(drop=True)
    merged.to_csv(ranges_csv, index=False)

    unit_map = _read_csv_or_empty(unit_map_csv, UNIT_MAP_COLUMNS)
    unit_map["biomarker"] = unit_map["biomarker"].fillna("*").astype(str)
    unit_map["from_unit"] = unit_map["from_unit"].fillna("").astype(str).str.strip()
    unit_map["to_unit"] = unit_map["to_unit"].fillna("").astype(str).str.strip()
    unit_map["multiplier"] = pd.to_numeric(unit_map["multiplier"], errors="coerce").fillna(1.0)
    unit_map["offset"] = pd.to_numeric(unit_map["offset"], errors="coerce").fillna(0.0)

    existing_unit_keys = {
        (str(row["from_unit"]).strip().lower(), str(row["to_unit"]).strip().lower())
        for _, row in unit_map.iterrows()
    }
    for unit in sorted({str(x).strip() for x in merged["unit"].dropna().tolist() if str(x).strip()}):
        key = (unit.lower(), unit.lower())
        if key in existing_unit_keys:
            continue
        unit_map = pd.concat(
            [
                unit_map,
                pd.DataFrame(
                    [{"biomarker": "*", "from_unit": unit, "to_unit": unit, "multiplier": 1.0, "offset": 0.0}],
                    columns=UNIT_MAP_COLUMNS,
                ),
            ],
            ignore_index=True,
        )
        existing_unit_keys.add(key)

    unit_map = unit_map.drop_duplicates(subset=["biomarker", "from_unit", "to_unit"], keep="first")
    unit_map = unit_map.sort_values(["from_unit", "to_unit"]).reset_index(drop=True)
    unit_map.to_csv(unit_map_csv, index=False)

    return {
        "existing_rows": int(len(existing)),
        "added_rows": int(len(candidate_df)),
        "final_rows": int(len(merged)),
        "unit_map_rows": int(len(unit_map)),
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_query_pool(range_df: pd.DataFrame) -> list[str]:
    biomarkers = [
        str(x).strip()
        for x in range_df.get("biomarker", pd.Series([], dtype=str)).dropna().tolist()
        if str(x).strip()
    ]
    biomarkers = list(dict.fromkeys(biomarkers))

    nutrient_queries = [
        "high glucose nutrition advice",
        "low hemoglobin iron rich foods",
        "high LDL cholesterol diet advice",
        "high triglycerides reduce sugar advice",
        "blood pressure sodium reduction diet",
        "high inflammation anti inflammatory foods",
        "vitamin deficiency dietary improvement",
        "anemia food recommendations",
        "cholesterol lowering fiber foods",
        "balanced blood sugar meal planning",
    ]

    queries: list[str] = []
    for biomarker in biomarkers:
        clean = biomarker.replace("_", " ")
        queries.append(f"high {clean} nutrition advice")
        queries.append(f"low {clean} nutrition advice")
        queries.append(f"normal range support for {clean}")

    queries.extend(nutrient_queries)
    return list(dict.fromkeys([q for q in queries if q.strip()]))


def synthesize_retrieval_eval(
    retrieval_eval_path: Path,
    ranges_csv: Path,
    target_rows: int,
    top_k: int,
) -> dict:
    existing_rows = _read_jsonl(retrieval_eval_path)
    existing_valid: list[dict] = []
    seen_queries: set[str] = set()

    for row in existing_rows:
        query = str(row.get("query", "")).strip()
        relevant_ids = [str(x).strip() for x in row.get("relevant_ids", []) if str(x).strip()]
        if not query or not relevant_ids:
            continue
        if query in seen_queries:
            continue
        existing_valid.append({"query": query, "relevant_ids": relevant_ids})
        seen_queries.add(query)

    docs = load_recommendation_docs()
    if not docs:
        _write_jsonl(retrieval_eval_path, existing_valid)
        return {
            "existing_rows": int(len(existing_valid)),
            "added_rows": 0,
            "final_rows": int(len(existing_valid)),
            "doc_count": 0,
        }

    texts = [str(doc.get("text", "")) for doc in docs]
    ids = [str(doc.get("id", "")) for doc in docs]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    doc_matrix = vectorizer.fit_transform(texts)

    range_df = _read_csv_or_empty(ranges_csv, RANGE_COLUMNS)
    query_pool = _build_query_pool(range_df)

    synthesized: list[dict] = []
    for query in query_pool:
        if len(existing_valid) + len(synthesized) >= target_rows:
            break
        if query in seen_queries:
            continue

        qvec = vectorizer.transform([query])
        scores = cosine_similarity(qvec, doc_matrix).flatten()
        top_indices = scores.argsort()[::-1][: max(1, top_k)]
        relevant_ids = [ids[idx] for idx in top_indices if ids[idx]]
        if not relevant_ids:
            continue

        synthesized.append({"query": query, "relevant_ids": relevant_ids})
        seen_queries.add(query)

    final_rows = existing_valid + synthesized
    _write_jsonl(retrieval_eval_path, final_rows)

    return {
        "existing_rows": int(len(existing_valid)),
        "added_rows": int(len(synthesized)),
        "final_rows": int(len(final_rows)),
        "doc_count": int(len(docs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize readiness assets for interpretation and evaluation")
    parser.add_argument("--project-root", default=".", help="Project root")
    parser.add_argument("--target-range-rows", type=int, default=80, help="Target number of reference range rows")
    parser.add_argument("--range-min-samples", type=int, default=25, help="Minimum observations per biomarker/unit")
    parser.add_argument("--target-retrieval-eval-rows", type=int, default=120, help="Target retrieval benchmark rows")
    parser.add_argument("--retrieval-top-k", type=int, default=3, help="Number of relevant ids per query")
    parser.add_argument(
        "--report",
        default="artifacts/metrics/readiness_synthesis_report.json",
        help="Path to write synthesis summary report",
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    ranges_csv = root / "knowledge_base" / "reference_ranges" / "biomarker_reference_ranges.csv"
    unit_map_csv = root / "knowledge_base" / "reference_ranges" / "unit_conversion_map.csv"
    observations_csv = root / "data" / "interim" / "normalized_records" / "biomarker_observations.csv"
    retrieval_eval_path = root / "data" / "processed" / "retrieval_eval.jsonl"

    range_stats = synthesize_reference_ranges(
        ranges_csv=ranges_csv,
        observations_csv=observations_csv,
        unit_map_csv=unit_map_csv,
        target_rows=int(args.target_range_rows),
        min_samples=int(args.range_min_samples),
    )
    retrieval_stats = synthesize_retrieval_eval(
        retrieval_eval_path=retrieval_eval_path,
        ranges_csv=ranges_csv,
        target_rows=int(args.target_retrieval_eval_rows),
        top_k=int(args.retrieval_top_k),
    )

    report = {
        "reference_ranges": range_stats,
        "retrieval_eval": retrieval_stats,
    }

    report_path = root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"[OK] Wrote synthesis report: {report_path}")


if __name__ == "__main__":
    main()
