"""Rank and merge retrieval candidates."""

from __future__ import annotations


METHOD_WEIGHT = {
    "semantic": 0.6,
    "tfidf": 0.4,
}


def rank_candidates(candidates: list) -> list:
    """Merge duplicate candidates and return globally ranked results."""
    if not candidates:
        return []

    merged: dict[str, dict] = {}
    for candidate in candidates:
        key = str(candidate.get("id", "")) or str(candidate.get("text", ""))
        raw_score = float(candidate.get("score", 0.0))
        method = str(candidate.get("method", "tfidf"))
        weighted = raw_score * METHOD_WEIGHT.get(method, 0.3)

        if key not in merged:
            merged[key] = {
                **candidate,
                "methods": [method],
                "combined_score": weighted,
            }
        else:
            merged[key]["combined_score"] += weighted
            if method not in merged[key]["methods"]:
                merged[key]["methods"].append(method)

    ranked = sorted(merged.values(), key=lambda x: x["combined_score"], reverse=True)
    for item in ranked:
        item["combined_score"] = round(float(item["combined_score"]), 4)
    return ranked

