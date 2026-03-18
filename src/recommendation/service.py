"""Recommendation orchestration service."""

from __future__ import annotations

from src.recommendation.ranker import rank_candidates
from src.recommendation.semantic_retriever import retrieve as retrieve_semantic
from src.recommendation.template_generator import render_summary
from src.recommendation.tfidf_retriever import retrieve as retrieve_tfidf


def build_recommendation_query(interpreted_rows: list[dict], ner_entities: list[dict], explicit_query: str = "") -> str:
    if explicit_query and explicit_query.strip():
        return explicit_query.strip()

    abnormal = [row for row in interpreted_rows if row.get("status") in {"low", "high"}]
    abnormal_bits = [f"{row.get('biomarker', '')} {row.get('status', '')}" for row in abnormal]

    biomarker_entities = [
        ent.get("text", "")
        for ent in ner_entities
        if ent.get("label") == "BIOMARKER"
    ]

    query_parts = abnormal_bits + biomarker_entities[:5]
    return " ; ".join(part for part in query_parts if part).strip()


def generate_recommendations(
    *,
    interpreted_rows: list[dict],
    ner_entities: list[dict],
    status_summary: dict,
    patient_id: str,
    query: str = "",
    top_k: int = 5,
) -> dict:
    final_query = build_recommendation_query(
        interpreted_rows=interpreted_rows,
        ner_entities=ner_entities,
        explicit_query=query,
    )

    tfidf_hits = retrieve_tfidf(final_query, top_k=top_k) if final_query else []
    semantic_hits = retrieve_semantic(final_query, top_k=top_k) if final_query else []
    ranked = rank_candidates(tfidf_hits + semantic_hits)[:top_k]

    summary = render_summary(
        {
            "patient_id": patient_id,
            "status_summary": status_summary,
            "interpreted_rows": interpreted_rows,
            "recommendations": ranked,
        }
    )

    return {
        "query": final_query,
        "results": ranked,
        "summary": summary,
    }
