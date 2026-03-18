"""Semantic retriever using embeddings and FAISS."""

from __future__ import annotations

import numpy as np

from src.recommendation.kb_loader import load_recommendation_docs


def retrieve(query: str, top_k: int = 5) -> list:
    """Retrieve semantically similar snippets using embeddings + FAISS."""
    docs = load_recommendation_docs()
    if not docs:
        return []

    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except Exception:
        # Keep pipeline resilient if heavy deps are unavailable at runtime.
        return []

    model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_texts = [doc["text"] for doc in docs]
    doc_embeddings = model.encode(doc_texts, convert_to_numpy=True, normalize_embeddings=True)
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    dim = int(doc_embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings.astype(np.float32))

    k = min(max(1, top_k), len(docs))
    scores, indices = index.search(query_embedding.astype(np.float32), k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append(
            {
                "id": docs[idx]["id"],
                "text": docs[idx]["text"],
                "source": docs[idx].get("source", "unknown"),
                "score": round(float(score), 4),
                "method": "semantic",
            }
        )

    return results

