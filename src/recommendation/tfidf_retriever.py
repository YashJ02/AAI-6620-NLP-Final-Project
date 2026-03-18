"""Baseline TF-IDF retriever."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.recommendation.kb_loader import load_recommendation_docs


def retrieve(query: str, top_k: int = 5) -> list:
    """Retrieve top-k recommendation snippets using TF-IDF cosine similarity."""
    docs = load_recommendation_docs()
    if not docs:
        return []

    texts = [doc["text"] for doc in docs]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    doc_matrix = vectorizer.fit_transform(texts)
    query_vec = vectorizer.transform([query])

    scores = cosine_similarity(query_vec, doc_matrix).flatten()
    top_indices = scores.argsort()[::-1][: max(1, top_k)]

    results = []
    for idx in top_indices:
        results.append(
            {
                "id": docs[idx]["id"],
                "text": docs[idx]["text"],
                "source": docs[idx].get("source", "unknown"),
                "score": round(float(scores[idx]), 4),
                "method": "tfidf",
            }
        )

    return results

