"""Build TF-IDF and FAISS retrieval indexes."""

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.recommendation.kb_loader import load_recommendation_docs


def build_tfidf_index(docs: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = [doc["text"] for doc in docs]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(texts)

    np.savez_compressed(
        out_dir / "tfidf_index.npz",
        matrix=matrix.toarray(),
        vocabulary=np.array(vectorizer.get_feature_names_out(), dtype=object),
        doc_ids=np.array([doc["id"] for doc in docs], dtype=object),
    )


def build_semantic_index(docs: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
    except Exception:
        (out_dir / "semantic_index.status").write_text(
            "sentence-transformers/faiss unavailable; skipped", encoding="utf-8"
        )
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([doc["text"] for doc in docs], convert_to_numpy=True, normalize_embeddings=True)
    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(out_dir / "faiss.index"))

    meta = {"doc_ids": [doc["id"] for doc in docs], "model": "all-MiniLM-L6-v2"}
    (out_dir / "faiss_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    docs = load_recommendation_docs()
    build_tfidf_index(docs, Path("knowledge_base/retrieval_index/tfidf"))
    build_semantic_index(docs, Path("knowledge_base/retrieval_index/faiss"))
    print(f"[OK] Built retrieval indexes for {len(docs)} documents")


if __name__ == "__main__":
    main()

