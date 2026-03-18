from src.recommendation.ranker import rank_candidates


def test_rank_candidates_merges_duplicate_ids():
    candidates = [
        {"id": "a", "text": "x", "score": 0.8, "method": "tfidf"},
        {"id": "a", "text": "x", "score": 0.9, "method": "semantic"},
        {"id": "b", "text": "y", "score": 0.7, "method": "tfidf"},
    ]

    ranked = rank_candidates(candidates)

    assert len(ranked) == 2
    assert ranked[0]["id"] == "a"
    assert ranked[0]["combined_score"] >= ranked[1]["combined_score"]
