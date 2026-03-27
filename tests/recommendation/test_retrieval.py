from src.recommendation.service import build_recommendation_query
from src.recommendation.service import generate_recommendations


def test_build_recommendation_query_prefers_explicit_query():
    query = build_recommendation_query(
        interpreted_rows=[{"biomarker": "Hemoglobin", "status": "low"}],
        ner_entities=[{"label": "BIOMARKER", "text": "Ferritin"}],
        explicit_query="diet for low hemoglobin",
    )

    assert query == "diet for low hemoglobin"


def test_generate_recommendations_returns_empty_results_for_empty_query():
    output = generate_recommendations(
        interpreted_rows=[],
        ner_entities=[],
        status_summary={"low": 0, "normal": 0, "high": 0, "unknown": 0},
        patient_id="demo",
        query="",
        top_k=5,
    )

    assert output["query"] == ""
    assert output["results"] == []
    assert "Report summary for demo" in output["summary"]

