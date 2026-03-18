from src.recommendation.template_generator import render_summary


def test_render_summary_contains_status_and_recommendations():
    text = render_summary(
        {
            "patient_id": "demo",
            "status_summary": {"low": 1, "normal": 2, "high": 0, "unknown": 0},
            "interpreted_rows": [
                {
                    "biomarker": "Hemoglobin",
                    "status": "low",
                    "value": 11.0,
                    "unit": "g/dL",
                    "reference_range": "13.0-17.0",
                }
            ],
            "recommendations": [{"text": "Eat iron-rich foods.", "combined_score": 0.8}],
        }
    )

    assert "demo" in text
    assert "Low markers" in text
    assert "Eat iron-rich foods." in text
