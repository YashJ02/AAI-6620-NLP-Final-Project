"""Generate user-facing recommendation summaries."""

from __future__ import annotations

from jinja2 import Template


SUMMARY_TEMPLATE = Template(
    """
Report summary for {{ patient_id }}:

Status overview:
- Low markers: {{ status_summary.low }}
- Normal markers: {{ status_summary.normal }}
- High markers: {{ status_summary.high }}
- Unknown markers: {{ status_summary.unknown }}

{% if flagged_rows %}
Flagged biomarkers:
{% for row in flagged_rows %}
- {{ row.biomarker }}: {{ row.status }} (value={{ row.value }} {{ row.unit }}, range={{ row.reference_range }})
{% endfor %}
{% endif %}

Top recommendation evidence:
{% for item in recommendations %}
- {{ item.text }} (score={{ item.combined_score if item.combined_score is defined else item.score }})
{% endfor %}
""".strip()
)


def render_summary(context: dict) -> str:
    status_summary = context.get("status_summary") or {
        "low": 0,
        "normal": 0,
        "high": 0,
        "unknown": 0,
    }
    interpreted_rows = context.get("interpreted_rows") or []
    recommendations = context.get("recommendations") or []

    flagged_rows = [
        row
        for row in interpreted_rows
        if row.get("status") in {"low", "high"}
    ]

    return SUMMARY_TEMPLATE.render(
        patient_id=context.get("patient_id", "unknown"),
        status_summary=status_summary,
        flagged_rows=flagged_rows,
        recommendations=recommendations[:5],
    )

