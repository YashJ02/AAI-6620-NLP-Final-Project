"""Generate user-facing recommendation summaries."""


def render_summary(context: dict) -> str:
    return f"Summary unavailable for {context.get('patient_id', 'unknown')}"

