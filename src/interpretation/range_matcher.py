"""Match biomarker values against reference ranges."""


def classify_against_range(value: float, ref_low: float, ref_high: float) -> str:
    if value < ref_low:
        return "low"
    if value > ref_high:
        return "high"
    return "normal"

