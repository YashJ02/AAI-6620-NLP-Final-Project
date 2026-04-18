"""Helpers for parsing lab report tables into row-level records."""

import re


VALUE_RE = re.compile(r"(?<![A-Za-z])(-?\d+(?:\.\d+)?)")
RANGE_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:-|to)\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
UNIT_RE = re.compile(r"\b(?:g/dL|mg/dL|mmol/L|mIU/L|IU/L|U/L|10\^\d+/L|%|pg|fL|ng/mL)\b", re.IGNORECASE)

BIOMARKER_ALIASES = {
    "hb": "Hemoglobin",
    "hgb": "Hemoglobin",
    "hemoglobin": "Hemoglobin",
    "rbc": "RBC",
    "wbc": "WBC",
    "plt": "Platelets",
    "platelet": "Platelets",
    "platelets": "Platelets",
    "tsh": "TSH",
    "glucose": "Glucose",
}

NON_BIOMARKER_PREFIXES = (
    "page",
    "registration",
    "from previous",
    "h no",
    "male",
    "female",
    "normal",
    "high",
    "low",
    "borderline",
    "optimal",
    "desirable",
    "very high",
    "up to",
)

ADMINISTRATIVE_TOKENS = {
    "address",
    "age",
    "date",
    "doctor",
    "from",
    "health",
    "id",
    "lab",
    "limited",
    "medplus",
    "mobile",
    "name",
    "no",
    "page",
    "patient",
    "phone",
    "previous",
    "registration",
    "report",
    "sample",
    "services",
    "sex",
    "time",
    "years",
}

EXPLANATORY_TOKENS = {
    "absorption",
    "advised",
    "change",
    "changes",
    "clinical",
    "comment",
    "considered",
    "defined",
    "definition",
    "follow",
    "followup",
    "history",
    "indicates",
    "interpretation",
    "note",
    "notes",
    "recommended",
    "suggests",
}

KNOWN_BIOMARKER_TOKENS = {
    *(alias.lower() for alias in BIOMARKER_ALIASES.keys()),
    *(alias.lower() for alias in BIOMARKER_ALIASES.values()),
    "cholesterol",
    "triglycerides",
    "triglyceride",
    "albumin",
    "calcium",
    "psa",
    "vitamin",
    "vitamin d",
    "hdl",
    "ldl",
    "vldl",
    "hematocrit",
    "mcv",
    "mch",
    "mchc",
    "rdw",
    "neutrophils",
    "lymphocytes",
    "monocytes",
    "eosinophils",
    "basophils",
    "creatinine",
    "urea",
    "bilirubin",
    "ast",
    "alt",
    "alp",
}


def _tokenize_words(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token]


def _normalize_spaces(text: str) -> str:
    return " ".join(text.split())


def _normalize_biomarker_name(name: str) -> str:
    normalized = _normalize_spaces(name)
    normalized = re.sub(r"^[A-Za-z0-9]+\)\s*", "", normalized)
    normalized = normalized.strip(" :=<>-.")
    if not normalized:
        return ""

    simple = re.sub(r"[^A-Za-z0-9 ]", "", normalized).lower().strip()
    return BIOMARKER_ALIASES.get(simple, normalized)


def _compute_confidence(*, biomarker: str, unit: str, reference_range: str, source_line: str) -> float:
    # Simple deterministic score based on expected row signals.
    score = 0.5
    if biomarker and re.search(r"[A-Za-z]", biomarker):
        score += 0.2
    if unit:
        score += 0.15
    if reference_range:
        score += 0.15
    if len(source_line.split()) < 3:
        score -= 0.2
    return max(0.0, min(1.0, round(score, 2)))


def _contains_known_biomarker(text: str) -> bool:
    compact = " ".join(_tokenize_words(text))
    if not compact:
        return False

    compact_tokens = set(compact.split())
    for token in KNOWN_BIOMARKER_TOKENS:
        normalized = " ".join(_tokenize_words(token))
        if not normalized:
            continue
        if " " in normalized:
            if re.search(rf"\b{re.escape(normalized)}\b", compact):
                return True
        elif normalized in compact_tokens:
            return True
    return False


def _contains_administrative_token(text: str) -> bool:
    tokens = set(_tokenize_words(text))
    return any(token in tokens for token in ADMINISTRATIVE_TOKENS)


def _looks_like_non_biomarker(label: str) -> bool:
    cleaned = re.sub(r"[^a-z0-9 ]", " ", label.lower())
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return True

    if any(cleaned.startswith(prefix) for prefix in NON_BIOMARKER_PREFIXES):
        return True

    if _contains_administrative_token(cleaned):
        return True

    tokens = cleaned.split()
    if any(token in EXPLANATORY_TOKENS for token in tokens):
        return True

    alpha_tokens = [token for token in tokens if re.search(r"[a-z]", token)]
    if len(alpha_tokens) > 6:
        return True

    if not _contains_known_biomarker(cleaned) and len(alpha_tokens) > 5:
        return True

    if not _contains_known_biomarker(cleaned) and len(re.findall(r"[a-z]", cleaned)) < 3:
        return True

    return False


def _parse_line(line: str) -> dict | None:
    clean = _normalize_spaces(line)
    if not clean:
        return None

    value_matches = list(VALUE_RE.finditer(clean))
    if not value_matches:
        return None

    range_match = RANGE_RE.search(clean)
    unit_match = UNIT_RE.search(clean)

    # Rows without either a unit or a reference range are usually headers/noise lines.
    if not range_match and not unit_match:
        return None

    value_match = None
    biomarker_candidate = ""
    for candidate_match in value_matches:
        candidate_label = clean[: candidate_match.start()].strip(" :-")
        if not candidate_label:
            continue
        if _looks_like_non_biomarker(candidate_label):
            continue
        value_match = candidate_match
        biomarker_candidate = candidate_label
        break

    if value_match is None:
        return None

    value = value_match.group(1)
    reference_range = ""
    if range_match:
        reference_range = f"{range_match.group(1)}-{range_match.group(2)}"

    biomarker_normalized = _normalize_biomarker_name(biomarker_candidate)

    # Lines that only describe category bands (for example "Borderline High 200-239")
    # should not be treated as measured biomarkers.
    if range_match and not unit_match and not _contains_known_biomarker(biomarker_normalized):
        return None

    parsed = {
        "biomarker": biomarker_candidate,
        "biomarker_normalized": biomarker_normalized,
        "value": value,
        "unit": unit_match.group(0) if unit_match else "",
        "reference_range": reference_range,
        "source_line": clean,
    }

    # Avoid obvious false positives where the biomarker is just punctuation.
    if not re.search(r"[A-Za-z]", parsed["biomarker"]):
        return None

    parsed["confidence"] = _compute_confidence(
        biomarker=parsed["biomarker_normalized"],
        unit=parsed["unit"],
        reference_range=parsed["reference_range"],
        source_line=parsed["source_line"],
    )

    return parsed


def parse_table_rows(raw_table: list) -> list:
    """Parse candidate table rows from block-like text objects or strings."""
    if not raw_table:
        return []

    lines: list[str] = []
    for item in raw_table:
        if isinstance(item, dict):
            text = item.get("text", "")
            if text:
                lines.extend(text.splitlines())
        elif isinstance(item, str):
            lines.extend(item.splitlines())

    parsed_rows: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()
    for line in lines:
        parsed = _parse_line(line)
        if not parsed:
            continue

        key = (
            parsed["biomarker_normalized"].lower(),
            parsed["value"],
            parsed["unit"].lower(),
            parsed["reference_range"],
        )
        if key in seen:
            continue
        seen.add(key)
        parsed_rows.append(parsed)

    return parsed_rows

