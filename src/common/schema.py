"""Shared data schemas."""

from dataclasses import dataclass


@dataclass
class BiomarkerRecord:
    biomarker: str
    value: str
    unit: str
    reference_range: str

