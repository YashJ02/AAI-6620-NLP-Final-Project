"""Helpers for running project scripts directly via `python scripts/<name>.py`."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root_on_path() -> Path:
    """Ensure project root is importable so `src.*` imports resolve reliably."""
    project_root = Path(__file__).resolve().parents[1]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root
