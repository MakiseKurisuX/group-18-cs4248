"""Artifact helpers for JSON, CSV, and persisted experiment metadata.

These helpers keep simple file I/O out of the training and evaluation
entrypoints so those modules can focus on orchestration instead of repetitive
path and serialization code.
"""

import json
from pathlib import Path

import pandas as pd

from .config import get_mode_model_dir, get_mode_tuning_path


def save_json(payload: dict, output_path: Path) -> None:
    """Write a dict as indented JSON, creating parent directories if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    """Read a JSON file into a Python dict."""
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv(frame: pd.DataFrame, output_path: Path) -> None:
    """Write a DataFrame to CSV, creating parent directories if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def save_predictions(frame: pd.DataFrame, output_path: Path) -> None:
    """Write a predictions DataFrame to CSV.

    Kept as a semantic alias for callers where the intent is clearer than a
    generic ``save_csv`` call.
    """
    save_csv(frame, output_path)


def get_threshold_path(mode: str) -> Path:
    """Return the persisted threshold path for a trained mode."""
    return get_mode_model_dir(mode) / "threshold.json"


def save_threshold(mode: str, payload: dict) -> Path:
    """Persist a calibrated threshold payload and return its output path."""
    threshold_path = get_threshold_path(mode)
    save_json(payload, threshold_path)
    return threshold_path


def load_tuning_summary(mode: str) -> dict | None:
    """Load an Optuna tuning summary for a mode if it has already been run."""
    tuning_path = get_mode_tuning_path(mode)
    if not tuning_path.exists():
        return None
    return load_json(tuning_path)
