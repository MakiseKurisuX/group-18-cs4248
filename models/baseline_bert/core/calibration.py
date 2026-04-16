"""Threshold calibration helpers shared by training and evaluation flows."""

import numpy as np

from .artifacts import save_threshold
from .config import LABEL_COLUMN, ORIGINAL_VAL_PATH, get_mode_model_dir
from .dataset import load_input_dataframe
from .inference import predict_batches
from .metrics import compute_classification_metrics


def find_optimal_threshold(
    probs_sarcastic: np.ndarray,
    true_labels: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict:
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    best = {"threshold": 0.5, "f1": -1.0}
    for threshold in thresholds:
        predictions = (probs_sarcastic >= threshold).astype(int)
        metrics = compute_classification_metrics(true_labels, predictions)
        if metrics["f1"] > best["f1"]:
            best = {"threshold": round(float(threshold), 4), **metrics}
    return best


def calibrate_threshold_for_frame(
    val_frame,
    model_reference,
    max_length: int,
    use_domain_context: bool = False,
    progress_description: str = "Threshold calibration",
) -> dict:
    _, probabilities, _ = predict_batches(
        val_frame,
        model_reference=model_reference,
        max_length=max_length,
        progress_description=progress_description,
        use_domain_context=use_domain_context,
    )
    true_labels = val_frame[LABEL_COLUMN].astype(int).to_numpy()
    return find_optimal_threshold(probabilities[:, 1], true_labels)


def calibrate_threshold_for_mode(
    mode: str,
    max_length: int,
    use_domain_context: bool = False,
    val_frame=None,
) -> dict:
    resolved_val_frame = val_frame
    if resolved_val_frame is None:
        resolved_val_frame = load_input_dataframe(str(ORIGINAL_VAL_PATH), dataset_name="original_val")
    result = calibrate_threshold_for_frame(
        val_frame=resolved_val_frame,
        model_reference=get_mode_model_dir(mode),
        max_length=max_length,
        use_domain_context=use_domain_context,
    )
    save_threshold(mode, result)
    return result
