"""Inference, evaluation, and threshold-aware reporting for BERT sarcasm detection."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import set_seed

from core.artifacts import get_threshold_path, load_json, save_json, save_predictions
from core.calibration import calibrate_threshold_for_mode
from core.cli_args import add_domain_context_args
from core.config import (
    CALIBRATABLE_BERT_MODES,
    DATASET_COLUMN,
    DEFAULT_BERT_MODE,
    FILE_SOURCE_COLUMN,
    INDEX_COLUMN,
    LABEL_COLUMN,
    LINK_COLUMN,
    MAX_LENGTH,
    ORIGINAL_TEST_PATH,
    SEED,
    TEXT_COLUMN,
    ensure_directories,
    get_mode_metrics_path,
    get_mode_predictions_path,
)
from core.dataset import load_input_dataframe
from core.inference import predict_batches, resolve_bert_model_reference
from core.metrics import compute_classification_metrics


predict_bert = predict_batches


def resolve_evaluation_threshold(
    mode: str | None,
    threshold: float | None,
    max_length: int,
    use_domain_context: bool = False,
    model_path: str | Path | None = None,
    match_training_validation: bool = False,
) -> float:
    """Resolve the effective evaluation threshold for a mode.

    Explicit CLI or caller-provided thresholds win. Otherwise, calibrated
    modes use their saved threshold when available and calibrate on the
    validation split on first use.
    """
    if threshold is not None:
        return float(threshold)

    if match_training_validation:
        return 0.5

    if mode in CALIBRATABLE_BERT_MODES and model_path is None:
        threshold_path = get_threshold_path(mode)
        if threshold_path.exists():
            try:
                payload = load_json(threshold_path)
                return float(payload.get("threshold", 0.5))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        calibration_result = calibrate_threshold_for_mode(
            mode=mode,
            max_length=max_length,
            use_domain_context=use_domain_context,
        )
        return float(calibration_result["threshold"])

    return 0.5


def build_prediction_report(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    token_lengths: list[int] | None = None,
) -> pd.DataFrame:
    text_series = frame[TEXT_COLUMN].astype(str)
    label_col = frame[LABEL_COLUMN]
    pred_int = predictions.astype(int)

    report_columns = {
        "Index": frame[INDEX_COLUMN].tolist(),
        "Article_Link": frame[LINK_COLUMN].tolist(),
        "Headline": text_series.tolist(),
        "Dataset": frame[DATASET_COLUMN].tolist(),
        "Probability of non sarcastic": np.round(probabilities[:, 0], 6),
        "Probability of sarcastic": np.round(probabilities[:, 1], 6),
        "Confidence": np.round(np.max(probabilities, axis=1), 6),
        "Predicted is sarcastic": pred_int,
        "Actual label": label_col.astype("Int64"),
    }

    if not label_col.isna().any():
        actual_np = label_col.astype(int).to_numpy()
        report_columns["Is correct?"] = (pred_int == actual_np).tolist()
        report_columns["False +ve"] = ((pred_int == 1) & (actual_np == 0)).tolist()
        report_columns["False -ve"] = ((pred_int == 0) & (actual_np == 1)).tolist()
    else:
        actual_values = [None if pd.isna(value) else int(value) for value in label_col]
        pred_list = predictions.tolist()
        report_columns["Is correct?"] = [None if actual is None else bool(pred == actual) for pred, actual in zip(pred_list, actual_values)]
        report_columns["False +ve"] = [None if actual is None else bool(pred == 1 and actual == 0) for pred, actual in zip(pred_list, actual_values)]
        report_columns["False -ve"] = [None if actual is None else bool(pred == 0 and actual == 1) for pred, actual in zip(pred_list, actual_values)]

    if token_lengths is None:
        token_lengths = [len(text.split()) for text in text_series]

    report_columns["Text length"] = text_series.str.len().tolist()
    report_columns["Approximate token length"] = token_lengths
    report_columns["Is exclamation?"] = text_series.str.endswith("!").tolist()
    report_columns["Is question?"] = text_series.str.endswith("?").tolist()
    report_columns["Is full stop?"] = text_series.str.endswith(".").tolist()

    if FILE_SOURCE_COLUMN in frame.columns:
        report_columns["File source"] = frame[FILE_SOURCE_COLUMN].fillna("").tolist()

    return pd.DataFrame(report_columns)


def evaluate_split(
    split_or_path: str,
    model_path: str | Path | None = None,
    mode: str | None = None,
    dataset_name: str | None = None,
    output_path: str | Path | None = None,
    max_length: int = MAX_LENGTH,
    use_domain_context: bool = False,
    save_predictions_output: bool = True,
    threshold: float | None = None,
    match_training_validation: bool = False,
) -> dict:
    ensure_directories()
    set_seed(SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    frame = load_input_dataframe(split_or_path, dataset_name=dataset_name)
    resolved_model_reference = resolve_bert_model_reference(mode=mode, model_path=model_path)
    output_mode = mode or DEFAULT_BERT_MODE
    effective_threshold = resolve_evaluation_threshold(
        output_mode,
        threshold,
        max_length=max_length,
        use_domain_context=use_domain_context,
        model_path=model_path,
        match_training_validation=match_training_validation,
    )
    _, probabilities, token_lengths = predict_batches(
        frame,
        model_reference=resolved_model_reference,
        max_length=max_length,
        progress_description=f"Evaluating {dataset_name or str(frame[DATASET_COLUMN].iloc[0])}",
        use_domain_context=use_domain_context,
    )

    if match_training_validation:
        predictions = np.argmax(probabilities, axis=1).astype(int)
    else:
        predictions = (probabilities[:, 1] >= effective_threshold).astype(int)
    dataset_label = dataset_name or str(frame[DATASET_COLUMN].iloc[0])
    metrics = {
        "model_type": "bert",
        "mode": output_mode,
        "dataset": dataset_label,
        "num_examples": int(len(frame)),
        "threshold": effective_threshold,
        "evaluation_style": "trainer_validation" if match_training_validation else "posthoc",
        "output_path": (
            str(output_path or get_mode_predictions_path(output_mode, dataset_label))
            if save_predictions_output
            else None
        ),
        "metrics_path": str(get_mode_metrics_path(output_mode, dataset_label)),
    }

    if not frame[LABEL_COLUMN].isna().any():
        labels = frame[LABEL_COLUMN].astype(int).to_numpy()
        metrics.update(compute_classification_metrics(labels, predictions))
    else:
        metrics["has_labels"] = False

    if save_predictions_output:
        report_frame = build_prediction_report(frame, predictions, probabilities, token_lengths=token_lengths)
        resolved_output_path = Path(output_path) if output_path else get_mode_predictions_path(output_mode, dataset_label)
        save_predictions(report_frame, resolved_output_path)

    if metrics.get("has_labels", True):
        save_json(metrics, get_mode_metrics_path(output_mode, dataset_label))

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a BERT sarcasm detection model.")
    parser.add_argument("--split", default=str(ORIGINAL_TEST_PATH), help="Split name or explicit path to a CSV/JSON/JSONL file.")
    parser.add_argument("--model-path", default=None, help="Optional override for the saved model path.")
    parser.add_argument("--mode", default=None, help="BERT experiment variant to evaluate.")
    parser.add_argument("--dataset-name", default=None, help="Optional dataset label to stamp into the output CSV.")
    parser.add_argument("--output-path", default=None, help="Optional override for the output CSV path.")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH, help="Tokenizer max length for BERT inference.")
    add_domain_context_args(parser)
    parser.add_argument("--skip-predictions", action="store_true", help="Compute metrics without writing the per-example predictions CSV.")
    parser.add_argument(
        "--match-training-validation",
        action="store_true",
        help=(
            "Use the same decision rule as the Trainer validation metrics "
            "(argmax / 0.5 threshold, no auto-loaded calibrated threshold)."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional classification threshold applied to p(sarcastic). "
            "If omitted, calibrated modes auto-load outputs/models/{mode}/threshold.json and "
            "other modes fall back to 0.5."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_split(
        split_or_path=args.split,
        model_path=args.model_path if args.model_path else None,
        mode=args.mode,
        dataset_name=args.dataset_name,
        output_path=args.output_path,
        max_length=args.max_length,
        use_domain_context=args.use_domain_context,
        save_predictions_output=not args.skip_predictions,
        threshold=args.threshold,
        match_training_validation=args.match_training_validation,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
