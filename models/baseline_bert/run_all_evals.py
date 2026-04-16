"""Evaluate one or more BERT modes across the standard evaluation datasets."""

import argparse
import json

from core.cli_args import add_domain_context_args
from core.config import (
    CALIBRATABLE_BERT_MODES,
    MAX_LENGTH,
    PRETRAINED_MODE_MODEL_NAMES,
    SUPPORTED_BERT_MODES,
    ensure_directories,
    get_mode_model_dir,
)
from core.modes import get_evaluation_targets
from core.training import log_step
from evaluate import evaluate_split, resolve_evaluation_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more baseline_bert modes across the standard datasets."
    )
    parser.add_argument(
        "--mode",
        nargs="+",
        default=["all"],
        help="One or more modes to evaluate, or 'all' to run every supported mode.",
    )
    add_domain_context_args(parser)
    parser.add_argument(
        "--include-master-eval",
        action="store_true",
        help="Also evaluate on master_copy_dedup_v2 (skipped by default).",
    )
    parser.add_argument(
        "--skip-prediction-exports",
        action="store_true",
        help="Skip writing per-example prediction CSVs.",
    )
    return parser.parse_args()


def checkpoint_exists(mode: str) -> bool:
    """Return True if the model checkpoint for this mode is available."""
    if mode in PRETRAINED_MODE_MODEL_NAMES:
        return True
    model_dir = get_mode_model_dir(mode)
    return (model_dir / "config.json").exists()


def resolve_modes(mode_args: list[str]) -> list[str]:
    if mode_args == ["all"]:
        return list(SUPPORTED_BERT_MODES)

    invalid_modes = [mode for mode in mode_args if mode not in SUPPORTED_BERT_MODES]
    if invalid_modes:
        supported = ", ".join(SUPPORTED_BERT_MODES)
        raise ValueError(f"Unsupported mode(s): {invalid_modes}. Supported modes: {supported}")
    return mode_args


def main() -> None:
    args = parse_args()
    ensure_directories()

    modes = resolve_modes(args.mode)
    evaluation_targets = get_evaluation_targets(include_master_copy=args.include_master_eval)

    summary = {
        "modes": modes,
        "datasets": [target["dataset_name"] for target in evaluation_targets],
        "use_domain_context": bool(args.use_domain_context),
        "evaluations": [],
    }

    # Uncalibrated sweep: all modes at threshold=0.5
    for mode in modes:
        if not checkpoint_exists(mode):
            log_step(f"Skipping mode '{mode}' — no checkpoint found")
            continue
        log_step(f"Running uncalibrated evaluation sweep for mode '{mode}'")
        for target in evaluation_targets:
            log_step(
                f"Evaluating mode '{mode}' on dataset '{target['dataset_name']}' "
                f"from {target['path']}"
            )
            metrics = evaluate_split(
                split_or_path=target["path"],
                mode=mode,
                dataset_name=target["dataset_name"],
                use_domain_context=args.use_domain_context,
                save_predictions_output=not args.skip_prediction_exports,
                threshold=0.5,
            )
            summary["evaluations"].append(metrics)

    # Calibrated sweep: calibratable modes with their tuned threshold,
    # saved under the {mode}_calibrated naming convention.
    calibratable_in_sweep = [m for m in modes if m in CALIBRATABLE_BERT_MODES and checkpoint_exists(m)]
    if calibratable_in_sweep:
        log_step("Running calibrated threshold sweep for calibratable modes")
        for mode in calibratable_in_sweep:
            calibrated_threshold = resolve_evaluation_threshold(
                mode=mode,
                threshold=None,
                max_length=MAX_LENGTH,
                use_domain_context=args.use_domain_context,
            )
            log_step(f"Using calibrated threshold {calibrated_threshold:.4f} for mode '{mode}'")
            for target in evaluation_targets:
                log_step(
                    f"Evaluating mode '{mode}_calibrated' on dataset '{target['dataset_name']}'"
                )
                metrics = evaluate_split(
                    split_or_path=target["path"],
                    model_path=get_mode_model_dir(mode),
                    mode=f"{mode}_calibrated",
                    dataset_name=target["dataset_name"],
                    use_domain_context=args.use_domain_context,
                    save_predictions_output=not args.skip_prediction_exports,
                    threshold=calibrated_threshold,
                )
                summary["evaluations"].append(metrics)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
