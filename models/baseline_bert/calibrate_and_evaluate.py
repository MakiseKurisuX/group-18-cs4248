"""Standalone threshold calibration and re-evaluation helper."""

import argparse
import json

from core.calibration import calibrate_threshold_for_mode
from core.cli_args import add_domain_context_args
from core.config import CALIBRATABLE_BERT_MODES, MAX_LENGTH, ensure_directories, get_mode_model_dir
from core.modes import get_evaluation_targets
from core.training import log_step
from evaluate import evaluate_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate threshold on the validation set and re-evaluate a saved BERT model."
    )
    parser.add_argument("--mode", choices=CALIBRATABLE_BERT_MODES, required=True)
    parser.add_argument("--output-suffix", default="calibrated")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    add_domain_context_args(parser)
    parser.add_argument("--skip-master-eval", action="store_true", help="Skip master_copy_dedup_v2 evaluation.")
    parser.add_argument("--skip-prediction-exports", action="store_true", help="Skip writing per-example prediction CSVs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    output_mode = f"{args.mode}_{args.output_suffix}"
    log_step(f"Source model: {args.mode}")
    log_step(f"Output namespace: {output_mode}")

    calibration_result = calibrate_threshold_for_mode(
        mode=args.mode,
        max_length=args.max_length,
        use_domain_context=args.use_domain_context,
    )
    threshold = calibration_result["threshold"]

    evaluation_metrics = []
    for target in get_evaluation_targets(include_master_copy=not args.skip_master_eval):
        metrics = evaluate_split(
            split_or_path=target["path"],
            model_path=get_mode_model_dir(args.mode),
            mode=output_mode,
            dataset_name=target["dataset_name"],
            max_length=args.max_length,
            use_domain_context=args.use_domain_context,
            save_predictions_output=not args.skip_prediction_exports,
            threshold=threshold,
        )
        evaluation_metrics.append(metrics)

    summary = {
        "source_mode": args.mode,
        "output_mode": output_mode,
        "threshold": threshold,
        "calibration_result": calibration_result,
        "evaluations": evaluation_metrics,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
