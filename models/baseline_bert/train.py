"""Unified training entrypoint for all baseline BERT experiment modes.

This script replaces the older split between ``train_bert.py`` and
``train_bert_improved.py``. It handles:

- zero-shot ``pretrained`` / ``pretrained_large`` summaries
- baseline fine-tuning with or without Optuna
- improved fine-tuning
- augmented/master modes that reuse or retune improved hyperparameters

Post-training dataset evaluation now lives in ``evaluate.py`` and
``run_all_evals.py`` so this file can stay focused on training orchestration
and summary writing.
"""

import argparse
import json

from core.artifacts import load_tuning_summary, save_json
from core.cli_args import add_domain_context_args
from core.config import (
    AUGMENTATION_CANDIDATES_PATH,
    ORIGINAL_WITH_TUNING_MODE,
    get_mode_model_dir,
    get_mode_training_summary_path,
    get_mode_tuning_path,
    ensure_directories,
)
from core.modes import AUGMENTED_WITH_TUNING_MODE, IMPROVED_MODE_SET, MASTER_NO_TUNING_MODE, MODE_SPECS
from core.training import log_step
from core.tuning import run_optuna_search


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the unified training entrypoint."""
    parser = argparse.ArgumentParser(description="Train a BERT experiment mode and write its training summary.")
    parser.add_argument("--mode", choices=sorted(MODE_SPECS.keys()), default=ORIGINAL_WITH_TUNING_MODE)
    add_domain_context_args(parser)
    parser.add_argument("--skip-tuning", action="store_true", help="Skip Optuna tuning for tunable modes.")
    parser.add_argument(
        "--retune",
        action="store_true",
        default=False,
        help="For augmented_with_tuning: run a fresh Optuna search on augmented data instead of reusing improved_with_tuning.",
    )
    return parser.parse_args()


def select_hyperparameters(args: argparse.Namespace, spec, splits: dict) -> tuple[dict, dict | None, bool]:
    """Select the final hyperparameters for a mode.

    Depending on the mode and flags, this may:
    - use defaults directly
    - run a fresh Optuna search
    - reuse a prior tuning summary from another mode
    """
    selected_hyperparameters = spec.default_hyperparameters()
    tuning_summary = None
    used_tuning = False

    if args.mode in {ORIGINAL_WITH_TUNING_MODE} and not args.skip_tuning:
        tuning_summary = run_optuna_search(
            args.mode,
            splits["train"],
            splits["val"],
            use_domain_context=args.use_domain_context,
            build_hyperparameters=spec.trial_hyperparameters_builder,
            train_model=spec.train_model,
            model_name=spec.model_name,
        )
        selected_hyperparameters = tuning_summary["best_trial"]["hyperparameters"]
        used_tuning = True
        return selected_hyperparameters, tuning_summary, used_tuning

    if args.mode in IMPROVED_MODE_SET and args.mode not in {AUGMENTED_WITH_TUNING_MODE, MASTER_NO_TUNING_MODE}:
        if spec.trial_hyperparameters_builder is not None and not args.skip_tuning:
            tuning_summary = run_optuna_search(
                args.mode,
                splits["train"],
                splits["val"],
                use_domain_context=args.use_domain_context,
                build_hyperparameters=spec.trial_hyperparameters_builder,
                train_model=spec.train_model,
                model_name=spec.model_name,
            )
            selected_hyperparameters = tuning_summary["best_trial"]["hyperparameters"]
            used_tuning = True
        elif spec.trial_hyperparameters_builder is not None:
            log_step(f"Skipping Optuna tuning for mode '{args.mode}' because --skip-tuning was provided")
        else:
            log_step(f"Using default hyperparameters for mode '{args.mode}' without Optuna tuning")
        return selected_hyperparameters, tuning_summary, used_tuning

    if args.mode == AUGMENTED_WITH_TUNING_MODE:
        if not args.skip_tuning and args.retune:
            tuning_summary = run_optuna_search(
                args.mode,
                splits["train"],
                splits["val"],
                use_domain_context=args.use_domain_context,
                build_hyperparameters=spec.trial_hyperparameters_builder,
                train_model=spec.train_model,
                model_name=spec.model_name,
            )
            selected_hyperparameters = tuning_summary["best_trial"]["hyperparameters"]
            used_tuning = True
        else:
            prior_tuning = load_tuning_summary(spec.reuse_tuning_from)
            if prior_tuning is not None:
                selected_hyperparameters = dict(prior_tuning["best_trial"]["hyperparameters"])
                log_step(
                    f"Reusing {spec.reuse_tuning_from} hyperparameters "
                    f"(eval_f1={prior_tuning['best_trial']['value']:.4f}). Pass --retune to search on augmented data."
                )
            else:
                log_step(f"{spec.reuse_tuning_from} tuning results not found; using default improved hyperparameters")
        return selected_hyperparameters, tuning_summary, used_tuning

    if args.mode == MASTER_NO_TUNING_MODE:
        prior_tuning = load_tuning_summary(spec.reuse_tuning_from)
        if prior_tuning is not None:
            selected_hyperparameters = dict(prior_tuning["best_trial"]["hyperparameters"])
            selected_hyperparameters["batch_size"] = 16
            selected_hyperparameters["grad_accum"] = 1
            selected_hyperparameters["n_topic_clusters"] = 10
            selected_hyperparameters["num_epochs"] = 3
            log_step(
                f"master_no_tuning: reusing {spec.reuse_tuning_from} hyperparameters "
                f"(eval_f1={prior_tuning['best_trial']['value']:.4f}) with "
                f"batch_size=16, grad_accum=1, n_topic_clusters=10, num_epochs=3."
            )
        else:
            selected_hyperparameters["num_epochs"] = 3
            log_step(
                "master_no_tuning: improved_with_tuning tuning results not found; "
                "using default improved hyperparameters with num_epochs=3."
            )
        return selected_hyperparameters, tuning_summary, used_tuning

    if args.mode == ORIGINAL_WITH_TUNING_MODE and args.skip_tuning:
        log_step("Skipping Optuna tuning for original_with_tuning because --skip-tuning was provided")
    else:
        log_step(f"Using default hyperparameters for mode '{args.mode}' without Optuna tuning")
    return selected_hyperparameters, tuning_summary, used_tuning


def main() -> None:
    """Run the selected experiment mode end to end and emit a JSON summary."""
    args = parse_args()
    ensure_directories()

    spec = MODE_SPECS[args.mode]
    log_step(f"Running BERT experiment mode '{args.mode}'")

    if spec.train_model is None:
        training_summary = {
            "mode": args.mode,
            "trained": False,
            "used_tuning": False,
            "use_domain_context": bool(args.use_domain_context),
            "model_reference": spec.model_name,
            "message": "No training was run for this pretrained mode. Use evaluate.py to generate dataset metrics.",
        }
        save_json(training_summary, get_mode_training_summary_path(args.mode))
        print(json.dumps(training_summary, indent=2))
        return

    splits = spec.load_splits()
    selected_hyperparameters, tuning_summary, used_tuning = select_hyperparameters(args, spec, splits)

    final_model_dir = get_mode_model_dir(args.mode)
    log_step(f"Training final model for mode '{args.mode}'")
    final_training = spec.train_model(
        train_frame=splits["train"],
        val_frame=splits["val"],
        hyperparameters=selected_hyperparameters,
        output_dir=final_model_dir,
        save_artifacts=True,
        model_name=spec.model_name,
        use_domain_context=args.use_domain_context,
    )

    training_summary = {
        "mode": args.mode,
        "trained": True,
        "used_tuning": used_tuning,
        "use_domain_context": bool(args.use_domain_context),
        "selected_hyperparameters": selected_hyperparameters,
        "training_metrics": final_training["train_metrics"],
        "validation_metrics": final_training["validation_metrics"],
        "tuning_summary_path": str(get_mode_tuning_path(args.mode)) if tuning_summary else None,
        "model_output_dir": str(final_model_dir),
        "message": "Training complete. Run evaluate.py or run_all_evals.py for post-training dataset metrics.",
    }

    if spec.improvements:
        training_summary["improvements"] = list(spec.improvements)

    if args.mode == AUGMENTED_WITH_TUNING_MODE:
        training_summary["augmentation_candidates_path"] = str(AUGMENTATION_CANDIDATES_PATH)
        training_summary["augmented_train_size"] = len(splits["train"])
    if args.mode == MASTER_NO_TUNING_MODE:
        training_summary["master_train_size"] = len(splits["train"])

    save_json(training_summary, get_mode_training_summary_path(args.mode))
    print(json.dumps(training_summary, indent=2))


if __name__ == "__main__":
    main()
