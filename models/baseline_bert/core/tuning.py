"""Shared Optuna orchestration for BERT experiment modes.

This module owns the reusable tuning flow used by both the original baseline
and the improved variants. The mode-specific pieces are injected by passing a
hyperparameter-builder function and a training function.
"""

from transformers import AutoTokenizer

from .config import BERT_MODEL_NAME, SEED, TUNING_DIR, get_mode_tuning_config, get_mode_tuning_path
from .training import _json_safe, log_step


def _get_optuna():
    """Import Optuna lazily so non-tuning workflows do not require it at import time."""
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for hyperparameter tuning. "
            "Install it with `pip install optuna` or `conda env create -f models/baseline_bert/environment.yml`."
        ) from exc
    return optuna


def build_trial_hyperparameters(trial, tuning_grid: dict) -> dict:
    """Map a baseline Optuna trial to a concrete hyperparameter dictionary."""
    learning_rate_options = sorted({float(value) for value in tuning_grid["learning_rate"]})
    if len(learning_rate_options) == 1:
        learning_rate = learning_rate_options[0]
    else:
        learning_rate = trial.suggest_float(
            "learning_rate",
            min(learning_rate_options),
            max(learning_rate_options),
            log=True,
        )

    wr_min, wr_max = sorted(float(value) for value in tuning_grid["warmup_ratio"])
    wd_min, wd_max = sorted(float(value) for value in tuning_grid["weight_decay"])
    ldf_min, ldf_max = sorted(float(value) for value in tuning_grid["llrd_decay_factor"])
    ls_min, ls_max = sorted(float(value) for value in tuning_grid["label_smoothing"])

    return {
        "learning_rate": float(learning_rate),
        "batch_size": int(trial.suggest_categorical("batch_size", [int(value) for value in tuning_grid["batch_size"]])),
        "num_epochs": int(trial.suggest_categorical("num_epochs", [int(value) for value in tuning_grid["num_epochs"]])),
        "max_length": int(trial.suggest_categorical("max_length", [int(value) for value in tuning_grid["max_length"]])),
        "warmup_ratio": float(trial.suggest_float("warmup_ratio", wr_min, wr_max)),
        "weight_decay": float(trial.suggest_float("weight_decay", wd_min, wd_max, log=True)),
        "llrd_decay_factor": float(trial.suggest_float("llrd_decay_factor", ldf_min, ldf_max)),
        "grad_accum": int(trial.suggest_categorical("grad_accum", [int(value) for value in tuning_grid["grad_accum"]])),
        "label_smoothing": float(trial.suggest_float("label_smoothing", ls_min, ls_max)),
    }


def _format_trial_start(trial_number: int, hyperparameters: dict) -> str:
    """Format a trial-start log message with all sampled hyperparameters."""
    fields = ", ".join(f"{key}={value}" for key, value in hyperparameters.items())
    return f"Optuna trial {trial_number} started with {fields}"


def run_optuna_search(
    mode: str,
    train_frame,
    val_frame,
    use_domain_context: bool,
    build_hyperparameters,
    train_model,
    model_name: str = BERT_MODEL_NAME,
) -> dict:
    """Run an Optuna study and return a serialized tuning summary.

    The objective is always validation F1. Tokenizer loading is shared across
    trials to avoid repeated work, while model initialization stays per-trial.
    """
    optuna = _get_optuna()
    tuning_config = get_mode_tuning_config(mode)
    tuning_grid = tuning_config["grid"]
    n_trials = tuning_config["n_trials"]
    log_step(f"Starting Optuna tuning for mode '{mode}' with {n_trials} trial(s)")

    # The tokenizer is constant across trials, so load it once and reuse it.
    shared_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(1, min(2, n_trials)), n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def objective(trial):
        hyperparameters = build_hyperparameters(trial, tuning_grid)
        log_step(_format_trial_start(trial.number, hyperparameters))
        trial.set_user_attr("hyperparameters", _json_safe(hyperparameters))
        trial_dir = TUNING_DIR / mode / f"trial_{trial.number}"
        outcome = train_model(
            train_frame=train_frame,
            val_frame=val_frame,
            hyperparameters=hyperparameters,
            output_dir=trial_dir,
            save_artifacts=False,
            model_name=model_name,
            trial=trial,
            tokenizer=shared_tokenizer,
            use_domain_context=use_domain_context,
        )
        validation_metrics = outcome["validation_metrics"]
        log_step(
            f"Optuna trial {trial.number} finished with "
            f"eval_accuracy={validation_metrics.get('eval_accuracy')}, eval_f1={validation_metrics.get('eval_f1')}"
        )
        trial.set_user_attr("validation_metrics", _json_safe(validation_metrics))
        return float(validation_metrics.get("eval_f1", 0.0))

    study.optimize(objective, n_trials=n_trials)

    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        raise RuntimeError("Optuna did not complete any tuning trials successfully.")

    best_trial = study.best_trial
    serialized_trials = []
    for trial in study.trials:
        serialized_trials.append(
            {
                "number": int(trial.number),
                "state": str(trial.state),
                "value": None if trial.value is None else float(trial.value),
                "params": _json_safe(trial.params),
                "hyperparameters": _json_safe(trial.user_attrs.get("hyperparameters", {})),
                "validation_metrics": _json_safe(trial.user_attrs.get("validation_metrics", {})),
            }
        )

    tuning_summary = {
        "mode": mode,
        "tuning_backend": "optuna",
        "selection_metric": "eval_f1",
        "n_trials": int(n_trials),
        "best_trial": {
            "number": int(best_trial.number),
            "value": float(best_trial.value),
            "params": _json_safe(best_trial.params),
            "hyperparameters": _json_safe(best_trial.user_attrs.get("hyperparameters", {})),
            "validation_metrics": _json_safe(best_trial.user_attrs.get("validation_metrics", {})),
        },
        "trials": serialized_trials,
    }
    log_step(f"Optuna selected trial {best_trial.number} for mode '{mode}' with eval_f1={best_trial.value}")
    save_json(tuning_summary, get_mode_tuning_path(mode))
    return tuning_summary
