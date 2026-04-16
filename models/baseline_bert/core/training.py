"""Shared training orchestration for baseline and improved BERT modes.

This module contains the reusable pieces that both baseline and improved
training paths depend on:

- HuggingFace Trainer metric adaptation
- Layer-wise learning-rate decay (LLRD) optimizer construction
- checkpoint/no-checkpoint callback wiring
- generic train/evaluate/save orchestration

Mode-specific behavior is injected by passing a different trainer class and
extra initialization kwargs rather than duplicating the whole training loop.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from .config import (
    BATCH_SIZE,
    BERT_MODEL_NAME,
    EARLY_STOPPING_PATIENCE,
    EVAL_BATCH_SIZE,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    LLRD_DECAY_FACTOR,
    LR_SCHEDULER_TYPE,
    MAX_LENGTH,
    NUM_EPOCHS,
    PAD_TO_MULTIPLE_OF,
    SEED,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    ensure_directories,
)
from .dataset import build_dataset_from_frame
from .metrics import compute_classification_metrics


def trainer_metrics(eval_prediction):
    """Adapt Trainer eval output to the project's scalar classification metrics."""
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    metrics = compute_classification_metrics(labels, predictions)
    metrics.pop("confusion_matrix", None)
    return metrics


def log_step(message: str) -> None:
    """Print a namespaced log line and flush immediately."""
    print(f"[baseline_bert] {message}", flush=True)


def _json_safe(value):
    """Recursively convert numpy scalars to JSON-serializable Python values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


class OptunaPruningCallback(TrainerCallback):
    """Report validation F1 to Optuna and prune underperforming trials early."""
    def __init__(self, trial):
        self.trial = trial

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or "eval_f1" not in metrics:
            return control

        import optuna

        step = int(state.epoch or state.global_step or 0)
        self.trial.report(float(metrics["eval_f1"]), step=step)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Trial pruned at step {step} with eval_f1={metrics['eval_f1']:.4f}")
        return control


class BestMetricTrackerCallback(TrainerCallback):
    """Track the best validation metrics when checkpoints are disabled."""
    def __init__(self, metric_name: str = "eval_f1"):
        self.metric_name = metric_name
        self.best_value = None
        self.best_metrics = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.metric_name not in metrics:
            return control

        metric_value = float(metrics[self.metric_name])
        if self.best_value is None or metric_value > self.best_value:
            self.best_value = metric_value
            self.best_metrics = _json_safe(metrics)
        return control


def build_llrd_param_groups(model, base_lr: float, weight_decay: float, decay_factor: float) -> list[dict]:
    """Build AdamW parameter groups with layer-wise learning-rate decay.

    The classifier head keeps the full learning rate while deeper BERT layers
    receive progressively smaller rates. Bias and LayerNorm parameters never
    receive weight decay, following standard BERT fine-tuning practice.
    """
    no_decay = {"bias", "LayerNorm.weight"}
    num_layers = model.config.num_hidden_layers
    assigned: set[str] = set()
    groups: list[dict] = []

    def _add(named_params: list[tuple[str, torch.nn.Parameter]], lr: float) -> None:
        wd_params, no_wd_params = [], []
        for name, param in named_params:
            assigned.add(name)
            if any(term in name for term in no_decay):
                no_wd_params.append(param)
            else:
                wd_params.append(param)
        if wd_params:
            groups.append({"params": wd_params, "lr": lr, "weight_decay": weight_decay})
        if no_wd_params:
            groups.append({"params": no_wd_params, "lr": lr, "weight_decay": 0.0})

    _add([(name, param) for name, param in model.named_parameters() if "classifier" in name], base_lr)
    _add([(name, param) for name, param in model.named_parameters() if "pooler" in name], base_lr * decay_factor)

    for layer_idx in range(num_layers - 1, -1, -1):
        depth = num_layers - layer_idx
        _add(
            [(name, param) for name, param in model.named_parameters() if f"encoder.layer.{layer_idx}." in name],
            base_lr * (decay_factor ** depth),
        )

    _add(
        [(name, param) for name, param in model.named_parameters() if "embeddings" in name],
        base_lr * (decay_factor ** (num_layers + 1)),
    )

    leftover = [(name, param) for name, param in model.named_parameters() if name not in assigned]
    if leftover:
        _add(leftover, base_lr)
    return groups


def get_default_hyperparameters() -> dict:
    """Return the baseline training configuration from ``config.py``."""
    return {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "max_length": MAX_LENGTH,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "llrd_decay_factor": LLRD_DECAY_FACTOR,
        "grad_accum": 1,
        "label_smoothing": LABEL_SMOOTHING,
    }


def build_training_arguments(hyperparameters: dict, output_dir: Path, save_checkpoints: bool) -> TrainingArguments:
    """Construct version-compatible HuggingFace ``TrainingArguments``."""
    training_argument_kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "save_strategy": "epoch" if save_checkpoints else "no",
        "logging_strategy": "epoch",
        "learning_rate": hyperparameters["learning_rate"],
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "per_device_train_batch_size": hyperparameters["batch_size"],
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": hyperparameters.get("grad_accum", 1),
        "num_train_epochs": hyperparameters["num_epochs"],
        "warmup_ratio": hyperparameters["warmup_ratio"],
        "label_smoothing_factor": hyperparameters.get("label_smoothing", LABEL_SMOOTHING),
        "load_best_model_at_end": save_checkpoints,
        "report_to": "none",
        "seed": SEED,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    }
    if save_checkpoints:
        training_argument_kwargs.update(
            {
                "metric_for_best_model": "f1",
                "greater_is_better": True,
                "save_total_limit": 1,
            }
        )

    try:
        return TrainingArguments(eval_strategy="epoch", **training_argument_kwargs)
    except TypeError:
        return TrainingArguments(evaluation_strategy="epoch", **training_argument_kwargs)


def build_callbacks(save_checkpoints: bool, trial=None):
    """Build the callback stack for either final training or Optuna trials."""
    best_metric_tracker = None
    callbacks = []
    if save_checkpoints:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE))
    else:
        best_metric_tracker = BestMetricTrackerCallback(metric_name="eval_f1")
        callbacks.append(best_metric_tracker)
    if trial is not None:
        callbacks.append(OptunaPruningCallback(trial))
    return callbacks, best_metric_tracker


def _instantiate_trainer(trainer_class, tokenizer, trainer_kwargs):
    """Instantiate a Trainer across transformers versions with minor API drift."""
    try:
        return trainer_class(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        return trainer_class(tokenizer=tokenizer, **trainer_kwargs)


def _format_log_fields(fields: dict[str, object]) -> str:
    """Format a flat dict of log fields into a human-readable string."""
    return ", ".join(f"{key}={value}" for key, value in fields.items())


def run_training(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    hyperparameters: dict,
    output_dir: Path,
    save_artifacts: bool,
    model_name: str = BERT_MODEL_NAME,
    trial=None,
    tokenizer=None,
    use_domain_context: bool = False,
    trainer_class=Trainer,
    trainer_init_kwargs: dict | None = None,
    extra_log_fields: dict[str, object] | None = None,
    run_label: str = "training run",
) -> dict:
    """Run the shared BERT fine-tuning loop and return train/validation metrics.

    This is the common engine behind both ``train_baseline_model`` and the
    improved-mode trainer. The caller controls the trainer flavor and any
    extra kwargs, but dataset building, optimizer construction, callback
    handling, and final artifact saving stay identical.
    """
    ensure_directories()
    set_seed(SEED)
    if torch.cuda.is_available():
        # TF32 speeds up matrix multiplies on Ampere+ GPUs with negligible
        # practical impact on fine-tuning quality.
        torch.set_float32_matmul_precision("high")

    log_fields = {
        "lr": hyperparameters["learning_rate"],
        "batch_size": hyperparameters["batch_size"],
        "grad_accum": hyperparameters.get("grad_accum", 1),
        "epochs": hyperparameters["num_epochs"],
        "max_length": hyperparameters["max_length"],
        "warmup_ratio": hyperparameters["warmup_ratio"],
        "weight_decay": hyperparameters["weight_decay"],
        "llrd_decay_factor": hyperparameters.get("llrd_decay_factor", LLRD_DECAY_FACTOR),
        "label_smoothing": hyperparameters.get("label_smoothing", LABEL_SMOOTHING),
        "domain_context": use_domain_context,
    }
    if extra_log_fields:
        log_fields.update(extra_log_fields)
    log_step(f"Starting {run_label} with {_format_log_fields(log_fields)}")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_dataset = build_dataset_from_frame(
        train_frame,
        tokenizer=tokenizer,
        max_length=hyperparameters["max_length"],
        require_labels=True,
        use_domain_context=use_domain_context,
    )
    val_dataset = build_dataset_from_frame(
        val_frame,
        tokenizer=tokenizer,
        max_length=hyperparameters["max_length"],
        require_labels=True,
        use_domain_context=use_domain_context,
    )

    pad_to_multiple_of = PAD_TO_MULTIPLE_OF if torch.cuda.is_available() else None
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple_of)

    # LLRD keeps lower BERT layers conservative while letting the classifier and
    # upper encoder layers adapt more aggressively to the task.
    effective_decay_factor = hyperparameters.get("llrd_decay_factor", LLRD_DECAY_FACTOR)
    llrd_param_groups = build_llrd_param_groups(
        model,
        base_lr=hyperparameters["learning_rate"],
        weight_decay=hyperparameters["weight_decay"],
        decay_factor=effective_decay_factor,
    )
    optimizer = torch.optim.AdamW(llrd_param_groups, betas=(0.9, 0.999), eps=1e-8)
    log_step(f"LLRD optimizer built: {len(llrd_param_groups)} param groups, decay_factor={effective_decay_factor}")

    save_checkpoints = save_artifacts
    training_args = build_training_arguments(hyperparameters, output_dir=output_dir, save_checkpoints=save_checkpoints)
    callbacks, best_metric_tracker = build_callbacks(save_checkpoints=save_checkpoints, trial=trial)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": collator,
        "compute_metrics": trainer_metrics,
        "callbacks": callbacks,
        # Passing ``None`` as the scheduler tells Trainer to build the scheduler
        # around our custom optimizer instead of replacing it.
        "optimizers": (optimizer, None),
    }
    if trainer_init_kwargs:
        trainer_kwargs.update(trainer_init_kwargs)

    trainer = _instantiate_trainer(trainer_class=trainer_class, tokenizer=tokenizer, trainer_kwargs=trainer_kwargs)

    train_output = trainer.train()
    if save_checkpoints:
        eval_metrics = trainer.evaluate()
    elif best_metric_tracker is not None and best_metric_tracker.best_metrics is not None:
        eval_metrics = best_metric_tracker.best_metrics
    else:
        eval_metrics = trainer.evaluate()

    log_step(
        f"Finished {run_label} with validation metrics "
        f"accuracy={eval_metrics.get('eval_accuracy')}, f1={eval_metrics.get('eval_f1')}"
    )

    if save_artifacts:
        log_step(f"Saving trained model to {output_dir}")
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

    return {
        "train_metrics": _json_safe(train_output.metrics),
        "validation_metrics": _json_safe(eval_metrics),
    }


def train_baseline_model(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    hyperparameters: dict,
    output_dir: Path,
    save_artifacts: bool,
    model_name: str = BERT_MODEL_NAME,
    trial=None,
    tokenizer=None,
    use_domain_context: bool = False,
) -> dict:
    """Train the standard baseline model with the shared training runner."""
    return run_training(
        train_frame=train_frame,
        val_frame=val_frame,
        hyperparameters=hyperparameters,
        output_dir=output_dir,
        save_artifacts=save_artifacts,
        model_name=model_name,
        trial=trial,
        tokenizer=tokenizer,
        use_domain_context=use_domain_context,
        trainer_class=Trainer,
        trainer_init_kwargs=None,
        extra_log_fields=None,
        run_label="training run",
    )
