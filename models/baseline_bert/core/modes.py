"""Mode registry and dataset split loaders for the baseline BERT package.

This module defines the small amount of behavior that varies across experiment
variants:

- which train/validation splits to load
- which default hyperparameters to start from
- which training function to call
- whether Optuna tuning and threshold calibration apply
- whether a mode reuses tuning from another mode

The goal is to keep those choices declarative so ``train.py`` can remain a thin
orchestrator instead of accumulating a long chain of mode-specific conditionals.
"""

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .artifacts import save_csv
from .config import (
    AUGMENTED_WITH_TUNING_MODE,
    BERT_LARGE_MODEL_NAME,
    BERT_MODEL_NAME,
    DIAGNOSTIC_VAL_PATH,
    IMPROVED_LARGE_NO_TUNING_MODE,
    IMPROVED_LARGE_WITH_TUNING_MODE,
    IMPROVED_NO_TUNING_MODE,
    IMPROVED_WITH_TUNING_MODE,
    MASTER_DATA_PATH,
    MASTER_NO_TUNING_MODE,
    MASTER_TRAINING_PATH,
    ORIGINAL_NO_TUNING_MODE,
    ORIGINAL_TEST_PATH,
    ORIGINAL_TRAIN_PATH,
    ORIGINAL_VAL_PATH,
    ORIGINAL_WITH_TUNING_MODE,
    PRETRAINED_BERT_MODE,
    PRETRAINED_LARGE_BERT_MODE,
    TEXT_COLUMN,
)
from .dataset import load_input_dataframe
from .improved import (
    build_improved_trial_hyperparameters,
    get_improved_default_hyperparameters,
    get_improved_large_default_hyperparameters,
    load_augmented_candidates,
    persist_augmented_train,
    train_improved_model,
)
from .training import get_default_hyperparameters, log_step, train_baseline_model
from .tuning import build_trial_hyperparameters


def load_original_splits() -> dict[str, pd.DataFrame]:
    """Load the standard train/validation splits from ``data/processed/original``."""
    log_step("Loading original dataset splits from data/processed/original/")
    return {
        "train": load_input_dataframe(str(ORIGINAL_TRAIN_PATH), dataset_name="original_train"),
        "val": load_input_dataframe(str(ORIGINAL_VAL_PATH), dataset_name="original_val"),
    }


def load_augmented_splits() -> dict[str, pd.DataFrame]:
    """Build the augmented training split by appending accepted candidates."""
    original_splits = load_original_splits()
    candidates = load_augmented_candidates()
    augmented_train = pd.concat([original_splits["train"], candidates], ignore_index=True)
    persist_augmented_train(augmented_train)
    log_step(
        f"Augmented training set: {len(original_splits['train']):,} original "
        f"+ {len(candidates):,} candidates = {len(augmented_train):,} total examples"
    )
    return {"train": augmented_train, "val": original_splits["val"]}


def load_master_splits() -> dict[str, pd.DataFrame]:
    """Prepare the master training split while excluding original val/test headlines.

    This prevents direct text leakage from the held-out original validation and
    test sets into the much larger master-copy training pool.
    """
    if not MASTER_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Master dataset not found at {MASTER_DATA_PATH}. Ensure the file exists before running master_no_tuning."
        )

    master_frame = load_input_dataframe(str(MASTER_DATA_PATH), dataset_name="master_copy_dedup_v2")
    val_frame = load_input_dataframe(str(ORIGINAL_VAL_PATH), dataset_name="original_val")
    test_frame = load_input_dataframe(str(ORIGINAL_TEST_PATH), dataset_name="original_test")

    exclude_headlines = set(
        pd.concat([val_frame[TEXT_COLUMN], test_frame[TEXT_COLUMN]], ignore_index=True)
        .dropna()
        .str.strip()
        .tolist()
    )
    filtered_master = master_frame[~master_frame[TEXT_COLUMN].str.strip().isin(exclude_headlines)].reset_index(drop=True)
    save_csv(filtered_master, MASTER_TRAINING_PATH)
    log_step(
        f"Master training set prepared: {len(filtered_master):,} rows retained "
        f"after excluding {len(exclude_headlines):,} val/test headlines"
    )
    return {"train": filtered_master, "val": val_frame}


def get_evaluation_targets(include_master_copy: bool = True, include_diagnostic_val: bool = True) -> list[dict]:
    """Return the ordered list of evaluation datasets for a training run."""
    targets = [
        {
            "path": str(ORIGINAL_TEST_PATH),
            "dataset_name": "original_test",
        },
    ]
    if include_master_copy:
        targets.append(
            {
                "path": str(MASTER_DATA_PATH),
                "dataset_name": "master_copy_dedup_v2",
            }
        )
    if include_diagnostic_val:
        targets.append(
            {
                "path": str(DIAGNOSTIC_VAL_PATH),
                "dataset_name": "diagnostic_val",
            }
        )
    return targets


@dataclass(frozen=True)
class ModeSpec:
    """Declarative configuration for one experiment mode."""
    load_splits: Callable[..., object] | None
    default_hyperparameters: Callable[..., object] | None
    train_model: Callable[..., object] | None
    model_name: str = BERT_MODEL_NAME
    trial_hyperparameters_builder: Callable[..., object] | None = None
    auto_calibrate_threshold: bool = False
    reuse_tuning_from: str | None = None
    improvements: tuple[str, ...] = ()


IMPROVED_MODE_SET = frozenset(
    {
        IMPROVED_NO_TUNING_MODE,
        IMPROVED_LARGE_NO_TUNING_MODE,
        IMPROVED_WITH_TUNING_MODE,
        IMPROVED_LARGE_WITH_TUNING_MODE,
        AUGMENTED_WITH_TUNING_MODE,
        MASTER_NO_TUNING_MODE,
    }
)

# Central registry used by ``train.py`` to map ``--mode`` to concrete behavior.
MODE_SPECS: dict[str, ModeSpec] = {
    PRETRAINED_BERT_MODE: ModeSpec(
        load_splits=None,
        default_hyperparameters=None,
        train_model=None,
    ),
    PRETRAINED_LARGE_BERT_MODE: ModeSpec(
        load_splits=None,
        default_hyperparameters=None,
        train_model=None,
        model_name=BERT_LARGE_MODEL_NAME,
    ),
    ORIGINAL_NO_TUNING_MODE: ModeSpec(
        load_splits=load_original_splits,
        default_hyperparameters=get_default_hyperparameters,
        train_model=train_baseline_model,
    ),
    ORIGINAL_WITH_TUNING_MODE: ModeSpec(
        load_splits=load_original_splits,
        default_hyperparameters=get_default_hyperparameters,
        train_model=train_baseline_model,
        trial_hyperparameters_builder=build_trial_hyperparameters,
    ),
    IMPROVED_NO_TUNING_MODE: ModeSpec(
        load_splits=load_original_splits,
        default_hyperparameters=get_improved_default_hyperparameters,
        train_model=train_improved_model,
        auto_calibrate_threshold=True,
        improvements=("topic_balanced_sampling", "supervised_contrastive_loss"),
    ),
    IMPROVED_LARGE_NO_TUNING_MODE: ModeSpec(
        load_splits=load_original_splits,
        default_hyperparameters=get_improved_large_default_hyperparameters,
        train_model=train_improved_model,
        model_name=BERT_LARGE_MODEL_NAME,
        auto_calibrate_threshold=True,
        improvements=("topic_balanced_sampling", "supervised_contrastive_loss"),
    ),
    IMPROVED_WITH_TUNING_MODE: ModeSpec(
        load_splits=load_original_splits,
        default_hyperparameters=get_improved_default_hyperparameters,
        train_model=train_improved_model,
        trial_hyperparameters_builder=build_improved_trial_hyperparameters,
        auto_calibrate_threshold=True,
        improvements=("topic_balanced_sampling", "supervised_contrastive_loss"),
    ),
    IMPROVED_LARGE_WITH_TUNING_MODE: ModeSpec(
        load_splits=load_original_splits,
        default_hyperparameters=get_improved_large_default_hyperparameters,
        train_model=train_improved_model,
        model_name=BERT_LARGE_MODEL_NAME,
        trial_hyperparameters_builder=build_improved_trial_hyperparameters,
        auto_calibrate_threshold=True,
        improvements=("topic_balanced_sampling", "supervised_contrastive_loss"),
    ),
    AUGMENTED_WITH_TUNING_MODE: ModeSpec(
        load_splits=load_augmented_splits,
        default_hyperparameters=get_improved_default_hyperparameters,
        train_model=train_improved_model,
        trial_hyperparameters_builder=build_improved_trial_hyperparameters,
        auto_calibrate_threshold=True,
        reuse_tuning_from=IMPROVED_WITH_TUNING_MODE,
        improvements=("topic_balanced_sampling", "supervised_contrastive_loss"),
    ),
    MASTER_NO_TUNING_MODE: ModeSpec(
        load_splits=load_master_splits,
        default_hyperparameters=get_improved_default_hyperparameters,
        train_model=train_improved_model,
        auto_calibrate_threshold=True,
        reuse_tuning_from=IMPROVED_WITH_TUNING_MODE,
        improvements=("topic_balanced_sampling", "supervised_contrastive_loss"),
    ),
}
