"""Central configuration for the BERT sarcasm detection pipeline."""

import re
from pathlib import Path


BASELINE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASELINE_DIR.parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DIAGNOSTIC_DATA_DIR = PROJECT_ROOT / "data" / "diagnostic"
ORIGINAL_DATA_DIR = PROCESSED_DATA_DIR / "original"
MASTER_DATA_DIR = PROCESSED_DATA_DIR / "master"

ORIGINAL_TRAIN_PATH = ORIGINAL_DATA_DIR / "train.csv"
ORIGINAL_VAL_PATH = ORIGINAL_DATA_DIR / "val.csv"
ORIGINAL_TEST_PATH = ORIGINAL_DATA_DIR / "test.csv"
MASTER_DATA_PATH = MASTER_DATA_DIR / "master_copy_dedup_v2.csv"
DIAGNOSTIC_VAL_PATH = DIAGNOSTIC_DATA_DIR / "validation_set.csv"

AUGMENTED_DATA_DIR = PROCESSED_DATA_DIR / "augmented"
AUGMENTED_TRAIN_PATH = AUGMENTED_DATA_DIR / "train.csv"
AUGMENTATION_CANDIDATES_PATH = PROJECT_ROOT / "augmentation_output_bert" / "augmentation_candidates.csv"
MASTER_TRAINING_PATH = MASTER_DATA_DIR / "train.csv"

OUTPUT_DIR = BASELINE_DIR / "outputs"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
TUNING_DIR = OUTPUT_DIR / "tuning"
ERROR_DIAGNOSTIC_RESULTS_DIR = OUTPUT_DIR / "error_diagnostic_results"
IMPROVED_WITH_TUNING_DIAGNOSTIC_ERRORS_PATH = (
    ERROR_DIAGNOSTIC_RESULTS_DIR / "improved_with_tuning_diagnostic_val" / "error_root_causes.csv"
)

TEXT_COLUMN = "headline"
LABEL_COLUMN = "is_sarcastic"
LINK_COLUMN = "article_link"
INDEX_COLUMN = "index"
DATASET_COLUMN = "dataset"
FILE_SOURCE_COLUMN = "file_source"

SEED = 42

BERT_MODEL_NAME = "bert-base-uncased"
BERT_LARGE_MODEL_NAME = "bert-large-uncased"

PRETRAINED_BERT_MODE = "pretrained"
PRETRAINED_LARGE_BERT_MODE = "pretrained_large"
ORIGINAL_NO_TUNING_MODE = "original_no_tuning"
ORIGINAL_WITH_TUNING_MODE = "original_with_tuning"
IMPROVED_NO_TUNING_MODE = "improved_no_tuning"
IMPROVED_LARGE_NO_TUNING_MODE = "improved_large_no_tuning"
IMPROVED_WITH_TUNING_MODE = "improved_with_tuning"
IMPROVED_LARGE_WITH_TUNING_MODE = "improved_large_with_tuning"
AUGMENTED_WITH_TUNING_MODE = "augmented_with_tuning"
MASTER_NO_TUNING_MODE = "master_no_tuning"

SUPPORTED_BERT_MODES = (
    PRETRAINED_BERT_MODE,
    PRETRAINED_LARGE_BERT_MODE,
    ORIGINAL_NO_TUNING_MODE,
    ORIGINAL_WITH_TUNING_MODE,
    IMPROVED_NO_TUNING_MODE,
    IMPROVED_LARGE_NO_TUNING_MODE,
    IMPROVED_WITH_TUNING_MODE,
    IMPROVED_LARGE_WITH_TUNING_MODE,
    AUGMENTED_WITH_TUNING_MODE,
    MASTER_NO_TUNING_MODE,
)

CALIBRATABLE_BERT_MODES = (
    IMPROVED_NO_TUNING_MODE,
    IMPROVED_LARGE_NO_TUNING_MODE,
    IMPROVED_WITH_TUNING_MODE,
    IMPROVED_LARGE_WITH_TUNING_MODE,
    AUGMENTED_WITH_TUNING_MODE,
    MASTER_NO_TUNING_MODE,
)

DEFAULT_BERT_MODE = ORIGINAL_WITH_TUNING_MODE

MAX_LENGTH = 64
PAD_TO_MULTIPLE_OF = 8
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 1e-5
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
EARLY_STOPPING_PATIENCE = 2
DEFAULT_N_TRIALS = 20
LR_SCHEDULER_TYPE = "cosine"
LABEL_SMOOTHING = 0.1
LLRD_DECAY_FACTOR = 0.9
USE_DOMAIN_CONTEXT = False

DEFAULT_TUNING_GRID = {
    "learning_rate": (1e-5, 5e-5),
    "batch_size": (8, 16, 32),
    "num_epochs": (3, 4, 5),
    "max_length": (64,),
    "warmup_ratio": (0.05, 0.20),
    "weight_decay": (0.005, 0.05),
    "llrd_decay_factor": (0.80, 0.95),
    "grad_accum": (1, 2, 4),
    "label_smoothing": (0.05, 0.15),
}

IMPROVED_TUNING_GRID = {
    **DEFAULT_TUNING_GRID,
    "contrastive_weight": (0.05, 0.20),
    "contrastive_temperature": (0.05, 0.15),
    "n_topic_clusters": (6, 8, 10),
}

IMPROVED_LARGE_TUNING_GRID = {
    **IMPROVED_TUNING_GRID,
    "batch_size": (4, 8),
    "grad_accum": (2, 4),
}

MODE_TUNING_CONFIGS = {
    ORIGINAL_WITH_TUNING_MODE: {
        "n_trials": DEFAULT_N_TRIALS,
        "grid": DEFAULT_TUNING_GRID,
    },
    IMPROVED_WITH_TUNING_MODE: {
        "n_trials": DEFAULT_N_TRIALS,
        "grid": IMPROVED_TUNING_GRID,
    },
    IMPROVED_LARGE_WITH_TUNING_MODE: {
        "n_trials": DEFAULT_N_TRIALS,
        "grid": IMPROVED_LARGE_TUNING_GRID,
    },
    AUGMENTED_WITH_TUNING_MODE: {
        "n_trials": DEFAULT_N_TRIALS,
        "grid": IMPROVED_TUNING_GRID,
    },
}

PRETRAINED_MODE_MODEL_NAMES = {
    PRETRAINED_BERT_MODE: BERT_MODEL_NAME,
    PRETRAINED_LARGE_BERT_MODE: BERT_LARGE_MODEL_NAME,
}


def get_mode_tuning_config(mode: str) -> dict:
    """Return the Optuna settings for a tuning-enabled mode."""
    try:
        config = MODE_TUNING_CONFIGS[mode]
    except KeyError as exc:
        raise KeyError(f"Mode '{mode}' does not define a tuning configuration.") from exc
    return {
        "n_trials": int(config["n_trials"]),
        "grid": dict(config["grid"]),
    }


def sanitize_name(value: str) -> str:
    """Convert an arbitrary string to a filesystem-safe lowercase identifier."""
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def get_mode_model_dir(mode: str) -> Path:
    return MODEL_OUTPUT_DIR / sanitize_name(mode)


def get_mode_metrics_path(mode: str, dataset_name: str) -> Path:
    return METRICS_DIR / f"{sanitize_name(mode)}_{sanitize_name(dataset_name)}_metrics.json"


def get_mode_predictions_path(mode: str, dataset_name: str) -> Path:
    return PREDICTIONS_DIR / f"{sanitize_name(mode)}_{sanitize_name(dataset_name)}_predictions.csv"


def get_mode_tuning_path(mode: str) -> Path:
    return METRICS_DIR / f"{sanitize_name(mode)}_tuning.json"


def get_mode_training_summary_path(mode: str) -> Path:
    return METRICS_DIR / f"{sanitize_name(mode)}_training_summary.json"


def get_mode_evaluation_summary_path(mode: str) -> Path:
    return METRICS_DIR / f"{sanitize_name(mode)}_evaluation_summary.csv"


def get_experiment_summary_path() -> Path:
    return METRICS_DIR / "bert_experiment_summary.csv"


def ensure_directories() -> None:
    """Create all required output directories if they do not already exist."""
    for path in (
        PROCESSED_DATA_DIR,
        DIAGNOSTIC_DATA_DIR,
        ORIGINAL_DATA_DIR,
        MASTER_DATA_DIR,
        AUGMENTED_DATA_DIR,
        OUTPUT_DIR,
        MODEL_OUTPUT_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        TUNING_DIR,
        ERROR_DIAGNOSTIC_RESULTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)

