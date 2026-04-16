# Baseline BERT Sarcasm Detection

This folder contains the BERT-based sarcasm detection pipeline for the project. The default backbone is `bert-base-uncased`, with optional `bert-large-uncased` variants for the larger improved modes.

The task is binary sarcasm detection on headline text. An optional domain-context mode (`--use-domain-context`) injects the article source domain as the second BERT segment, but this is intended as an ablation rather than the default setup.

## Setup

Create the environment from the file in this folder:

```bash
conda env create -f models/baseline_bert/environment.yml
conda activate CS4248-Project
```

The environment file uses a normal shared Conda env name and targets Python 3.11. It includes the packages needed for training, evaluation, local error diagnostics, and plotting.

Training is significantly faster with a CUDA-capable GPU. The exported environment currently uses CUDA 11.8 PyTorch wheels.

## Files

| File | Purpose |
|---|---|
| `core/` | Shared config, dataset loading, training, tuning, inference, and mode definitions |
| `train.py` | Unified training entrypoint for all baseline and improved BERT modes |
| `calibrate_and_evaluate.py` | Re-calibrates a saved fine-tuned model on the validation set and re-evaluates it without retraining |
| `evaluate.py` | Evaluates a saved model on a split or custom file |
| `run_all_evals.py` | Evaluates one or more modes across the standard BERT datasets |
| `predict.py` | Interactive, single-text, and batch inference |
| `error_diagnostic.py` | Local error-diagnostic analysis over saved prediction CSVs |
| `run_diagnostic_eval.py` | Runs the local error diagnostic over existing `diagnostic_val` prediction files |
| `data_augment.py` | BERT-based augmentation pipeline that uses diagnosed errors to retrieve corrective examples |

Related project assets used by this folder:

- `data/diagnostic/validation_set.csv`
- `data/processed/original/`
- `data/processed/master/`

The processed datasets used by this pipeline already exist under the repo `data/` folder.

## Modes

### Baseline modes

| Mode | Description |
|---|---|
| `pretrained` | `bert-base-uncased` with no fine-tuning |
| `pretrained_large` | `bert-large-uncased` with no fine-tuning |
| `original_no_tuning` | Fine-tuned on the original dataset with fixed hyperparameters |
| `original_with_tuning` | Fine-tuned on the original dataset after Optuna search |

### Improved modes

| Mode | Description |
|---|---|
| `improved_no_tuning` | Improved training with fixed hyperparameters |
| `improved_large_no_tuning` | Improved training with `bert-large-uncased` and safer large-model defaults |
| `improved_with_tuning` | Improved training after Optuna search |
| `improved_large_with_tuning` | Improved training with `bert-large-uncased` after Optuna search |
| `augmented_with_tuning` | Improved training on augmentation candidates, reusing `improved_with_tuning` hyperparameters by default |
| `master_no_tuning` | Improved training on the full master dataset, reusing `improved_with_tuning` hyperparameters with scale adjustments |

Standard post-training evaluation targets used by `run_all_evals.py` are:

- `data/processed/original/test.csv`
- `data/diagnostic/validation_set.csv`
- `data/processed/master/master_copy_dedup_v2.csv` *(excluded by default; pass `--include-master-eval` to include)*

## Training behavior

All fine-tuned runs use:

- cosine learning-rate scheduling
- label smoothing
- layer-wise learning-rate decay
- gradient accumulation
- `max_length=64` by default
- dynamic padding
- BF16 / TF32 / padding-to-8 when supported by the detected CUDA setup

The tuning-enabled modes use Optuna with a TPE sampler and median pruner.

Change the Optuna trial count and search spaces in `models/baseline_bert/core/config.py`. The training CLI no longer exposes per-run tuning overrides such as `--n-trials`.

The default baseline search space in `core/config.py` is:

| Parameter | Range |
|---|---|
| learning rate | `1e-5` to `5e-5` |
| batch size | `8`, `16`, `32` |
| epochs | `3`, `4`, `5` |
| max token length | `64` |
| warmup ratio | `0.05` to `0.20` |
| weight decay | `0.005` to `0.05` |
| LLRD decay factor | `0.80` to `0.95` |
| gradient accumulation | `1`, `2`, `4` |
| label smoothing | `0.05` to `0.15` |

The improved modes extend that grid with:

- `contrastive_weight`
- `contrastive_temperature`
- `n_topic_clusters`

## Improved model additions

The improved modes add three main ideas:

### 1. Topic-balanced sampling

Training headlines are clustered with TF-IDF plus K-Means. Each example receives a sampling weight based on its `(topic cluster, label)` bucket so the model does not overfit topic-label shortcuts.

### 2. Supervised contrastive loss

A supervised contrastive loss is added on top of cross-entropy using the final CLS representation.

### 3. Threshold calibration

The improved family (`improved_*`, `augmented_with_tuning`, and `master_no_tuning`) supports post-training threshold calibration on the original validation split. The calibrated threshold is created by `evaluate.py` or `calibrate_and_evaluate.py` on first use and saved to:

```text
models/baseline_bert/outputs/models/{mode}/threshold.json
```

## Running the main training modes

Baseline:

```bash
python models/baseline_bert/train.py --mode original_no_tuning
python models/baseline_bert/train.py --mode original_with_tuning
```

Improved:

```bash
python models/baseline_bert/train.py --mode improved_no_tuning
python models/baseline_bert/train.py --mode improved_large_no_tuning
python models/baseline_bert/train.py --mode improved_with_tuning
python models/baseline_bert/train.py --mode improved_large_with_tuning
```

`train.py` writes the training summary and saved model artifacts only. To produce dataset-level metrics and prediction CSVs, run `evaluate.py` or `run_all_evals.py` after training.

For `pretrained` and `pretrained_large`, `train.py` does not fit or save a local model. It only writes a summary record for that mode. Use `evaluate.py` or `run_all_evals.py` to generate the actual dataset metrics for those zero-shot baselines.

`improved_large_no_tuning` uses safer default large-model settings to reduce GPU memory pressure.

`improved_large_with_tuning` narrows the large-model batch/grid defaults so tuning is less likely to hit OOM on a typical single GPU.

## Augmentation workflow

The augmentation pipeline is driven by diagnosed `improved_with_tuning` validation errors.

Current flow:

```text
train.py -> training summary + checkpoint -> evaluate.py (diagnostic_val) -> run_diagnostic_eval.py -> error_diagnostic_results/.../error_root_causes.csv -> data_augment.py -> augmentation candidates -> train.py --mode augmented_with_tuning
```

### Step 1. Train the source mode

```bash
python models/baseline_bert/train.py --mode improved_with_tuning
```

### Step 2. Generate diagnostic-set predictions

```bash
python models/baseline_bert/evaluate.py \
    --mode improved_with_tuning \
    --split data/diagnostic/validation_set.csv \
    --dataset-name diagnostic_val
```

If you also want the standard evaluation sweep, use:

```bash
python models/baseline_bert/run_all_evals.py --mode improved_with_tuning
```

### Step 3. Run the local diagnostic

```bash
python models/baseline_bert/run_diagnostic_eval.py --mode improved_with_tuning
```

This uses the `diagnostic_val` predictions already written by evaluation:

- validation set: `data/diagnostic/validation_set.csv`
- predictions: `models/baseline_bert/outputs/predictions/`
- diagnostic results: `models/baseline_bert/outputs/error_diagnostic_results/`

If the diagnostic predictions are missing for a mode, generate them with `evaluate.py` or `run_all_evals.py`.

### Step 4. Run the augmentation pipeline

Run from the project root:

```bash
python models/baseline_bert/data_augment.py
```

On the original environment, `KMP_DUPLICATE_LIB_OK=TRUE` may be required:

```bash
KMP_DUPLICATE_LIB_OK=TRUE python models/baseline_bert/data_augment.py
```

The script reads diagnosed errors from:

```text
models/baseline_bert/outputs/error_diagnostic_results/improved_with_tuning_diagnostic_val/error_root_causes.csv
```

and writes augmentation artifacts to:

```text
data/augmentation_output/baseline_bert/improved_with_tuning/
```

### Step 5. Train the augmented model

```bash
python models/baseline_bert/train.py --mode augmented_with_tuning
```

To re-run Optuna instead of reusing `improved_with_tuning` hyperparameters:

```bash
python models/baseline_bert/train.py --mode augmented_with_tuning --retune
```

### Alternative. Train on the full master dataset

```bash
python models/baseline_bert/train.py --mode master_no_tuning
```

This mode reuses `improved_with_tuning` tuning results when available and applies scale adjustments for the larger training set.

## Calibration and re-evaluation

If a model is already trained, you can calibrate and re-evaluate without retraining:

```bash
python models/baseline_bert/calibrate_and_evaluate.py --mode improved_no_tuning
```

All six calibratable modes are supported:

```bash
python models/baseline_bert/calibrate_and_evaluate.py --mode improved_no_tuning
python models/baseline_bert/calibrate_and_evaluate.py --mode improved_large_no_tuning
python models/baseline_bert/calibrate_and_evaluate.py --mode improved_with_tuning
python models/baseline_bert/calibrate_and_evaluate.py --mode improved_large_with_tuning
python models/baseline_bert/calibrate_and_evaluate.py --mode augmented_with_tuning
python models/baseline_bert/calibrate_and_evaluate.py --mode master_no_tuning
```

The `pretrained`, `pretrained_large`, `original_no_tuning`, and `original_with_tuning` modes are not supported because they are not part of the calibratable mode family.

By default `calibrate_and_evaluate.py` evaluates on all three datasets including `master_copy_dedup_v2`. To skip the master evaluation:

```bash
python models/baseline_bert/calibrate_and_evaluate.py --mode improved_with_tuning --skip-master-eval
```

## Local error diagnostics

### Run the diagnostic after evaluation

Run the local diagnostic over the `diagnostic_val` predictions already created by `evaluate.py` or `run_all_evals.py`:

```bash
python models/baseline_bert/run_diagnostic_eval.py --mode improved_with_tuning
python models/baseline_bert/run_diagnostic_eval.py --mode all
```

### Diagnose an existing predictions CSV

```bash
python models/baseline_bert/error_diagnostic.py --mode improved_with_tuning --dataset diagnostic_val
```

You can also auto-discover:

```bash
python models/baseline_bert/error_diagnostic.py --mode all --dataset all
```

### Outputs

Diagnostic artifacts are written to:

```text
models/baseline_bert/outputs/error_diagnostic_results/{mode}_{dataset}/
```

Each run produces:

- `error_root_causes.csv`
- `cluster_report.txt`
- `error_distribution.png`
- `tsne_analysis.png`
- `cluster_analysis.png`

## Evaluation and inference

Evaluate a saved model on a split or custom file:

```bash
python models/baseline_bert/evaluate.py \
    --mode original_with_tuning \
    --split data/processed/master/master_copy_dedup_v2.csv \
    --dataset-name master_copy_dedup_v2
```

Evaluate a zero-shot baseline directly:

```bash
python models/baseline_bert/evaluate.py \
    --mode pretrained_large \
    --split data/processed/master/master_copy_dedup_v2.csv \
    --dataset-name master_copy_dedup_v2
```

For calibrated improved-family modes, `evaluate.py` now auto-loads `outputs/models/{mode}/threshold.json` when it exists. If the threshold file is missing, `evaluate.py` calibrates on `data/processed/original/val.csv`, saves `threshold.json`, and then evaluates the requested dataset. The original baseline modes and pretrained modes stay at `0.5` unless you pass `--threshold` manually.

To match the Trainer validation semantics from training exactly, use:

```bash
python models/baseline_bert/evaluate.py \
    --mode improved_with_tuning \
    --split data/processed/original/val.csv \
    --dataset-name original_val \
    --match-training-validation
```

Run all supported modes across `original_test` and `diagnostic_val` (master skipped by default):

```bash
python models/baseline_bert/run_all_evals.py --mode all
```

Include `master_copy_dedup_v2` as well:

```bash
python models/baseline_bert/run_all_evals.py --mode all --include-master-eval
```

Apply a calibrated threshold manually:

```bash
python models/baseline_bert/evaluate.py \
    --mode improved_with_tuning \
    --split data/processed/master/master_copy_dedup_v2.csv \
    --dataset-name master_copy_dedup_v2 \
    --threshold 0.72
```

Single prediction:

```bash
python models/baseline_bert/predict.py \
    --mode original_with_tuning \
    --text "oh great, another monday morning meeting"
```

Interactive mode:

```bash
python models/baseline_bert/predict.py --mode original_with_tuning
```

Batch inference:

```bash
python models/baseline_bert/predict.py \
    --mode original_with_tuning \
    --input-path data/processed/original/test.csv \
    --dataset-name original_test \
    --output-path my_predictions.csv
```

## Output layout

Main model artifacts are written under:

```text
models/baseline_bert/outputs/
|-- models/
|-- metrics/
|-- predictions/
|-- tuning/
`-- error_diagnostic_results/
```

Additional cross-folder outputs:

- augmented merged training split: `data/processed/augmented/train.csv`
- augmentation candidates and reports: `data/augmentation_output/baseline_bert/improved_with_tuning/`

## Notes

- Domain context is off by default.
- `max_length=64` is the default because the inputs are short headline-only texts.
- On Windows, `dataloader_num_workers` stays at `0`.
- If you retrain `improved_with_tuning`, regenerate the diagnostic results before rerunning `data_augment.py`.
- `master_no_tuning` excludes original validation and test headlines from its training set.

