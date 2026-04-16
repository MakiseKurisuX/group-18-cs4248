# `run_sarcasm.py` — DeBERTa Sarcasm Detection with Optuna

End-to-end pipeline for training a **DeBERTa-v3-base** model with **LoRA** (Low-Rank Adaptation) on the News Headlines Dataset for Sarcasm Detection. Hyperparameters are tuned automatically via **Optuna**, and the best configuration is retrained on the full training data before final evaluation.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Project Structure](#project-structure)
3. [Data](#data)
4. [How It Works](#how-it-works)
5. [Usage](#usage)
6. [Hyperparameter Search Space](#hyperparameter-search-space)
7. [Outputs](#outputs)
8. [Key Configuration](#key-configuration)
9. [How LoRA Works with Optuna](#how-lora-works-with-optuna)
10. [Error Diagnostic Pipeline](#error-diagnostic-pipeline)

---

## Requirements

| Package | Purpose |
|---|---|
| `torch` | Deep learning backend |
| `transformers` | DeBERTa model & Trainer API |
| `peft` | LoRA adapter injection |
| `optuna` | Bayesian hyperparameter optimisation |
| `datasets` | HuggingFace Dataset abstraction |
| `scikit-learn` | Metrics & train/test splitting |
| `pandas`, `numpy` | Data wrangling |

Install with:

```bash
pip install torch transformers peft optuna datasets scikit-learn pandas numpy
```

We ran this on the NUS SoC cluster using A100 GPUs — approximately 100 minutes for 10 trial runs. The script auto-detects GPU availability and selects the appropriate mixed-precision mode (bf16 for Ampere+, fp16 otherwise, none for CPU).

This was the .sh file used in order to train on the cluster.

```
#!/bin/bash
#SBATCH --job-name=deberta_lora
#SBATCH --gpus=a100-40
#SBATCH --time=03:00:00
#SBATCH --output=sarcasm_%j.out
#SBATCH --error=sarcasm_%j.err

unset CUDA_VISIBLE_DEVICES
export HF_HOME=~/hf-cache
export CUDA_LAUNCH_BLOCKING=0

source ~/venv/bin/activate
cd ~/ml-nlp
nvidia-smi
python ~/ml-nlp/run_sarcasm.py
```

---

## Data

### Training / Validation Set

Loaded from `{location}/input/Sarcasm_Headlines_Dataset.json` (JSONL format).

Each line must contain:
- `headline` — the news headline text
- `is_sarcastic` — binary label (`0` or `1`)
- `article_link` — used to infer the source (e.g. The Onion, HuffPost)

A **90/10 train/val split** is performed with stratification (joint label + source when possible, falling back to label-only or random).

### Test Set

Loaded separately from `{location}/output/testing_dataset_final.csv` with the same schema.

---

## How It Works

The pipeline runs in four sequential phases:

### Phase 1 — Data Preparation

1. Load JSONL training data; derive feature columns (`source`, `text_len`, `token_len_approx`, `exclamation`, `question`).
2. Split into train (90%) and validation (10%) with stratified sampling.
3. Load the external test CSV.
4. Tokenize all splits with the `microsoft/deberta-v3-base` tokenizer (max length 128).
5. Save a `dataset_profile.json` snapshot.

### Phase 2 — Optuna Hyperparameter Search

For each of **N trials** (default 10):

1. Sample training and LoRA hyperparameters (see [search space](#hyperparameter-search-space)).
2. Attach a LoRA adapter to a fresh DeBERTa base model.
3. Train with early stopping (patience = 2 epochs, metric = F1).
4. Evaluate on **both** val and test sets; record the val–test F1 gap.
5. Report val F1 for Optuna's **MedianPruner** (5 startup trials, 1 warmup step).

The study uses a **TPE sampler** and maximises validation F1.

### Phase 3 — Retrain Best Model

The best hyperparameters are used to retrain a fresh LoRA model on the **combined train + val data** for the optimal number of epochs, leaving the test set completely untouched during training.

### Phase 4 — Evaluation & Error Analysis

1. Predict on the test set; compute accuracy, precision, recall, F1, and ROC-AUC.
2. Save per-sample predictions with probabilities and confidence scores.
3. Compute **slice metrics** (by source, headline length bin, exclamation/question marks).
4. Extract the top-200 **high-confidence errors** and top-200 **uncertain samples**.
5. Generate a Markdown **gap report** summarising performance, worst slices, and pointers to error files.

---

## Usage

```bash
python run_sarcasm.py
```

There are no CLI arguments. All configuration is set via constants at the top of the file (see [Key Configuration](#key-configuration)). Each run creates a timestamped directory under `{location}/results/` (e.g. `20260415_221000/`).

---

## Hyperparameter Search Space

| Parameter | Type | Range / Choices |
|---|---|---|
| `learning_rate` | float (log) | `1e-5` – `1e-3` |
| `warmup_ratio` | float | `0.0` – `0.2` |
| `weight_decay` | float | `0.0` – `0.1` |
| `batch_size` | categorical | `16`, `32` |
| `num_epochs` | categorical | `3`, `4`, `5` |
| `label_smoothing` | float | `0.0` – `0.15` |
| `lora_r` | categorical | `4`, `8`, `16`, `32` |
| `lora_alpha` | categorical | `8`, `16`, `32`, `64` |
| `lora_dropout` | float | `0.0` – `0.3` |

LoRA is applied to the **query, key, value, and dense** projection layers.

---

## Outputs

All artifacts are written to `deBERTa/results/<run_id>/`:

| File | Description |
|---|---|
| `dataset_profile.json` | Row count, label distribution, top sources, text-length stats |
| `trial_results.csv` | Val & test metrics for every Optuna trial, sorted by val F1 |
| `best_params.json` | The winning hyperparameters |
| `training_args.json` | Full HuggingFace `TrainingArguments` for the final retrain |
| `metrics.json` | Final test metrics, confusion matrix, LoRA config, GPU info |
| `predictions.csv` | Per-sample predictions with probabilities, confidence, and error type |
| `slice_metrics.csv` | Accuracy / F1 / AUC broken down by source, length, punctuation |
| `high_confidence_errors.csv` | Top-200 misclassifications ranked by model confidence |
| `uncertain_samples.csv` | Top-200 samples closest to the 0.5 decision boundary |
| `gap_report.md` | Human-readable summary: metrics, worst slices, error pointers |

---

## Key Configuration

These constants are defined at the top of `run_sarcasm.py` and can be edited before running:

```python
SEED         = 42
MODEL_NAME   = 'microsoft/deberta-v3-base'
DATA_PATH    = '{location}/input/Sarcasm_Headlines_Dataset.json'
N_TRIALS     = 10                 # Number of Optuna trials
```

- **Validation split size** is `0.1` (set in the `safe_split` call).
- **Max token length** is `128` (set in the `tokenize` function).
- **Early stopping patience** is `2` epochs.
- **Pruner** uses `MedianPruner(n_startup_trials=5, n_warmup_steps=1)`.

---

## How LoRA Works with Optuna

### What is LoRA?

[LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) is a parameter-efficient fine-tuning method. Instead of updating all ~86 M parameters of DeBERTa-v3-base, LoRA **freezes** the pretrained weights and injects small trainable rank-decomposition matrices into selected layers. In this script, adapters are attached to four projection layers in every Transformer block:

```
query_proj, key_proj, value_proj, dense
```

The key LoRA hyperparameters are:

| Parameter | Role |
|---|---|
| `r` (rank) | Dimension of the low-rank matrices. Higher → more capacity, more parameters |
| `alpha` | Scaling factor applied to the adapter output (effective scale = `alpha / r`) |
| `dropout` | Dropout on the adapter path for regularisation |

With typical settings (e.g. `r=8, alpha=16`), only **~0.2–0.5%** of the total parameters are trainable, which makes each training run **much faster and cheaper** than full fine-tuning.

### How Optuna Orchestrates the Search

Optuna treats each trial as an independent experiment. On every trial the script:

1. **Samples a full configuration** — Optuna's TPE (Tree-structured Parzen Estimator) sampler proposes values for all 9 hyperparameters jointly, covering both LoRA-specific knobs (`r`, `alpha`, `dropout`) and standard training knobs (`learning_rate`, `batch_size`, `num_epochs`, etc.).

2. **Builds a fresh model** — A new DeBERTa base model is loaded from HuggingFace and a LoRA adapter is injected with the sampled `r`, `alpha`, and `dropout`. This ensures no weight leakage between trials.

3. **Trains and evaluates** — The LoRA-augmented model is trained on the training split with early stopping (patience = 2). After training, it is evaluated on both the validation and test sets.

4. **Reports val F1 to Optuna** — Optuna records the validation F1 score and uses it to:
   - Guide future sampling (TPE builds a probabilistic model of "good" vs "bad" parameter regions).
   - Decide whether to **prune** (terminate early) underperforming trials via the MedianPruner.

5. **Frees GPU memory** — The model and trainer are explicitly deleted and `torch.cuda.empty_cache()` is called before the next trial.

### Why LoRA Makes This Feasible

Full fine-tuning of DeBERTa for 10 Optuna trials would be prohibitively expensive. LoRA reduces the trainable parameter count by **~200×**, which means:

- Each trial trains in **minutes** instead of tens of minutes.
- GPU memory usage is significantly lower, allowing larger batch sizes.
- The total search (10 trials) can complete in roughly the time a single full fine-tuning run would take.

### Final Retrain

After the search completes, the best hyperparameters are used to retrain a **fresh** LoRA model on the combined train + val data (to maximise the amount of data seen), and the final model is evaluated on the held-out test set. This two-stage process (HPO on val → retrain on train+val → evaluate on test) prevents information leakage into the test metrics.

---

## Error Diagnostic Pipeline

`error_diagnostic_pipeline.ipynb` is a **post-training analysis notebook** that takes the predictions produced by `run_sarcasm.py` and investigates *why* the model makes errors. It combines KNN-based neighbourhood analysis, heuristic root-cause categorisation, t-SNE visualisation, and KMeans cluster analysis to produce an actionable error report.

A standalone CLI version of the same pipeline lives in `error_diagnostic/error_diagnostic.py`.

### Inputs

| File | Description |
|---|---|
| `{location}/results/<run_id>/predictions.csv` | Per-sample predictions (produced by Phase 4 of `run_sarcasm.py`) |
| `{location}/input/Sarcasm_Headlines_Dataset.json` | The full reference / training dataset (JSONL) |
| `{location}/results/<run_id>/best_checkpoints/checkpoint-*` | A saved LoRA checkpoint whose encoder is used to compute embeddings |

### Configuration (Cell 2)

All paths and parameters are set in the notebook's second code cell:

```python
PREDICTIONS_CSV = Path("{location}/results/<run_id>/predictions.csv")
DATASET_JSON    = Path("{location}/input/Sarcasm_Headlines_Dataset.json")
EMBEDDING_MODEL = Path("{location}/results/<run_id>/best_checkpoints/checkpoint-XXXX").resolve()
BATCH_SIZE      = 64
K_NEIGHBORS     = 10
```

Update `<run_id>` and the checkpoint number to match the training run you want to diagnose.

### Notebook Walkthrough

The notebook is organised into the following sequential steps:

| Cell | Step | What It Does |
|---|---|---|
| 1 | Imports | Loads pandas, torch, transformers, sklearn, matplotlib, seaborn |
| 2 | Configuration | Sets file paths, batch size, K, and device |
| 3 | Load Predictions | Reads `predictions.csv` and filters to only the misclassified rows |
| 4 | Load Reference Data | Reads the full JSONL dataset as the KNN knowledge base |
| 5 | Compute Embeddings | Loads the LoRA checkpoint (merged into the base model), then mean-pools + L2-normalises the encoder hidden states for both the reference set and the error set |
| 6 | KNN Search | Fits a cosine-distance KNN on the reference embeddings and queries it with the error embeddings |
| 7 | Heuristic Root-Cause Categorisation | Assigns each error to one of six diagnostic "angles" (see below) |
| 8 | Bar Chart | Plots the distribution of errors across the six categories |
| 9 | Peek at Examples | Displays sample headlines for Label Conflict and Outlier categories |
| 10 | t-SNE Visualisation | Projects all 768-D embeddings to 2-D; plots side-by-side unclustered and FP/FN-coloured scatter plots |
| 11 | KMeans Cluster Analysis | Clusters the error embeddings (K = 8), auto-names each cluster with TF-IDF keywords, and overlays them on the t-SNE plot |
| 12–14 | Top Word Features | Analyses the most common unigrams/bigrams in true positives, true negatives, false positives, and false negatives |

### The Six Diagnostic Angles

Every misclassified sample is assigned to exactly one root-cause category using the following heuristic decision tree (evaluated in priority order):

| Priority | Angle | Condition | Interpretation |
|---|---|---|---|
| 1 | **Outlier / Zero-Shot Zone** | Closest neighbour distance > 0.15 | The headline is far from anything in the training set — the model has never seen anything like it |
| 2 | **Low-Signal Ambiguity** | Confidence < 0.60 *and* token length < 6 | The headline is too short/vague for the model to extract a reliable signal |
| 3 | **Systematic Bias / Spurious Rule** | Closest neighbour has the *same* label as the prediction *and* confidence > 0.90 | The model has learned a spurious correlation that makes it confidently wrong |
| 4 | **Punctuation Spurious Correlation** | False positive *and* contains `!` or `?` *and* closest distance > 0.10 | The model over-relies on exclamation/question marks as sarcasm indicators |
| 5 | **Label Conflict (Fuzzy Boundary)** | Closest neighbour has a *different* label from the true label | Nearly identical headlines exist with opposite labels in the dataset |
| 6 | **Mixed Neighbourhood** | 30–70% of K neighbours are sarcastic | The headline sits in a tangled region where sarcastic and non-sarcastic samples are interleaved |

Anything that doesn't match the above is marked **Uncategorized Error**.

### Outputs

The notebook produces inline plots and DataFrames. The key exportable artefact is:

| Output | Description |
|---|---|
| `error_root_causes.csv` | One row per misclassified sample with all prediction columns, KNN statistics (closest distance, neighbour sarcastic ratio), the assigned diagnostic angle, and KMeans cluster ID + theme |
| Inline plots | Error distribution bar chart, t-SNE scatter (unclustered + FP/FN), KMeans cluster overlay, top-word bar charts for TP/TN/FP/FN |
| Cluster report | Printed text report showing each KMeans cluster's size, FP/FN split, TF-IDF keywords, and sample headlines |
