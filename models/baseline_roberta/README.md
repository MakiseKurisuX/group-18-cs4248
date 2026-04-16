# Baseline RoBERTa Sarcasm Detection

This folder contains the scripts, data, and outputs for RoBERTa Sarcasm Detection. All variations of models are built upon `roberta-base`.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Python version used is 3.12. Python versions 3.13+ may not work well with certain imports.

## Files

| File | Purpose |
| --- | --- |
| `dataset.py ` | SarcasmDataset and SarcasmDatasetWithContext Classes |
| `evaluation.py` | Script to perform model evaluation |
| `hft_optuna.py` | Script to run optuna hyperparameter tuning |
| `predict.py` | Script to generate predictions.csv |
| `preprocess.py` | Generate train, test, val splits |
| `retrieve_article_context.py` | Retrieve article context for URLs in Kaggle dataset |
| `trainer_augmented.py` | Training script for the data augmentation |
| `trainer_baseline.py` | Training script for baseline roberta using `roberta-base` |
| `trainer_improved.py` | Training script for the improved roberta |
| `trainer.py` | Stripped down training script |

---

## RoBERTa variants

| Mode | Description |
| `pretrained` | Default `roberta-base` |
| `original_no_tuning` | Default `roberta-base` rameters |
| `original_with_tuning` | Default `roberta-base` after Optuna hyperparameter fine-tuning |
| `improved_no_tuning` | Improved RoBERTa model rameters |
| `improved_with_tuning` | Improved RoBERTa model after Optuna hyperparamter fine-tuning |

Each model is evaluated on (from project root):
- `data/processed/original/test.csv`
- `data/processed/master/master_copy_dedup_v2.csv`
- `data/validation_set.csv`

---

## Hyperparamter fine-tuning

All e following techniques:

| Parameter | Range | Type |
| --- | --- |
| Linear LR Schedule | ['cosine', 'linear', 'constant_with_warmup'] | Categorical |
| Label Smoothing | [0.0, 0.2] | Continuous |
| Learning Rate | [1e-5, 1e-4] | Continuous |
| Batch Size | [8, 16, 32, 64] | Categorical |
| Warmup Steps | [0, 1000] | Continuous with step 100 |
| Weight Decay | [0.0, 0.3] | Continuous |
| Epoch Count | [2, 3, 4, 5] | Categorical |

---

## Improved RoBERTa Model

The improved RoBERTa model uses the scripts `trainer_improved.py` for training. `retrieve_article_context.py` is used to obtain article description from the URLs given in the Kaggle dataset.

In `dataset.py`, the `SarcasmDatasetWithContext` class is used to handle this new context, and `build_model_input` appends the additional context to the headline before it is passed as model input for training.

---

### Data Augmentation

### Overview

The `augmented_with_tuning` mode trains on an augmented dataset produced by `data_augment.py`. The pipeline analyses the model's diagnosed error patterns and surgically retrieves corrective training examples from `master_copy_dedup_v2.csv`.

The full augmentation process is:

```
Error diagnostic → data_augment.py → augmentation_candidates.csv → augmented_with_tuning
```

We perform 3 rounds of this data augmentation process.

### Error-Driven Augmentation Pipeline (`data_augment.py`)

The script takes the diagnosed errors from `error_root_causes.csv` (the held-out validation set, never seen during training) and uses the fine-tuned BERT checkpoint as a semantic embedder to find relevant counter-examples in the master pool.

**The 4 pipeline steps:**

| Step | What happens |
|---|---|
| 1. Load & deduplicate | Loads `master_copy_dedup_v2.csv` as the candidate pool; excludes all headlines in train/val/test splits and in the error file itself to prevent leakage |
| 2. Embed | Encodes both error headlines and the master pool using mean-pooled, L2-normalised BERT embeddings from the `improved_with_tuning` checkpoint |
| 2.5 Cluster | Groups error embeddings into K clusters (K auto-tuned via silhouette score); auto-names each cluster with TF-IDF keywords |
| 3. Select | For each cluster, queries the label-specific KNN pool (false positives → non-sarcastic pool; false negatives → sarcastic pool) with angle-specific distance thresholds and criticality caps |
| 4. Dedup & export | Cross-cluster deduplication; writes `data/augmentation_output/baseline_roberta/round_1/augmentation_candidates.csv` ready to concat with `train.csv` | (Round 2 and Round 3 are in the respective subfolders)

**Diagnostic angles and how they are handled:**

| Angle | Failure mode | Strategy | Distance threshold |
|---|---|---|---|
| Angle 2 | Outlier / zero-shot zone | Wide net to fill coverage void | 0.30 |
| Angle 3 | Systematic bias / spurious rule | Tight counter-examples to break the pattern | 0.20 |
| Angle 4 | Low-signal / too short | Moderate radius, min 6-word requirement | 0.25 |
| Angle 5 | Punctuation spurious correlation | Counter-examples at tight radius | 0.20 |
| Angle 6 | Mixed neighbourhood | Very tight to sharpen decision boundary | 0.15 |
| Angle 1 | Label conflict | **Skipped** — needs relabelling, not more data | — |
| Uncategorised | — | **Skipped** | — |

## Run Order

**1. Running the RoBERTa variants.**

Each of the `train.py` variants handle all the preprocessing and loading of RoBERTa variants in the file. Simply changing the OUTPUT_DIR location will suffice.

**2. Running data augmentation pipeline.**

- First, `original_with_tuning` has to be trained.
- Run `predict.py` with `original_with_tuning` with the desired dataset. We will stick to diagnostic_val.csv for this demo.
    - `original_with_tuning_diagnostic_val_predictions.csv` should be created
- Run the error_diagnostic script `error_diagnostic.py` using:
```bash
python error_diagnostic.py --model baseline_roberta --mode original_with_tuning --dataset diagnostic_val
```
    - `error_diagnostic.csv` should be created under `data/augmentation_output/baseline_roberta` (from project root).
- Run `data_augment.py` under `models/baseline_roberta` to get `augmentation_candidates.csv` under `data/augmentation_output/baseline_roberta`
- Next, run the script `append_augmented_data.py` from `models/baseline_roberta` to append the new candidates to `train.csv`
- We can now run `trainer_augmented.py` and `evaluation.py` to get our results.

---

## Outputs

All artifacts are written under `models/baseline_roberta/outputs/`:

```text
outputs/
├── metrics/
│   ├── {mode}_{dataset}_metrics.json
├── predictions/
│   └── {mode}_{dataset}_predictions.csv
```

Augmentation pipeline outputs are written to `augmentation_output/` in the project root:

```text
augmentation_output/
├──baseline_roberta
    ├── round_1
        ├── augmentation_candidates.csv   ← training-ready, matches train.csv schema
        └── augmentation_report.csv       ← source error, angle, distance, cluster metadata
    ├── round_2
        ├── augmentation_candidates.csv 
        └── augmentation_report.csv       
    └── round_3
        ├── augmentation_candidates.csv   
        └── augmentation_report.csv     
```

## Results Summary

Performance of all trained modes across evaluation datasets:

| Mode | original_test F1 | diagnostic_val F1 | master_copy F1 |
|---|---|---|---|
| `pretrained` | 0.0000 | 0.0000 | 0.0000 |
| `original_no_tuning` | 0.9620 | 0.7360 | 0.8632 |
| `original_with_tuning` | **0.9893** | 0.7484 | **0.8771** |
| `improved_no_tuning` | 0.9326 | 0.7188 | 0.8282 |
| `improved_with_tuning` | 0.9537 | 0.7329 | 0.8440 |
| `augmented_with_tuning_rd1` | 0.9663 | **0.7511** | 0.8679 |
| `augmented_with_tuning_rd2` | 0.9298 | 0.6539 | 0.8111 |
| `augmented_with_tuning_rd1` | 0.9026 | 0.6543 | 0.8213 |

`diagnostic_val` is a 1,825-headline held-out set (HuffPost + The Onion) never seen during training, validation, or hyperparameter tuning.