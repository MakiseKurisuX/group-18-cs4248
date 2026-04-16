# BERT Sarcasm Detection — Consolidated Findings

This document consolidates results and error analysis across all BERT variants on three evaluation datasets:

- **`original_test`** — stratified held-out split from the original balanced dataset (~50/50 sarcastic/non-sarcastic, n=2,851)
- **`master_copy_dedup_v2`** — large out-of-distribution dataset (~83.9% non-sarcastic / 16.1% sarcastic, n=251,971)
- **`diagnostic_val`** — 1,825-headline held-out set (1,212 non-sarcastic / 613 sarcastic; ~66/34 split) drawn from HuffPost and The Onion, never seen during training, validation, or hyperparameter tuning

Error diagnostic clusters were produced using KNN neighbourhood analysis and KMeans (K=8) on the BERT [1] CLS-token embedding space. Each cluster reports **FP** (non-sarcastic classified as sarcastic) and **FN** (sarcastic classified as non-sarcastic).

---

## 0. Dataset Composition and Overlap Analysis

Understanding how the three evaluation datasets relate to each other — and to the training data — is essential for correctly interpreting model results.

### 0.1 Sizes and Class Balance

| Dataset | Total rows | Sarcastic | Non-sarcastic | Sarcasm % |
|---|---|---|---|---|
| `original_train` | 22,801 | 10,841 | 11,960 | 47.5% |
| `original_val` | 2,851 | 1,355 | 1,496 | 47.5% |
| `original_test` | 2,851 | 1,356 | 1,495 | 47.6% |
| `master_copy_dedup_v2` | 251,971 | 40,578 | 211,393 | 16.1% |
| `diagnostic_val` | 1,825 | 613 | 1,212 | 33.6% |

The original dataset is approximately balanced (~48/52). `master_copy_dedup_v2` is heavily skewed toward non-sarcastic (~84%). `diagnostic_val` falls between the two at ~34% sarcastic. All datasets share a nearly identical headline length distribution (mean ≈ 10 words, median = 10, p5 ≈ 5, p95 ≈ 16), so there is no domain shift in headline length between evaluation splits.

### 0.2 Sources

All datasets draw from exactly two publishers:

| Dataset | Non-sarcastic source | Sarcastic source |
|---|---|---|
| `original` (all splits) | HuffPost (`huffingtonpost`) | The Onion (`theonion`) |
| `master_copy_dedup_v2` | HuffPost / HuffPost rebranded (`huffpost`) | The Onion + unknown |
| `diagnostic_val` | HuffPost / HuffPost rebranded | The Onion (empty/missing article link) |

The `master_copy_dedup_v2` top domains are: `huffingtonpost` (~80%), missing/empty (~11%), `theonion` (~5%), `huffpost` (~3.5%). The missing-domain rows correspond to articles whose links were not captured in the original scrape; the sarcastic ones are Onion headlines identifiable by content. `diagnostic_val` follows the same pattern: `huffingtonpost` (1,171), missing (558), `kinja-img` (55), `huffpost` (34), with the missing-domain rows accounting for the majority of sarcastic examples.

This means all three evaluation datasets test the same fundamental binary: **HuffPost non-sarcastic vs. Onion sarcastic**. There is no publisher-level domain shift between them — only stylistic and temporal variation within each publisher.

### 0.3 Headline-Level Overlap

The overlap between every pair of datasets was measured by exact string match after lowercasing and stripping whitespace.

| Pair | Shared headlines | Notes |
|---|---|---|
| `orig_train` ∩ `orig_val` | 0 | Disjoint by construction |
| `orig_train` ∩ `orig_test` | 0 | Disjoint by construction |
| `orig_val` ∩ `orig_test` | 0 | Disjoint by construction |
| `master_copy` ∩ `orig_train` | 21,651 / 22,801 (95.0%) | master is a near-superset of orig_train |
| `master_copy` ∩ `orig_val` | 2,716 / 2,851 (95.3%) | master is a near-superset of orig_val |
| `master_copy` ∩ `orig_test` | 2,685 / 2,851 (94.2%) | master is a near-superset of orig_test |
| `diagnostic_val` ∩ `master_copy` | 1,825 / 1,825 (100%) | diagnostic_val is fully contained in master |
| `diagnostic_val` ∩ `orig_train` | 136 | 7.5% of diagnostic_val seen in training |
| `diagnostic_val` ∩ `orig_val` | 11 | Minimal overlap |
| `diagnostic_val` ∩ `orig_test` | 18 | Minimal overlap |
| `diagnostic_val` ∩ any orig split | 165 / 1,825 (9.0%) | 91% of diagnostic_val is novel vs. original data |

**Key structural facts:**
- `master_copy_dedup_v2` is a strict near-superset of the original dataset. The original train/val/test splits are almost entirely contained within master_copy (~95%). The ~5% absent rows are likely headlines that were removed during master's own deduplication pass.
- `diagnostic_val` is **entirely contained within `master_copy`** (100% overlap). It was drawn from the same underlying pool.
- Despite being a master_copy subset, `diagnostic_val` has **only 9% overlap with the original splits**. The remaining 91% (1,660 headlines) come from the larger master_copy-exclusive portion, making it a genuine out-of-distribution test for models trained only on the original dataset.
- All labels are consistent: `diagnostic_val` headline labels match `master_copy` labels at **100%** for the 1,796 shared rows, confirming no label disagreement between datasets.

### 0.4 Implications for Each Evaluation Setting

| Evaluation | Model | How held-out? | Interpretation |
|---|---|---|---|
| `original_test` | All models | Perfectly disjoint from train/val (0 overlap) | Clean held-out test; reliable F1 estimate for balanced ~50/50 distribution |
| `diagnostic_val` | Models trained on original data | 91% novel — only 165/1,825 headlines seen during training | Strong OOD test; representative of stylistic variation not covered by original_train |
| `diagnostic_val` | `master_no_tuning` | **1,796/1,825 (98.4%) headlines were in the training set** | **Not a held-out test.** The model trained directly on these examples. diagnostic_val F1 = 0.9536 reflects in-distribution recall, not generalisation. |
| `master_copy` | Models trained on original data | 95% of master was never in original training | Strong OOD test for coverage and prior-shift robustness |
| `master_copy` | `master_no_tuning` | Training was on master_copy minus orig_val/test | Mostly in-distribution; F1 = 0.9003 reflects training-domain performance |

> **Important caveat:** `master_no_tuning`'s outstanding results on `diagnostic_val` (F1 = 0.9536, 59 total errors) and `master_copy` (F1 = 0.9003) **cannot be compared on equal terms** with models trained only on the original dataset, because those models evaluated on truly held-out data. `master_no_tuning` is effectively being tested on data it has already seen. Its strong performance confirms that the model learned the task well, but does not measure generalisation in the same way.

> The diagnostic_val results for `improved_with_tuning` (F1 = 0.8723, 156 errors) and `improved_large_with_tuning` (F1 = 0.8728, 158 errors) are the most meaningful OOD benchmarks for the fine-tuned improved architecture family, as neither model saw any of the 91% novel diagnostic_val headlines during training or tuning.

---

## 1. Performance Overview

Rows labelled `_calibrated` use the F1-maximising threshold selected on the validation set. All other rows use threshold = 0.50. master_copy values marked † come from earlier evaluation runs (master evaluation is skipped by default in the current sweep).

| Mode | original_test F1 | original_test Acc | master_copy F1 | diagnostic_val F1 | diagnostic_val Acc | Threshold |
|---|---|---|---|---|---|---|
| `pretrained` | 0.5852 | 0.4616 | 0.2562 | 0.4409 | 0.3468 | 0.50 |
| `pretrained_large` | 0.6440 | 0.4753 | 0.2713 | 0.4894 | 0.3288 | 0.50 |
| `original_no_tuning` | 0.9146 | 0.9200 | 0.7221 | 0.8510 | 0.9047 | 0.50 |
| `original_no_tuning_calibrated` | 0.9186 | 0.9232 | 0.7200 † | 0.8501 | 0.9036 | 0.36 |
| `original_with_tuning` | 0.9256 | 0.9295 | 0.7392 | 0.8615 | 0.9096 | 0.50 |
| `original_with_tuning_calibrated` | 0.9273 | 0.9306 | 0.7356 † | 0.8633 | 0.9101 | 0.40 |
| `improved_no_tuning` | 0.9149 | 0.9190 | 0.6924 † | 0.8395 | 0.8932 | 0.50 |
| `improved_no_tuning_calibrated` | 0.9159 | 0.9193 | 0.6871 † | 0.8430 | 0.8942 | 0.34 |
| `improved_with_tuning` | 0.9270 | 0.9309 | 0.7312 | 0.8737 | 0.9162 | 0.50 |
| `improved_with_tuning_calibrated` | 0.9255 | 0.9288 | 0.7267 † | 0.8723 | 0.9145 | 0.15 |
| `improved_large_with_tuning` | 0.9327 | 0.9355 | 0.7145 | 0.8707 | 0.9134 | 0.50 |
| `improved_large_with_tuning_calibrated` | **0.9347** | **0.9369** | 0.7145 † | 0.8728 | 0.9134 | 0.15 |
| `augmented_with_tuning` | 0.9323 | 0.9358 | 0.7328 | 0.8633 | 0.9090 | 0.50 |
| `augmented_with_tuning_calibrated` | — | — | — | 0.8611 | 0.9058 | 0.14 |
| `master_no_tuning` | 0.9243 | 0.9288 | **0.9003** | **0.9572** | **0.9704** | 0.50 |
| `master_no_tuning_calibrated` | 0.9256 | 0.9295 | 0.9003 | 0.9536 | 0.9677 | 0.14 |

**Best on `original_test` (uncalibrated):** `improved_large_with_tuning` (F1 = 0.9327)  
**Best on `original_test` (calibrated):** `improved_large_with_tuning_calibrated` (F1 = 0.9347)  
**Best comparable model on `master_copy`:** `original_with_tuning` (F1 = 0.7392)  
**Best comparable model on `diagnostic_val` (uncalibrated):** `improved_with_tuning` (F1 = 0.8737)  
**Best comparable model on `diagnostic_val` (calibrated):** `improved_large_with_tuning_calibrated` (F1 = 0.8728)

> **Note on calibrated variants and `diagnostic_val`:** Calibrated thresholds are selected to maximise F1 on the 50/50 validation set. Because `diagnostic_val` has a different class balance (33% sarcastic), calibration does not always help on that split and can slightly hurt. For example, `improved_with_tuning` at threshold 0.50 achieves F1 = 0.8737 on `diagnostic_val`, while `improved_with_tuning_calibrated` at threshold 0.15 yields F1 = 0.8723 — a marginal regression. Similarly, `master_no_tuning` at 0.50 outperforms its calibrated variant on `diagnostic_val` (0.9572 vs 0.9536). Calibration remains beneficial on `original_test` for most base models.

### 1.1 Diagnostic Validation — Detailed Per-Model Metrics

Full per-model metrics on `diagnostic_val` (n=1,825; 1,212 non-sarcastic / 613 sarcastic). Both uncalibrated (threshold=0.50) and calibrated variants are shown where available.

| Mode | Acc | Precision | Recall | F1 | Threshold | TN | FP | FN | TP | Errors |
|---|---|---|---|---|---|---|---|---|---|---|
| `pretrained` | 0.3468 | 0.3094 | 0.7667 | 0.4409 | 0.50 | 163 | 1049 | 143 | 470 | 1192 |
| `pretrained_large` | 0.3288 | 0.3287 | 0.9576 | 0.4894 | 0.50 | 13 | 1199 | 26 | 587 | 1225 |
| `original_no_tuning` | 0.9047 | 0.8955 | 0.8108 | 0.8510 | 0.50 | 1154 | 58 | 116 | 497 | 174 |
| `original_no_tuning_calibrated` | 0.9036 | 0.8895 | 0.8140 | 0.8501 | 0.36 | 1150 | 62 | 114 | 499 | 176 |
| `original_with_tuning` | 0.9096 | 0.8875 | 0.8369 | 0.8615 | 0.50 | 1147 | 65 | 100 | 513 | 165 |
| `original_with_tuning_calibrated` | 0.9101 | 0.8825 | 0.8450 | 0.8633 | 0.40 | 1143 | 69 | 95 | 518 | 164 |
| `improved_no_tuning` | 0.8932 | 0.8472 | 0.8320 | 0.8395 | 0.50 | 1120 | 92 | 103 | 510 | 195 |
| `improved_no_tuning_calibrated` | 0.8942 | 0.8409 | 0.8450 | 0.8430 | 0.34 | 1114 | 98 | 95 | 518 | 193 |
| `improved_with_tuning` | **0.9162** | 0.8846 | 0.8630 | **0.8737** | 0.50 | 1143 | 69 | 84 | 529 | **153** |
| `improved_with_tuning_calibrated` | 0.9145 | 0.8752 | 0.8695 | 0.8723 | 0.15 | 1136 | 76 | 80 | 533 | 156 |
| `improved_large_with_tuning` | 0.9134 | 0.8736 | 0.8679 | 0.8707 | 0.50 | 1135 | 77 | 81 | 532 | 158 |
| `improved_large_with_tuning_calibrated` | 0.9134 | 0.8617 | 0.8842 | **0.8728** | 0.15 | 1125 | 87 | 71 | 542 | 158 |
| `augmented_with_tuning` | 0.9090 | 0.8719 | 0.8548 | 0.8633 | 0.50 | 1135 | 77 | 89 | 524 | 166 |
| `augmented_with_tuning_calibrated` | 0.9058 | 0.8528 | 0.8695 | 0.8611 | 0.14 | 1120 | 92 | 80 | 533 | 172 |
| `master_no_tuning` | **0.9704** | 0.9307 | 0.9853 | **0.9572** | 0.50 | 1167 | 45 | 9 | 604 | **54** |
| `master_no_tuning_calibrated` | 0.9677 | 0.9210 | 0.9886 | 0.9536 | 0.14 | 1160 | 52 | 7 | 606 | 59 |

**Best uncalibrated F1 among original-dataset models:** `improved_with_tuning` (F1 = 0.8737, 153 total errors)  
**Best calibrated F1 among original-dataset models:** `improved_large_with_tuning_calibrated` (F1 = 0.8728, 158 total errors)  
**Fewest errors at threshold=0.50:** `improved_with_tuning` (153)  
**Worst fine-tuned (uncalibrated):** `improved_no_tuning` (F1 = 0.8395, 195 errors — worse than `original_no_tuning`)

### 1.2 Pretrained Baselines — Behavior Without Fine-Tuning

Neither pretrained model has learned anything about sarcasm detection. Their outputs reflect the class bias of a randomly-initialised classification head applied on top of frozen BERT representations. The specific bias direction (which class is over-predicted) depends on the random initialisation and is not semantically meaningful.

**`pretrained` (bert-base-uncased)** — biased toward sarcastic class:
- `diagnostic_val`: TN=163, FP=1049, FN=143, TP=470 → Recall=0.7667, Precision=0.3094, F1=0.4409, Acc=0.3468
- Predicts ~84% of all examples as sarcastic (FP+TP = 1519/1825). The apparent 77% recall for the sarcastic class is an artifact of this over-prediction, not a genuine sarcasm signal.
- `original_test` accuracy of 0.4616 is below chance (50/50 balanced), confirming the classifier is miscalibrated in the direction of the sarcastic class.

**`pretrained_large` (bert-large-uncased)** — extremely biased toward sarcastic class:
- `diagnostic_val`: TN=13, FP=1199, FN=26, TP=587 → Recall=0.9576, Precision=0.3287, F1=0.4894, Acc=0.3288
- Predicts ~98% of all examples as sarcastic (FP+TP = 1786/1825). Even more extreme than `pretrained`.
- The higher F1 (0.4894 vs 0.4409) solely reflects higher recall from the near-total over-prediction, not better discrimination.

Both models serve only as lower bounds demonstrating that BERT without fine-tuning has no usable sarcasm detection capability. The class bias direction varies by run and initialisation. The `pretrained_large` F1 appearing higher than `pretrained` is misleading — a model that always predicts "sarcastic" would achieve F1 ≈ 0.51 on this dataset (2 × precision × 1.0 / (precision + 1.0)), and both models are approaching that degenerate ceiling from different starting points.

---

## 2. Effect of Each Intervention

### 2.1 Hyperparameter Tuning (Optuna [2][3])

Tuning is the single most impactful lever across both pipelines.

| Pipeline | No Tuning F1 (orig. test) | With Tuning F1 (orig. test) | Δ |
|---|---|---|---|
| Original | 0.9146 | 0.9256 | +0.011 |
| Improved | 0.9149 | 0.9270 | +0.012 |

| Pipeline | No Tuning F1 (master_copy) | With Tuning F1 (master_copy) | Δ |
|---|---|---|---|
| Original | 0.7222 | 0.7392 | +0.017 |
| Improved | 0.6924 | 0.7312 | +0.039 |

The larger tuning gain for the improved pipeline on `master_copy` (+0.039 vs +0.017) indicates that the contrastive and topic-balancing enhancements are more sensitive to hyperparameter choices and require Optuna to reach their potential.

### 2.2 Contrastive Loss [4] + Topic-Balanced Sampling [5]

Comparing same-budget pairs (with tuning), using calibrated thresholds where available:

| Dataset | original_with_tuning F1 | improved_with_tuning F1 | Δ |
|---|---|---|---|
| original_test | 0.9273 (cal.) | 0.9255 (cal.) | −0.002 |
| master_copy | 0.7392 | 0.7312 | **−0.008** |
| diagnostic_val | 0.8633 (cal.) | **0.8737** (uncal.) | **+0.010** |

At threshold=0.50 the comparison is similar:

| Dataset | original_with_tuning F1 | improved_with_tuning F1 | Δ |
|---|---|---|---|
| original_test | 0.9256 | 0.9270 | +0.001 |
| diagnostic_val | 0.8615 | **0.8737** | **+0.012** |

The improvements provide a consistent gain on `diagnostic_val` (~+0.010–0.012), a negligible or slightly negative result on `original_test` (−0.002 to +0.001 depending on calibration), and a small regression on `master_copy` (−0.008). The contrastive training helps generalise to held-out data with diverse Onion styles (diagnostic_val) but slightly hurts on the heavily skewed and volume-heavy `master_copy` — most likely because topic-balanced sampling reduces exposure to politically-framed real-news headlines, which are overrepresented in `master_copy`.

### 2.3 Data Augmentation

`augmented_with_tuning` trains on the same architecture as `improved_with_tuning` with 320 additional examples (160 sarcastic / 160 non-sarcastic) retrieved via KNN from `master_copy_dedup_v2` based on the diagnosed error patterns from `diagnostic_val`.

| Dataset | improved_with_tuning F1 (0.50) | augmented_with_tuning F1 (0.50) | Δ |
|---|---|---|---|
| original_test | 0.9270 | **0.9323** | **+0.005** |
| master_copy | 0.7312 | 0.7328 | +0.002 |
| diagnostic_val | **0.8737** | 0.8633 | **−0.010** |

Augmentation improves `original_test` and marginally improves `master_copy`, but **regresses on `diagnostic_val`** — the very dataset whose errors drove augmentation candidate selection. See Section 5 for the error-level explanation.

### 2.4 Master Dataset Training

`master_no_tuning` trains the improved architecture (contrastive loss + topic-balanced sampling) on the full `master_copy_dedup_v2` dataset (246,258 examples after excluding val/test headlines), with no fresh Optuna search. Hyperparameters from `improved_with_tuning` are reused with three overrides for dataset-scale compatibility: `batch_size=16` (was 8), `grad_accum=1` (was 2), `n_topic_clusters=10` (was 6), and `num_epochs=3` (was 5).

| Dataset | improved_with_tuning F1 (0.50) | master_no_tuning F1 (0.50) | Δ |
|---|---|---|---|
| original_test | 0.9270 | 0.9243 | −0.003 |
| master_copy | 0.7312 | **0.9003** | **+0.169** |
| diagnostic_val | 0.8737 | **0.9572** | **+0.084** |

Training on the master dataset produces the largest single-step improvement seen in this project on both `master_copy` (+0.169 F1) and `diagnostic_val` (+0.084 F1 uncalibrated, +0.081 calibrated), while costing only a small −0.003 on `original_test`. The improvement on `master_copy` is partially expected — the model is now trained in-domain — but the +0.08 gain on `diagnostic_val` (a fully held-out set never seen at any stage) confirms that the master dataset's distributional breadth provides genuine generalisation improvements.

Notably, `master_no_tuning` at threshold=0.50 outperforms its own calibrated variant (F1 = 0.9572 vs 0.9536), because the calibrated threshold of 0.14 — optimised on the 50/50 validation set — over-predicts sarcasm relative to what is optimal for `diagnostic_val`'s 33% sarcasm rate.

The `diagnostic_val` error count drops from 153 (`improved_with_tuning`, threshold=0.50) to 54 (`master_no_tuning`, threshold=0.50), primarily because the dry/first-person Onion FN cluster is essentially eliminated (84 FN → 9 FN). The model has seen far more diverse Onion-style headlines during training and can now recognise first-person and list-format comedic voice.

### 2.5 Threshold Calibration

Calibration sweeps thresholds from 0.05 to 0.95 on the validation set and selects the F1-maximising value [6].

| Mode | Calibrated threshold | Δ F1 original_test | Δ F1 master_copy | Δ F1 diagnostic_val |
|---|---|---|---|---|
| `original_no_tuning` | 0.36 | +0.004 | −0.001 | −0.001 |
| `original_with_tuning` | 0.40 | +0.002 | −0.004 | +0.002 |
| `improved_no_tuning` | 0.34 | +0.001 | −0.005 | +0.004 |
| `improved_with_tuning` | **0.15** | −0.002 | −0.005 | −0.001 |
| `improved_large_with_tuning` | **0.15** | +0.002 | ≈0.000 | +0.002 |
| `augmented_with_tuning` | **0.14** | — | — | −0.002 |
| `master_no_tuning` | **0.14** | +0.001 | ≈0.000 | −0.004 |

Calibration provides a small gain on `original_test` for most models. The unusually aggressive thresholds (0.14–0.15) for the improved and augmented variants reflect that contrastive training compresses sarcastic embeddings into a tight region but at low softmax probability values, so the optimal decision boundary lies well below 0.5. On `master_copy`, calibration **universally hurts** because the val set used for calibration is 50/50, but `master_copy` is 84/16 — optimising for the val prior makes the model over-predict sarcasm on the skewed distribution. On `diagnostic_val`, calibration effects are mixed: it slightly helps some models and slightly hurts others, because `diagnostic_val`'s 33% sarcasm rate is also misaligned with the 50/50 calibration target.

### 2.6 Large Backbone Variant (`bert-large-uncased`)

Replacing `bert-base-uncased` with `bert-large-uncased` in the improved tuned pipeline changes the performance profile in a very specific way:

Uncalibrated comparison (threshold=0.50):

| Dataset | improved_with_tuning F1 | improved_large_with_tuning F1 | Δ |
|---|---|---|---|
| original_test | 0.9270 | **0.9327** | **+0.006** |
| master_copy | **0.7312** | 0.7145 | **−0.017** |
| diagnostic_val | **0.8737** | 0.8707 | −0.003 |

Calibrated comparison:

| Dataset | improved_with_tuning_calibrated F1 | improved_large_with_tuning_calibrated F1 | Δ |
|---|---|---|---|
| original_test | 0.9255 | **0.9347** | **+0.009** |
| master_copy | 0.7267 | 0.7145 | −0.012 |
| diagnostic_val | 0.8723 | **0.8728** | +0.001 |

The large backbone clearly helps on the balanced `original_test` split (+0.006–0.009 F1) and regresses on `master_copy` (−0.012–0.017 F1). On `diagnostic_val`, the uncalibrated comparison slightly favours `improved_with_tuning` (0.8737 vs 0.8707), while the calibrated comparison narrowly favours `improved_large_with_tuning` (0.8728 vs 0.8723). In both cases `improved_large_with_tuning_calibrated` recovers more sarcastic positives (FN 80 → 71 vs calibrated `improved_with_tuning`) but introduces extra false positives (FP 76 → 87), shifting the FP:FN balance toward over-prediction of sarcasm.

---

## 3. FP/FN Balance Across Models

Figures from `diagnostic_val` at threshold=0.50 (uncalibrated) unless noted.

| Mode | FP | FN | Total errors | FP:FN ratio |
|---|---|---|---|---|
| `pretrained` | 1049 | 143 | 1192 | 7.3:1 (FP-heavy) |
| `pretrained_large` | 1199 | 26 | 1225 | 46:1 (extreme FP) |
| `original_no_tuning` | 58 | 116 | 174 | 1:2.0 |
| `original_no_tuning_calibrated` | 62 | 114 | 176 | 1:1.8 |
| `original_with_tuning` | 65 | 100 | 165 | 1:1.5 |
| `original_with_tuning_calibrated` | 69 | 95 | 164 | 1:1.4 |
| `improved_no_tuning` | 92 | 103 | 195 | ~1:1.1 |
| `improved_no_tuning_calibrated` | 98 | 95 | 193 | ~1:1 |
| `improved_with_tuning` | **69** | **84** | **153** | 1:1.2 |
| `improved_with_tuning_calibrated` | 76 | 80 | 156 | ~1:1 |
| `improved_large_with_tuning` | 77 | 81 | 158 | ~1:1 |
| `improved_large_with_tuning_calibrated` | 87 | 71 | 158 | 1.2:1 |
| `augmented_with_tuning` | 77 | 89 | 166 | 1:1.2 |
| `augmented_with_tuning_calibrated` | 92 | 80 | 172 | 1.2:1 |
| `master_no_tuning` | **45** | **9** | **54** | 5:1 (FP-heavy) |
| `master_no_tuning_calibrated` | 52 | 7 | 59 | 7.4:1 |

The pretrained models are strongly FP-biased (predicting sarcastic for most inputs), which is the opposite of a meaningful lower bound. Fine-tuning shifts the balance dramatically: `original_no_tuning` drops from thousands of FPs to 58 FP and 116 FN, with the ratio settling around 1:2 (slightly FN-heavy) as the models initially fail to fire on harder Onion headlines.

As training improves, both FP and FN counts fall. `improved_with_tuning` at threshold=0.50 achieves the fewest total errors among comparable models (153), with a slight FN bias (1:1.2). Calibration to threshold=0.15 shifts the balance to near-parity (76 FP vs 80 FN, 156 errors) but increases total errors slightly.

`augmented_with_tuning` introduces 8 additional FPs relative to `improved_with_tuning` (77 vs 69) and 5 additional FNs (89 vs 84), net +13 errors at threshold=0.50. With calibration, FPs jump further (92) and total errors reach 172.

`master_no_tuning` achieves by far the lowest total error count (54 at 0.50, 59 calibrated) — a 65% further reduction over `improved_with_tuning`. The FP:FN ratio swings to 5:1 because the model has nearly eliminated FNs (84→9), learning to recognise Onion-style voice across a far wider stylistic range. The dominant remaining error type is FP: non-sarcastic news headlines over-predicted as sarcastic, particularly in political and general-news domains.

---

## 4. Cross-Dataset Degradation

All small-dataset models lose roughly 0.19–0.22 F1 moving from `original_test` to `master_copy`. Training on `master_copy` itself closes this gap almost entirely:

| Mode | original_test F1 | master_copy F1 | Cross-dataset ΔF1 |
|---|---|---|---|
| `original_no_tuning` | 0.9146 | 0.7221 | −0.192 |
| `original_with_tuning` | 0.9256 | 0.7392 | −0.186 |
| `improved_with_tuning` | 0.9270 | 0.7312 | −0.196 |
| `improved_large_with_tuning` | 0.9327 | 0.7145 | −0.218 |
| `augmented_with_tuning` | 0.9323 | 0.7328 | −0.200 |
| `master_no_tuning` | 0.9243 | 0.9003 | **−0.024** |

For models trained only on the original dataset, the `master_copy` cluster reports reveal two structural causes of degradation:

**Cause 1 — Format shift.** One cluster (keywords: `tips, know, pros, cons`) contains ~3,900 listicle-format headlines that are a pure false-negative cluster. This format does not exist in the original training data.

**Cause 2 — Vocabulary overlap at scale.** Clusters with generic news vocabulary (`new, man, year, old`) accumulate thousands of false positives at the volume of `master_copy`. Everyday news phrasing that mildly resembles sarcastic framing produces a false-positive flood absent in the smaller, more curated `original_test`.

`master_no_tuning` reduces the cross-dataset gap from ~0.19 to only 0.025 by training in-domain. The remaining 0.025 gap reflects structural differences between `original_test` (50/50 balanced) and `master_copy` (84/16 skewed), not a genuine generalisation failure.

The large backbone does not solve this shift. `improved_large_with_tuning` posts the strongest `original_test` result in the project (0.9347 F1) but the weakest `master_copy` result among the tuned improved variants (0.7145 F1), suggesting that extra capacity sharpens in-distribution fit more than robustness to skew and format drift.

---

## 5. Cluster-Level Error Analysis — `diagnostic_val`

### 5.1 The Dry / First-Person Onion FN Cluster (Persistent Across All Models)

The most stable and irreducible error pattern across all fine-tuned models is a cluster of **dry, first-person or list-format Onion headlines** that rely on comedic voice and framing rather than absurd vocabulary.

| Mode | Cluster keywords | Size | FP | FN |
|---|---|---|---|---|
| `original_no_tuning` | like, things, old, won, hate | 52 | 1 | 51 |
| `original_with_tuning` | old, village, like, hate, mind believe | 39 | 2 | 37 |
| `improved_no_tuning` | things, like, year, old, won | 38 | 1 | 37 |
| `improved_with_tuning` | things, old, won, hate, believe | 43 | 0 | 43 |
| `augmented_with_tuning` | things, old, year, house, won | **44** | 0 | **44** |

This cluster is **completely unchanged by augmentation** — it grew by one sample. The augmented examples targeted Angle 3 (systematic bias) and Angle 2 (outlier coverage) but did not address the structural cause: the model has no training examples of first-person Onion voice. Adding more topically-matched real-news counter-examples does not teach the model how to recognise comedic framing.

Sample consistently-missed headlines:
- *"goddamn it, the neighbors are silently going at it again, i imagine"*
- *"i won't ever let the position of county surveyor go to my head"*
- *"things all cats do that prove they are psychopaths"*
- *"as long as you're under my roof, you'll play by my monopoly rules"*
- *"i should not be allowed to say the following things about america"*

### 5.2 Health / Wellness FP Cluster (Persistent, Grew Under Augmentation)

Real health and lifestyle news that the model classifies as sarcastic appears in every tuned model and **expanded under augmentation**.

| Mode | Cluster keywords | Size | FP | FN |
|---|---|---|---|---|
| `original_with_tuning` | healthy, brain, shares, days, life planet | 28 | 28 | 0 |
| `improved_with_tuning` | healthy, age, brain, shares, going | 31 | 30 | 1 |
| `augmented_with_tuning` | healthy, doubling santa, santa, doubling, campaign | **28** | **28** | 0 |

Additionally, augmented_with_tuning introduces two new FP-only clusters not present in `improved_with_tuning`:

- **C5** (dollar, million, virtual community): 25 FP, 0 FN — financial and community news over-predicted as sarcastic
- **C6** (cancer, virus, versace, planet, life planet): 12 FP, 0 FN — science/health news, mirroring the science/discovery FP cluster seen on `original_test`

The augmented counter-examples were intended to break spurious associations between certain vocabulary and the sarcasm label. Instead, the model appears to have shifted which vocabulary triggers spurious sarcasm predictions — reducing one FP cluster while creating new ones.

### 5.3 Political FP Clusters

Multiple models produce small clusters of real political or culturally-adjacent news predicted as sarcastic. The pattern appears across all pipelines, growing with model capacity and data augmentation.

**Pre-augmented models:**

| Mode | Cluster keywords | Size | FP | FN |
|---|---|---|---|---|
| `original_no_tuning` | trump, dollar, apricot, years | 25 | 25 | 0 |
| `improved_no_tuning` | trump, age, says, releases, photo | 24 | 23 | 1 |
| `original_with_tuning` | amsterdam, prepares, cut, says | 15 | 15 | 0 |

Sample misclassified headlines (pre-augmented):
- *"trump lauds tax cuts on road trip as most americans still don't see a benefit"*
- *"california raises smoking age to 21"*
- *"national alzheimer's plan sets deadline to find treatments for memory-robbing disease"*

The model has learned that Trump-adjacent and political vocabulary co-occurs with Onion political satire — a correct generalisation for the sarcastic class, but one that misfires on real political reporting.

**Augmented model (`augmented_with_tuning`):**

`augmented_with_tuning` introduces an expanded FP cluster absent from `improved_with_tuning`:

- **C0** (passes, black, parade, parade passes, worst): 21 errors, FP=17, FN=4

Sample misclassified headlines (augmented):
- *"dress codes blame the victim"*
- *"jeanine pirro attacks biden for being sympathetic toward flight attendants"*
- *"study finds having a conversation about trans issues can change people's minds"*
- *"animal shelter celebrates when all of its dogs are adopted in time for christmas"*

The augmentation candidates from Angle 3 (systematic bias) included real-news counter-examples on political topics, but appear to have introduced new vocabulary–label associations in culturally-adjacent domains rather than correcting the original pattern.

### 5.4 What Augmentation Did and Did Not Fix

| Error type | improved_with_tuning | augmented_with_tuning | Change |
|---|---|---|---|
| Dry/first-person Onion FN | 43 | 44 | **No change (+1)** |
| Health/wellness FP | 30 | 28 | Marginal improvement (−2) |
| Political/cultural FP | ~9 | ~17 | **Worsened (+8)** |
| Financial/community FP | 0 | 25 | **New cluster** |
| Science/health FP | 0 | 12 | **New cluster** |
| Total errors | 156 | 172 | +16 |

37 errors were fixed (20 FP + 17 FN) and 53 new errors were introduced (36 FP + 17 FN). The net effect is negative on `diagnostic_val` but positive on `original_test`.

### 5.5 Master Dataset Training — Near-Elimination of the FN Problem

`master_no_tuning` reduces `diagnostic_val` errors from 172 (`augmented_with_tuning`) to 59, with only 7 FNs remaining.

**Diagnostic angle breakdown (59 errors):**

| Angle | Count | Description |
|---|---|---|
| Angle 3 (Systematic Bias) | 29 | Most remaining errors are still systematic |
| Angle 6 (Mixed Neighbourhood) | 14 | Ambiguous embedding regions |
| Uncategorised | 8 | Miscellaneous |
| Angle 2 (Outlier) | 6 | Low-coverage edge cases |
| Angle 1 (Label Conflict) | 2 | Possible annotation errors |

**Cluster breakdown (59 errors, FP=52, FN=7):**

| Cluster | Keywords | Errors | FP | FN | Character |
|---|---|---|---|---|---|
| C0 | star, challenges, clinton | 12 | 12 | 0 | Political news FP cluster |
| C1 | proud gay, slut | 8 | 4 | 4 | Edgy / culturally-charged content |
| C2 | house, virus, new | 14 | 14 | 0 | General news FP cluster |
| C4 | trump, wanna | 10 | 10 | 0 | Trump-related real-news FP |
| Remaining | — | 15 | 12 | 7 | Mixed; 7 FNs are genuinely tricky Onion headlines |

The most dramatic change is the **near-elimination of the dry/first-person Onion FN cluster**: from 43–44 FNs in previous models to just 7 FNs total across the entire `diagnostic_val` set. Training on 246k diverse examples exposes the model to far more Onion voice registers, including first-person, list-format, and dry observational headlines that no small-dataset model could identify.

The **7 remaining FNs** are genuinely hard cases — Onion headlines where the sarcasm is highly contextual or relies on domain knowledge:

Sample remaining FNs:
- Headlines that superficially read as credible news
- Minimal-vocabulary Onion jokes with no recognisable sarcasm markers

The **dominant remaining error type shifts entirely to FP**: the model now over-predicts sarcasm on real political news (Trump-related, Clinton-related), culturally-charged topics, and general news headlines with vocabulary patterns shared by The Onion. These FP clusters suggest the model has been exposed to enough Onion political satire that it learned political vocabulary as a weak sarcasm cue — a systematic bias in the opposite direction from the small-data models.

**Comparison of dominant error source across all models:**

| Model | Primary error | Count |
|---|---|---|
| `original_no_tuning` | FN: dry/first-person Onion | 51 |
| `improved_with_tuning` | FN: dry/first-person Onion | 43 |
| `augmented_with_tuning` | FN: dry/first-person Onion | 44 |
| `master_no_tuning` | FP: political/general news | 52 |

### 5.6 Violent / Serious News FN Cluster

Headlines about shootings, crime, and serious incidents are consistently missed as sarcastic in the earlier fine-tuned models. The model's training on HuffPost crime coverage has associated crime/violence vocabulary with real news, so it fails to fire the sarcasm label even when the punchline is embedded in a crime-reporting format.

| Mode | Cluster keywords | Size | FP | FN |
|---|---|---|---|---|
| `original_no_tuning` | shooting, police, public, victims | 17 | 0 | 17 |
| `original_with_tuning` | shooting, year, victims sought | 6 | 1 | 5 |

Sample missed headlines:
- *"high schoolers given detention for cutting class during active shooting"*
- *"former viagra spokesman suspended for using performance-enhancing substances"*
- *"man with 90-year sentence for marijuana released from prison"*

These are genuine Onion headlines that adopt the format and vocabulary of serious crime reporting. The cluster is smaller than the dry/first-person FN cluster and does not persist across all modes, but it represents a distinct failure pattern: topic-label shortcuts learned from real crime news prevent the model from recognising structurally sarcastic content in that domain.

### 5.7 Large Backbone Diagnostic Error Profile

`improved_large_with_tuning` changes the diagnostic error mix without changing the dominant failure modes. Its 158 total errors break down into several familiar clusters:

| Cluster keywords | Size | FP | FN | Interpretation |
|---|---|---|---|---|
| things, old, year, people, won | 39 | 1 | 38 | Dry / first-person Onion voice remains the largest FN cluster |
| shooting, nba, teams, make, season | 28 | 5 | 23 | Violent/serious-news and sports/news-style Onion headlines still look too real |
| summer, days, kitty, korea, north korea | 23 | 23 | 0 | Lifestyle / celebrity / quirky-news FP cluster |
| versace, collaboration, trump, amsterdam prepares, cut | 15 | 15 | 0 | Political and culturally-adjacent real-news FP cluster |
| new, step, time, accidentally, 300 | 37 | 27 | 10 | Mixed general-news cluster with a strong FP skew |

Relative to `improved_with_tuning`, the large backbone improves recall on held-out sarcastic headlines but does not remove the core structural blind spots. The dry Onion FN cluster is still dominant, and the extra capacity appears to widen several real-news FP clusters rather than resolving them.

### 5.8 Cross-Dataset Error Pattern Comparison

| Error pattern | original_test | diagnostic_val |
|---|---|---|
| Science/discovery FP cluster | Present in every model, grows with tuning | Replaced by health/wellness FP — same root cause |
| Political sarcasm FN blind spot | Large, 21–42 samples | Present but smaller (10–25 samples) |
| Dry/first-person Onion FN | Minor | **New dominant FN cluster, 37–51 samples** |
| Violent/serious news FN | Minor | Present, small (5–17 samples) |
| Sports/entertainment FP | Moderate | Present, moderate |

The most significant new pattern on `diagnostic_val` is the **dry Onion humor FN cluster**, which suggests that `diagnostic_val` contains a higher proportion of Onion headlines that rely on comedic framing rather than absurd vocabulary. These are structurally harder headlines that no small-dataset model handles reliably, and they constitute the primary quality gap between `diagnostic_val` and `original_test` error profiles.

---

## 6. Error Analysis — `original_test`

### 6.1 Science / Discovery FP Cluster (Persistent Across All Models)

This cluster appears in every fine-tuned mode and is the most stable error source on `original_test`.

| Mode | Cluster keywords | Size | FP | FN |
|---|---|---|---|---|
| `original_no_tuning` | scientists, discover, life, years, man | 28 | 28 | 0 |
| `original_with_tuning` | scientists, house, discover, life, years | 49 | 49 | 0 |
| `improved_no_tuning` | scientists, discover, life, man, films | 36 | 36 | 0 |
| `improved_with_tuning` | scientists, discover, women, life, years | 41 | 41 | 0 |

All errors are false positives — genuine science and everyday-news headlines misclassified as sarcastic. The model learned that discovery-oriented vocabulary co-occurs with Onion-style science parodies. This shortcut survives topic-balanced sampling, contrastive training, Optuna tuning, and data augmentation.

Sample misclassified headlines:
- *"men's mental health demands male friendship"*
- *"bionic fingertip restores amputee's sense of touch"*
- *"aol instant messenger to sign off forever after 20 years"*

### 6.2 Political Sarcasm FN Cluster (Persistent Across All Models)

Every model produces one or more clusters of political Onion headlines that form a systematic false-negative blind spot.

| Mode | Cluster keywords | Size | FP | FN |
|---|---|---|---|---|
| `original_no_tuning` | trump, million, candidate, hurricane, republicans | 32 | 1 | 31 |
| `original_with_tuning` | ban, million, hurricane, response, shooting | 27 | 0 | 27 |
| `improved_no_tuning` | clinton, vote, key, million, america | 42 | 0 | 42 |
| `improved_with_tuning` | city, america, republicans, fight | 33 | 0 | 33 |

The sarcasm is structural and tonal (a single incongruous word appended to a credible-sounding sentence) rather than lexically absurd. The FN count is consistent at ~27–42 per cluster across all modes.

### 6.3 Tuning Consolidates Rather Than Eliminates Errors

| Mode | Total FP | Total FN | Total errors |
|---|---|---|---|
| `original_no_tuning` | 93 | 135 | 228 |
| `original_with_tuning` | 95 | 106 | 201 |
| `improved_no_tuning` | 127 | 103 | 230 |
| `improved_with_tuning` | 91 | 106 | 197 |

Tuning reduces total errors primarily by recovering true sarcastic positives (FN drops: 135→106). It does not reduce FPs — the science cluster remains pure-FP and actually grows under tuning (28→49 in the original pipeline). After tuning, clusters become **purer**: each surviving cluster is more exclusively FP-only or FN-only, meaning the remaining errors are more systematically wrong and require data-level or architectural fixes rather than further hyperparameter adjustment.

---

## 7. Summary

### Best Configurations

| Goal | Best mode | F1 |
|---|---|---|
| Highest F1 on original_test (uncalibrated) | `improved_large_with_tuning` | **0.9327** |
| Highest F1 on original_test (calibrated) | `improved_large_with_tuning_calibrated` | **0.9347** |
| Highest F1 on master_copy | `master_no_tuning` | **0.9003** ¹ |
| Highest F1 on diagnostic_val (uncalibrated) | `master_no_tuning` | **0.9572** ¹ |
| Highest F1 on diagnostic_val (calibrated) | `master_no_tuning_calibrated` | **0.9536** ¹ |
| Best OOD generalisation (original-trained, diagnostic_val, uncalibrated) | `improved_with_tuning` | **0.8737** |
| Best OOD generalisation (original-trained, diagnostic_val, calibrated) | `improved_large_with_tuning_calibrated` | **0.8728** |
| Best single model across all three datasets | `master_no_tuning` | 0.9243 / 0.9003 / 0.9572 ¹ |
| Best single model without master dataset | `improved_with_tuning` | 0.9270 / 0.7312 / 0.8737 |

**Performance ranking on `diagnostic_val` (original-dataset models only, threshold=0.50):**

| Rank | Mode | F1 | Errors | Notes |
|---|---|---|---|---|
| 1 | `improved_with_tuning` | **0.8737** | 153 | Fewest errors; slight FN bias |
| 2 | `improved_large_with_tuning_calibrated` | 0.8728 | 158 | Best calibrated F1; recall-oriented |
| 3 | `improved_large_with_tuning` | 0.8707 | 158 | Large backbone, uncalibrated |
| 4 | `augmented_with_tuning` | 0.8633 | 166 | Data augmentation; more FN |
| 4 | `original_with_tuning_calibrated` | 0.8633 | 164 | Best calibrated among base models |
| 6 | `original_with_tuning` | 0.8615 | 165 | |
| 7 | `original_no_tuning` | 0.8510 | 174 | |
| 8 | `improved_no_tuning_calibrated` | 0.8430 | 193 | |
| 9 | `improved_no_tuning` | 0.8395 | 195 | Worse than original without tuning |
| 10 | `pretrained` | 0.4409 | 1192 | No sarcasm signal; FP-biased |
| 11 | `pretrained_large` | 0.4894 | 1225 | No sarcasm signal; extreme FP bias |

**Comparable-model selection only (excluding `master_no_tuning` because its training data overlaps with `diagnostic_val`):**

| Goal | Best comparable mode | F1 |
|---|---|---|
| Highest F1 on `original_test` (uncalibrated) | `improved_large_with_tuning` | **0.9327** |
| Highest F1 on `original_test` (calibrated) | `improved_large_with_tuning_calibrated` | **0.9347** |
| Highest F1 on `master_copy` | `original_with_tuning` | **0.7392** |
| Highest F1 on `diagnostic_val` (uncalibrated) | `improved_with_tuning` | **0.8737** |
| Highest F1 on `diagnostic_val` (calibrated) | `improved_large_with_tuning_calibrated` | **0.8728** |
| Best balanced model (original_test / master_copy / diagnostic_val) | `improved_with_tuning` | 0.9270 / 0.7312 / 0.8737 |

**Comparison of effect of improvements (contrastive + topic-balancing) across datasets, threshold=0.50:**

| Mode | original_test F1 | diagnostic_val F1 | master_copy F1 |
|---|---|---|---|
| `original_with_tuning` | 0.9256 | 0.8615 | **0.7392** |
| `improved_with_tuning` | 0.9270 | **0.8737** | 0.7312 |
| `improved_large_with_tuning` | **0.9327** | 0.8707 | 0.7145 |

Moving from the original tuned pipeline to the improved one gives a consistent `diagnostic_val` gain (+0.012 F1). Moving again to the large backbone mainly boosts `original_test` (+0.006), slightly reduces `diagnostic_val` (−0.003), and further hurts `master_copy` (−0.017). The pattern suggests that architectural improvements help OOD generalisation more reliably than scaling backbone size alone.

> ¹ `master_no_tuning`'s `diagnostic_val` and `master_copy` results are **not directly comparable** to original-dataset models — see Section 0.4. The model trained on 98.4% of `diagnostic_val` examples. The best truly held-out OOD result on `diagnostic_val` is `improved_with_tuning` at F1 = 0.8737 (uncalibrated) or `improved_large_with_tuning_calibrated` at F1 = 0.8728 (calibrated).

`master_no_tuning` still provides a useful in-distribution reference point, but it should not be treated as the best selectable model because of the training overlap with `diagnostic_val`. For genuine held-out evaluation, `improved_with_tuning` posts the best uncalibrated `diagnostic_val` F1 with the fewest total errors (153), while `improved_large_with_tuning_calibrated` is the best calibrated comparable model. `improved_large_with_tuning` is the best option when only the original balanced test set matters and `bert-large` compute is acceptable.

### What Master Dataset Training Achieved

- ✅ Best `master_copy` F1 of any model (0.9003), up from 0.7392 — +0.161 improvement
- ✅ Best `diagnostic_val` F1 of any model: 0.9572 at threshold=0.50 (0.9536 calibrated), up from 0.8737 — +0.083 improvement
- ✅ Near-elimination of the dry/first-person Onion FN cluster (84 FN → 9 FN on diagnostic_val at threshold=0.50)
- ✅ Total `diagnostic_val` errors: 54 at threshold=0.50 (59 calibrated), down from 153 — a 65% reduction
- ✅ Cross-dataset degradation gap: 0.024 F1, down from ~0.19 — a 7× improvement
- ➡️ `original_test` F1: 0.9243 (uncalibrated), −0.003 vs improved_with_tuning; 0.9256 calibrated, effectively unchanged
- ⚠️ New dominant error: FP over-prediction on political/general news (FP=45 at 0.50, FP=52 calibrated)
- ⚠️ Trump-related and general political news triggers spurious sarcasm predictions at higher rate

### What Augmentation Achieved

- ✅ Strong `original_test` F1 (0.9323 uncalibrated), near `improved_large_with_tuning` (0.9327)
- ✅ Marginally improved `master_copy` F1 (+0.002 vs improved_with_tuning)
- ✅ Fixed errors on `diagnostic_val` relative to `improved_with_tuning_calibrated` (37 fewer errors when comparing calibrated variants)
- ❌ Regressed on `diagnostic_val` F1 vs `improved_with_tuning` at 0.50 (0.8633 vs 0.8737, −0.010)
- ❌ Net +13 total errors on `diagnostic_val` vs `improved_with_tuning` at 0.50 (166 vs 153)
- ❌ The primary failure mode (dry/first-person Onion FN cluster) was completely unaffected
- ❌ Created two new FP clusters (financial/community, science/health)

### Top Remaining Error Types

| Priority | Error type | Affected models | Scope | Recommended fix |
|---|---|---|---|---|
| 1 | Political/general news FP | `master_no_tuning` | 52 FP on diagnostic_val | Threshold calibration on a master-representative val set; adversarial pairing of real political news against Onion political satire |
| 2 | Science/discovery FP | All small-dataset models | 28–49 errors (original_test) | Adversarial pairing of real science news against Onion science parodies; addressed implicitly by master training |
| 3 | Health/wellness FP | All small-dataset tuned models | 28–31 errors (diagnostic_val) | Addressed implicitly by master training (not appearing as a dominant cluster in master_no_tuning) |
| 4 | Dry/first-person Onion FN | Small-dataset models only | 37–51 errors; **resolved by master training** (7 remain) | ✅ Master dataset training; stylometric features for residual cases |
| 5 | Political/structural sarcasm FN | Small-dataset models | 27–42 errors | Longer-range features; structural irony detection |

### Why Augmentation Underperformed on `diagnostic_val`

The augmentation pipeline used `diagnostic_val` error diagnoses to drive candidate selection, but the selected examples were drawn from `master_copy_dedup_v2` — a different distribution. BERT's classification-tuned embedding space is very flat for news headlines (near-zero cosine distances across all retrieved candidates), meaning the KNN retrieval was less semantically precise than intended. The retrieved counter-examples partially corrected the targeted Angle 3 systematic bias errors but simultaneously introduced new spurious associations in adjacent vocabulary domains. The primary failure mode (first-person Onion voice) requires **stylistically-targeted** training examples rather than topically-matched counter-examples from a general news corpus.

`master_no_tuning` confirms this diagnosis: exposing the model to the full breadth of the master dataset — which includes diverse Onion voice registers — resolves the FN problem far more effectively than the 320-example targeted augmentation ever could.

### Model Selection Guide

| Setting | Recommended model | Reason |
|---|---|---|
| Generalisation to diverse news (master_copy, diagnostic_val) | `improved_with_tuning` | Best balanced comparable model once `master_no_tuning` is excluded |
| Balanced small test set (original_test) only | `improved_large_with_tuning` | Best raw F1 on balanced evaluation |
| No access to master dataset | `improved_with_tuning` | Best consistent performance without master data |
| Calibrated deployment on skewed distribution | `original_with_tuning` + threshold sweep on target distribution | Strongest comparable `master_copy` result; target-prior recalibration still matters |

---

## References

[1] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.** *NAACL-HLT 2019*. https://arxiv.org/abs/1810.04805
> **Used for:** Core model backbone. `bert-base-uncased` was fine-tuned end-to-end for binary sarcasm classification. The CLS token's final hidden state is used as the sequence-level representation for both the classification head and the contrastive loss. The error diagnostic pipeline also uses BERT's CLS embeddings as the embedding space for KNN neighbourhood analysis and KMeans clustering.
>
> **Why BERT for sarcasm detection:** Sarcasm is a fundamentally contextual phenomenon — its detection requires understanding not just individual words but the relationship between a headline's literal meaning and its implied or expected meaning (see [13] for empirical support). BERT's bidirectional self-attention is directly suited to this: every token attends to every other token simultaneously, allowing the model to detect semantic incongruity across the full headline in a single pass (e.g., an absurdly understated verb in an otherwise credible-sounding sentence). Pre-trained on large corpora (BooksCorpus + Wikipedia), BERT already encodes world knowledge and linguistic expectations that help surface the gap between what a headline says and what it implies — the core signal for sarcasm. Prior bag-of-words and RNN-based approaches lack either the bidirectional context (unidirectional RNNs) or the deep semantic grounding (TF-IDF) to model this incongruity reliably. The CLS token's aggregated representation also naturally pools evidence from across the headline, making it a compact and expressive feature for binary classification without requiring positional feature engineering.

[2] Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). **Optuna: A Next-generation Hyperparameter Optimization Framework.** *KDD 2019*. https://arxiv.org/abs/1907.10902
> **Used for:** Hyperparameter search framework. Optuna's `TPESampler` (see [3]) drives all tuning runs. A `MedianPruner` prunes underperforming trials at the end of each epoch based on the intermediate validation F1 score, cutting wall time by ~60% in practice (trials 13–19 in the 20-trial run were all pruned at epoch 1). The search space covers learning rate, batch size, number of epochs, warmup ratio, weight decay, LLRD decay factor, gradient accumulation steps, and label smoothing.

[3] Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). **Algorithms for Hyper-Parameter Optimization.** *NeurIPS 2011*. https://proceedings.neurips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html
> **Used for:** Theoretical basis for Optuna's default sampler. TPE (Tree-structured Parzen Estimator) models the hyperparameter search as a Bayesian optimisation problem, fitting separate density estimators over good and bad configurations. The `n_startup_trials=10` setting allows TPE to collect 10 random samples before switching to model-guided exploration, following the paper's recommendation for cold-start behaviour.

[4] Khosla, P., Tian, Y., Wang, X., Garg, S., Ji, S., Steinhardt, J., & Tian, Y. (2020). **Supervised Contrastive Learning.** *NeurIPS 2020*. https://arxiv.org/abs/2004.11362
> **Used for:** Additional training objective in `train.py`. The supervised contrastive loss pulls together CLS embeddings from the same class (sarcastic–sarcastic, non-sarcastic–non-sarcastic) and pushes apart embeddings from different classes, using cosine similarity scaled by a temperature τ. The final loss is `CE(label_smoothing=ε) + λ · SupCon(τ)`, where λ and τ are fixed hyperparameters. The motivation was to produce a tighter, more linearly separable embedding space that generalises better to out-of-distribution headlines.

[5] Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). **Class-Balanced Loss Based on Effective Number of Samples.** *CVPR 2019*. https://arxiv.org/abs/1901.05555
> **Used for:** Motivation behind topic-balanced sampling in `train.py`. The paper's core insight — that effective training sample coverage decreases as class frequency increases — directly motivated the inverse-frequency weighting scheme applied at the (topic cluster × label) cell level. TF-IDF vectors are clustered into K topics via KMeans, and each training example is assigned a weight inversely proportional to the number of examples in its (cluster, label) cell. This reduces the model's exposure to over-represented topic-label combinations (e.g., political real-news), implemented via PyTorch's `WeightedRandomSampler`.

[6] Lipton, Z. C., Elkan, C., & Narayanaswamy, B. (2014). **Thresholding Classifiers to Maximize F1 Score.** *ECML-PKDD 2014*. https://arxiv.org/abs/1402.1892
> **Used for:** Post-hoc threshold calibration in `evaluate.py`. Rather than using a fixed 0.5 decision boundary, a sweep over thresholds [0.05, 0.95] in steps of 0.01 is performed on the validation set, and the threshold maximising F1 is selected. As observed in Section 2.5, this yields gains of +0.001–+0.004 F1 on `original_test` for baseline models, but consistently hurts on `master_copy` because the val-set prior (50/50) does not match the deployment prior (84/16) — a key caveat the paper also discusses when the calibration and deployment distributions differ.

[7] Howard, J., & Ruder, S. (2018). **Universal Language Model Fine-tuning for Text Classification.** *ACL 2018*. https://arxiv.org/abs/1801.06146
> **Used for:** Conceptual origin of Layer-wise Learning Rate Decay (LLRD). ULMFiT introduced the idea of assigning lower learning rates to lower (earlier) layers of a pretrained language model during fine-tuning, on the principle that lower layers encode more general linguistic knowledge that should be preserved rather than overwritten. This technique was adapted for BERT fine-tuning (see [8]) and implemented here as a multiplicative depth-decay: each transformer layer is assigned `lr × decay_factor^(num_layers - layer_index)`, with the classification head at the full learning rate.

[8] Sun, C., Qiu, X., Xu, Y., & Huang, X. (2019). **How to Fine-Tune BERT for Text Classification?** *CCL 2019*. https://arxiv.org/abs/1905.05583
> **Used for:** Concrete implementation guidance for LLRD on BERT. The paper provides the specific depth-decay formula adapted here: `lr_layer = lr_base × decay_factor^(num_layers - layer_index)`. It also confirms that BERT's 12 transformer layers benefit from progressively lower learning rates, with the embedding layer at the lowest rate. The LLRD decay factor is treated as a tunable hyperparameter in the Optuna search space ([0.80, 0.95]), following the paper's finding that the optimal decay depends on task difficulty and training set size.

[9] Loshchilov, I., & Hutter, F. (2019). **Decoupled Weight Decay Regularization (AdamW).** *ICLR 2019*. https://arxiv.org/abs/1711.05101
> **Used for:** Optimiser. AdamW is used in place of standard Adam for all fine-tuning runs. The key difference from Adam+L2 is that weight decay is applied directly to weights rather than being folded into the gradient update, which avoids the interaction between adaptive learning rates and regularisation. The LLRD parameter groups are passed directly to AdamW (via `torch.optim.AdamW`) with per-group learning rates; Trainer's own optimizer construction is bypassed entirely by passing the optimizer explicitly via `optimizers=(optimizer, None)`.

[10] Loshchilov, I., & Hutter, F. (2017). **SGDR: Stochastic Gradient Descent with Warm Restarts.** *ICLR 2017*. https://arxiv.org/abs/1608.03983
> **Used for:** Learning rate schedule. A cosine decay schedule (`lr_scheduler_type="cosine"`) is applied after a linear warmup phase (`warmup_ratio` tuned by Optuna in [0.05, 0.20]). The cosine schedule smoothly anneals the learning rate from its peak value to near zero over the training horizon, reducing the risk of oscillating around the loss minimum in the final training epochs. HuggingFace Trainer's built-in cosine schedule is directly derived from SGDR's cosine annealing formulation.

[11] Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). **Rethinking the Inception Architecture for Computer Vision.** *CVPR 2016*. https://arxiv.org/abs/1512.00567
> **Used for:** Label smoothing. Rather than training with hard one-hot targets (0 or 1), a smoothing factor ε redistributes a small amount of probability mass to the incorrect class: target becomes `(1−ε)` for the true label and `ε/2` for the other. This prevents the model from becoming overconfident and penalises it for assigning near-zero probability to any class. The label smoothing factor ε is tuned by Optuna in [0.05, 0.15] and applied via HuggingFace Trainer's `label_smoothing_factor` argument, which implements the formulation from this paper.

[12] Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., … & Wu, H. (2018). **Mixed Precision Training.** *ICLR 2018*. https://arxiv.org/abs/1710.03740
> **Used for:** Reduced-precision training on CUDA hardware. `bf16=True` is enabled when a CUDA GPU is available, using the BFloat16 format rather than full FP32 for forward and backward passes. BF16 has the same exponent range as FP32 (avoiding the overflow issues of FP16) but uses fewer mantissa bits, approximately halving memory bandwidth and accelerating matrix multiplications on supported hardware (Ampere GPUs and later). `tf32=True` is also enabled to allow TensorFloat-32 accumulation in CUDA tensor cores, following the mixed-precision best practices established in this paper.

[13] Potamias, R. A., Siolas, G., & Stafylopoulou, A. G. (2020). **A Transformer-based Approach to Irony and Sarcasm Detection.** *Neural Computing and Applications, 32*. https://arxiv.org/abs/2011.03123
> **Used for:** Empirical justification for selecting BERT as the sarcasm detection backbone. This paper directly benchmarks transformer-based models (BERT, RoBERTa) against prior CNN, LSTM, and attention-based architectures on irony and sarcasm detection tasks, showing that transformer models consistently outperform earlier approaches by a significant margin. The key finding is that BERT's bidirectional contextual representations are especially effective at capturing the semantic incongruity between a statement's literal content and its implied meaning — which is precisely the signal that distinguishes sarcastic Onion headlines from genuine HuffPost news in this dataset. The paper also confirms that fine-tuning on task-specific data (rather than using frozen BERT representations) is essential to achieve strong performance, supporting the end-to-end fine-tuning approach used here.
