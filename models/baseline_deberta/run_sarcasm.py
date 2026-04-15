#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeBERTa + LoRA for Sarcasm Detection with Optuna HPO
Each trial evaluates on both val and test sets for comparison.
"""

# -- Imports & Config --

import os
import re
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
import optuna
from optuna.pruners import MedianPruner

SEED = 42
set_seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path('deBERTa')
DATA_PATH = BASE_DIR / 'Sarcasm_Headlines_Dataset.json'
RESULTS_ROOT = BASE_DIR / 'results'
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR = RESULTS_ROOT / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'microsoft/deberta-v3-base'
TEXT_COL = 'headline'
LABEL_COL = 'is_sarcastic'

# -- GPU diagnostics --
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

USE_FP16 = False
USE_BF16 = False

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    capability = torch.cuda.get_device_capability()
    print(f'GPU: {gpu_name}')
    print(f'VRAM: {vram_gb:.1f} GB')
    print(f'Compute capability: {capability}')
    if capability[0] >= 8:
        USE_BF16 = True
        print('Using bf16 mixed precision')
    else:
        USE_FP16 = True
        print('Using fp16 mixed precision')
else:
    print('No GPU detected - running on CPU (this will be slow)')

print(f'Device: {device}')
print(f'Run directory: {RUN_DIR.resolve()}')


# -- Load Dataset --

df = pd.read_json(DATA_PATH, lines=True)
required_cols = {'article_link', TEXT_COL, LABEL_COL}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f'Missing expected columns: {missing}')

def infer_source(url: str) -> str:
    u = str(url).lower()
    if 'theonion.com' in u:
        return 'theonion'
    if 'huffingtonpost.com' in u or 'huffpost.com' in u:
        return 'huffpost'
    m = re.search(r'https?://([^/]+)', u)
    return m.group(1) if m else 'unknown'

df['source'] = df['article_link'].map(infer_source)
df['text_len'] = df[TEXT_COL].astype(str).str.len()
df['token_len_approx'] = df[TEXT_COL].astype(str).str.split().str.len()
df['exclamation'] = df[TEXT_COL].astype(str).str.contains('!').astype(int)
df['question'] = df[TEXT_COL].astype(str).str.contains(r'\?').astype(int)

print('Shape:', df.shape)
print('Label distribution:')
print(df[LABEL_COL].value_counts(normalize=True).rename('ratio'))


# -- Dataset Profile --

profile = {
    'num_rows': int(len(df)),
    'label_counts': df[LABEL_COL].value_counts().to_dict(),
    'source_counts_top20': df['source'].value_counts().head(20).to_dict(),
    'text_len_mean': float(df['text_len'].mean()),
    'text_len_p95': float(df['text_len'].quantile(0.95)),
}
(RUN_DIR / 'dataset_profile.json').write_text(json.dumps(profile, indent=2), encoding='utf-8')


# -- Train/Val/Test Split --

def make_joint_strata(frame, label_col, source_col):
    return frame[label_col].astype(str) + '__' + frame[source_col].astype(str)

def collapse_rare(s, min_count=2, other='__other__'):
    vc = s.value_counts(dropna=False)
    rare = vc[vc < min_count].index
    return s.where(~s.isin(rare), other=other)

def stratify_is_valid(y, test_size):
    vc = y.value_counts(dropna=False)
    if len(vc) == 0 or vc.min() < 2:
        return False
    n = len(y)
    n_test = int(np.ceil(n * test_size))
    n_train = n - n_test
    n_classes = vc.size
    return n_classes <= n_test and n_classes <= n_train

def safe_split(frame, test_size, seed, label_col, source_col):
    y_joint = collapse_rare(make_joint_strata(frame, label_col, source_col), min_count=2)
    y_label = frame[label_col]
    for y, name in [(y_joint, 'joint'), (y_label, 'label'), (None, 'none')]:
        if y is None or stratify_is_valid(y, test_size):
            a, b = train_test_split(
                frame, test_size=test_size, random_state=seed,
                stratify=y if y is not None else None,
            )
            print(f'Split strategy used: {name}')
            return a, b
    raise RuntimeError('No valid split strategy found.')

train_df, val_df = safe_split(df, test_size=0.1, seed=SEED, label_col=LABEL_COL, source_col='source')

# Load external test set
test_df = pd.read_csv('data/output/testing_dataset_final.csv')
test_df['source'] = test_df['article_link'].map(infer_source)
test_df['text_len'] = test_df[TEXT_COL].astype(str).str.len()
test_df['token_len_approx'] = test_df[TEXT_COL].astype(str).str.split().str.len()
test_df['exclamation'] = test_df[TEXT_COL].astype(str).str.contains('!').astype(int)
test_df['question'] = test_df[TEXT_COL].astype(str).str.contains(r'\?').astype(int)
test_df.dropna(subset=[TEXT_COL, LABEL_COL], inplace=True)

print('Train/Val/Test sizes:', len(train_df), len(val_df), len(test_df))


# -- Tokenization --

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def to_hf_dataset(frame: pd.DataFrame) -> Dataset:
    out = frame[[TEXT_COL, LABEL_COL, 'source', 'text_len', 'token_len_approx', 'exclamation', 'question']].copy()
    out = out.rename(columns={LABEL_COL: 'label'})
    return Dataset.from_pandas(out, preserve_index=False)

train_ds = to_hf_dataset(train_df)
val_ds = to_hf_dataset(val_df)
test_ds = to_hf_dataset(test_df)

def tokenize(batch):
    return tokenizer(batch[TEXT_COL], truncation=True, max_length=128)

train_tok = train_ds.map(tokenize, batched=True)
val_tok = val_ds.map(tokenize, batched=True)
test_tok = test_ds.map(tokenize, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print('Tokenization complete.')


# -- Metrics --

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    logits = np.array(predictions)
    labels = np.array(labels)
    if logits.ndim == 3:
        logits = logits[:, 0, :]
    if labels.ndim > 1:
        labels = labels.ravel()
    probs = torch.softmax(torch.tensor(logits.astype(np.float32)), dim=-1).numpy()
    preds = probs.argmax(axis=1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = 0.0
    return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1, 'roc_auc': auc}


def evaluate_on_test(trainer, test_dataset):
    """Evaluate a trainer on the test set and return metrics."""
    raw_pred = trainer.predict(test_dataset)
    predictions = raw_pred.predictions
    labels = raw_pred.label_ids
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    logits = np.array(predictions)
    labels = np.array(labels)
    if logits.ndim == 3:
        logits = logits[:, 0, :]
    if labels.ndim > 1:
        labels = labels.ravel()
    probs = torch.softmax(torch.tensor(logits.astype(np.float32)), dim=-1).numpy()
    preds = probs.argmax(axis=1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = 0.0
    return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1, 'roc_auc': auc}


# -- LoRA + Optuna HPO --

# Store results for each trial (val + test)
trial_results = []

def objective(trial: optuna.Trial) -> float:
    """Optuna objective: maximize validation F1 with LoRA."""

    # -- Hyperparameter sampling --
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    warmup = trial.suggest_float('warmup_ratio', 0.0, 0.2)
    wd = trial.suggest_float('weight_decay', 0.0, 0.1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    num_epochs = trial.suggest_categorical('num_epochs', [3, 4, 5])
    label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.15)

    # LoRA hyperparameters
    lora_r = trial.suggest_categorical('lora_r', [4, 8, 16, 32])
    lora_alpha = trial.suggest_categorical('lora_alpha', [8, 16, 32, 64])
    lora_dropout = trial.suggest_float('lora_dropout', 0.0, 0.3)

    # Build model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=['query_proj', 'key_proj', 'value_proj', 'dense'],
        bias='none',
    )

    model = get_peft_model(base_model, lora_config)

    # Print trainable params on first trial
    if trial.number == 0:
        model.print_trainable_parameters()

    trial_dir = RUN_DIR / f'trial_{trial.number}'
    trial_dir.mkdir(exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(trial_dir / 'checkpoints'),
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='steps',
        logging_steps=100,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=num_epochs,
        warmup_ratio=warmup,
        weight_decay=wd,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=1,
        report_to='none',
        seed=SEED,
        fp16=USE_FP16,
        bf16=USE_BF16,
        dataloader_pin_memory=torch.cuda.is_available(),
        label_smoothing_factor=label_smoothing,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    val_result = trainer.evaluate()
    val_f1 = val_result['eval_f1']

    # Also evaluate on test set
    test_metrics = evaluate_on_test(trainer, test_tok)

    # Log both val and test results
    trial_info = {
        'trial': trial.number,
        'val_f1': val_f1,
        'val_accuracy': val_result['eval_accuracy'],
        'val_precision': val_result['eval_precision'],
        'val_recall': val_result['eval_recall'],
        'val_roc_auc': val_result['eval_roc_auc'],
        'test_f1': test_metrics['f1'],
        'test_accuracy': test_metrics['accuracy'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'test_roc_auc': test_metrics['roc_auc'],
        'val_test_f1_gap': val_f1 - test_metrics['f1'],
    }
    trial_info.update(trial.params)
    trial_results.append(trial_info)

    print(f"\n--- Trial {trial.number} ---")
    print(f"  Val  F1: {val_f1:.4f}  |  Test F1: {test_metrics['f1']:.4f}  |  Gap: {val_f1 - test_metrics['f1']:.4f}")
    print(f"  Params: r={lora_r}, alpha={lora_alpha}, lr={lr:.6f}, bs={batch_size}, epochs={num_epochs}")

    # Report for pruning
    trial.report(val_f1, step=num_epochs)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    # Clean up GPU memory
    del model, base_model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return val_f1


# -- Run Optuna study --
N_TRIALS = 10

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    study_name='sarcasm_lora_hpo',
)

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print(f'\nBest trial: {study.best_trial.number}')
print(f'Best Val F1: {study.best_value:.4f}')
print(f'Best params: {json.dumps(study.best_params, indent=2)}')


# -- Save all trial results --

results_df = pd.DataFrame(trial_results)
results_df = results_df.sort_values('val_f1', ascending=False)
results_path = RUN_DIR / 'trial_results.csv'
results_df.to_csv(results_path, index=False)
print(f'\nSaved all trial results: {results_path}')

# Print summary table
print('\n=== Trial Results Summary (sorted by Val F1) ===')
print(f"{'Trial':>5} {'Val F1':>8} {'Test F1':>8} {'Gap':>8} {'r':>3} {'alpha':>5} {'lr':>10} {'bs':>3} {'epochs':>6}")
print('-' * 70)
for _, row in results_df.head(15).iterrows():
    print(f"{int(row['trial']):>5} {row['val_f1']:>8.4f} {row['test_f1']:>8.4f} {row['val_test_f1_gap']:>8.4f} "
          f"{int(row['lora_r']):>3} {int(row['lora_alpha']):>5} {row['learning_rate']:>10.6f} "
          f"{int(row['batch_size']):>3} {int(row['num_epochs']):>6}")

# Save best params
bp = study.best_params
best_params_path = RUN_DIR / 'best_params.json'
best_params_path.write_text(json.dumps(bp, indent=2), encoding='utf-8')

# Find the trial with best TEST F1 too
best_test_trial = results_df.loc[results_df['test_f1'].idxmax()]
print(f"\nBest by Val  F1: Trial {study.best_trial.number} (val={study.best_value:.4f})")
print(f"Best by Test F1: Trial {int(best_test_trial['trial'])} (test={best_test_trial['test_f1']:.4f}, val={best_test_trial['val_f1']:.4f})")
print(f"Average Val-Test gap: {results_df['val_test_f1_gap'].mean():.4f}")


# -- Retrain best model on full Train+Val, evaluate on Test --

print('\n=== Retraining best model on Train+Val ===')

best_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=bp['lora_r'],
    lora_alpha=bp['lora_alpha'],
    lora_dropout=bp['lora_dropout'],
    target_modules=['query_proj', 'key_proj', 'value_proj', 'dense'],
    bias='none',
)

best_model = get_peft_model(best_model, lora_config)

from datasets import concatenate_datasets
full_train_tok = concatenate_datasets([train_tok, val_tok])

final_args = TrainingArguments(
    output_dir=str(RUN_DIR / 'best_checkpoints'),
    eval_strategy='no',
    save_strategy='epoch',
    logging_strategy='steps',
    logging_steps=50,
    learning_rate=bp['learning_rate'],
    per_device_train_batch_size=bp['batch_size'],
    per_device_eval_batch_size=64,
    num_train_epochs=bp['num_epochs'],
    warmup_ratio=bp['warmup_ratio'],
    weight_decay=bp['weight_decay'],
    save_total_limit=1,
    report_to='none',
    seed=SEED,
    fp16=USE_FP16,
    bf16=USE_BF16,
    dataloader_pin_memory=torch.cuda.is_available(),
    label_smoothing_factor=bp['label_smoothing'],
)

(RUN_DIR / 'training_args.json').write_text(final_args.to_json_string(), encoding='utf-8')

final_trainer = Trainer(
    model=best_model,
    args=final_args,
    train_dataset=full_train_tok,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

final_trainer.train()
print('Best model retrained on full train+val set.')


# -- Evaluate on Test Set --

raw_pred = final_trainer.predict(test_tok)
predictions = raw_pred.predictions
if isinstance(predictions, tuple):
    predictions = predictions[0]
logits = np.array(predictions)
labels = raw_pred.label_ids
if isinstance(labels, tuple):
    labels = labels[0]
labels = np.array(labels)
if labels.ndim > 1:
    labels = labels.ravel()

probs = torch.softmax(torch.tensor(logits.astype(np.float32)), dim=-1).numpy()
preds = probs.argmax(axis=1)
conf = probs.max(axis=1)

p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
acc = accuracy_score(labels, preds)
auc = roc_auc_score(labels, probs[:, 1])
cm = confusion_matrix(labels, preds)

metrics = {
    'accuracy': float(acc),
    'precision': float(p),
    'recall': float(r),
    'f1': float(f1),
    'roc_auc': float(auc),
    'confusion_matrix': cm.tolist(),
    'run_id': RUN_ID,
    'model_name': MODEL_NAME,
    'method': 'LoRA',
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
    'mixed_precision': 'bf16' if USE_BF16 else ('fp16' if USE_FP16 else 'none'),
    'lora_config': {
        'r': bp['lora_r'],
        'alpha': bp['lora_alpha'],
        'dropout': bp['lora_dropout'],
    },
    'best_optuna_params': bp,
}

(RUN_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
print('\nFinal Test Results (retrained on train+val):')
for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    print(f'  {k}: {metrics[k]:.4f}')
print(f'  Confusion matrix: {cm.tolist()}')


# -- Save Predictions --

pred_df = test_df.reset_index(drop=True).copy()
pred_df['true_label'] = labels
pred_df['pred_label'] = preds
pred_df['prob_non_sarcastic'] = probs[:, 0]
pred_df['prob_sarcastic'] = probs[:, 1]
pred_df['confidence'] = conf
pred_df['correct'] = (pred_df['true_label'] == pred_df['pred_label']).astype(int)
pred_df['error_type'] = np.where(
    pred_df['correct'] == 1, 'correct',
    np.where(pred_df['true_label'] == 1, 'false_negative', 'false_positive'))

pred_path = RUN_DIR / 'predictions.csv'
pred_df.to_csv(pred_path, index=False, encoding='utf-8')
print('Saved:', pred_path)


# -- Slice Metrics --

def bin_text_len(x):
    if x < 50:
        return 'short(<50)'
    if x < 90:
        return 'medium(50-89)'
    return 'long(>=90)'

pred_df['len_bin'] = pred_df['text_len'].map(bin_text_len)

slice_specs = [
    ('source', 'source'),
    ('len_bin', 'len_bin'),
    ('exclamation', 'exclamation'),
    ('question', 'question'),
]

rows = []
for slice_name, col in slice_specs:
    for group_value, g in pred_df.groupby(col):
        y_true = g['true_label'].values
        y_pred = g['pred_label'].values
        y_prob = g['prob_sarcastic'].values
        auc_g = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) >= 2 else np.nan
        p_g, r_g, f1_g, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        rows.append({
            'slice_type': slice_name,
            'slice_value': str(group_value),
            'n': int(len(g)),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(p_g),
            'recall': float(r_g),
            'f1': float(f1_g),
            'roc_auc': None if pd.isna(auc_g) else float(auc_g),
            'error_rate': float(1 - g['correct'].mean()),
            'avg_confidence': float(g['confidence'].mean()),
        })

slice_df = pd.DataFrame(rows).sort_values(['slice_type', 'f1'], ascending=[True, True])
slice_path = RUN_DIR / 'slice_metrics.csv'
slice_df.to_csv(slice_path, index=False, encoding='utf-8')


# -- Error Analysis --

hard_errors = pred_df[pred_df['correct'] == 0].sort_values('confidence', ascending=False)
hc_errors = hard_errors.head(200)
hc_errors.to_csv(RUN_DIR / 'high_confidence_errors.csv', index=False, encoding='utf-8')

uncertain = pred_df.iloc[np.argsort(np.abs(pred_df['prob_sarcastic'] - 0.5).values)[:200]].copy()
uncertain.to_csv(RUN_DIR / 'uncertain_samples.csv', index=False, encoding='utf-8')

print(f'Saved {len(hc_errors)} high-confidence errors and {len(uncertain)} uncertain samples.')


# -- Gap Report --

worst_slices = slice_df.sort_values(['error_rate', 'n'], ascending=[False, False]).head(10)
fn_rate = (pred_df['error_type'] == 'false_negative').mean()
fp_rate = (pred_df['error_type'] == 'false_positive').mean()

report_lines = [
    f'# LoRA DeBERTa Gap Report - {RUN_ID}',
    '',
    '## Method',
    '- LoRA (Low-Rank Adaptation)',
    f"- LoRA r: {bp['lora_r']}",
    f"- LoRA alpha: {bp['lora_alpha']}",
    f"- LoRA dropout: {bp['lora_dropout']:.3f}",
    f"- Label smoothing: {bp['label_smoothing']:.3f}",
    '',
    '## Overall Test Metrics',
    f"- Accuracy: {metrics['accuracy']:.4f}",
    f"- F1: {metrics['f1']:.4f}",
    f"- Precision: {metrics['precision']:.4f}",
    f"- Recall: {metrics['recall']:.4f}",
    f"- ROC-AUC: {metrics['roc_auc']:.4f}",
    f"- FN rate (all test): {fn_rate:.4f}",
    f"- FP rate (all test): {fp_rate:.4f}",
    '',
    '## Optuna Summary',
    f"- Trials run: {len(study.trials)}",
    f"- Best trial: #{study.best_trial.number}",
    f"- Best val F1: {study.best_value:.4f}",
    f"- Average val-test F1 gap: {results_df['val_test_f1_gap'].mean():.4f}",
    '',
    '## Best Hyperparameters',
]
for k, v in bp.items():
    report_lines.append(f"- {k}: {v}")

report_lines += [
    '',
    '## Worst Performing Slices',
]
for _, row in worst_slices.iterrows():
    report_lines.append(
        f"- {row['slice_type']}={row['slice_value']} | n={int(row['n'])} "
        f"| f1={row['f1']:.4f} | error_rate={row['error_rate']:.4f}"
    )

report_lines += [
    '',
    '## High-Confidence Error Patterns',
    '- See high_confidence_errors.csv for systematic patterns.',
    '- See uncertain_samples.csv for boundary cases.',
    '- See trial_results.csv for val vs test comparison across all trials.',
]

report_path = RUN_DIR / 'gap_report.md'
report_path.write_text('\n'.join(report_lines), encoding='utf-8')
print('Saved:', report_path)
print('All artifacts saved in:', RUN_DIR.resolve())