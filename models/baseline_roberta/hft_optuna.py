# Standard libraries
import json, os
from pathlib import Path

# Custom libraries
from transformers import RobertaTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, TrainerCallback
import pandas as pd
import torch
import optuna
from optuna.trial import TrialState
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Local imports
from config import DATA_PATH, NUM_LABELS, NUM_TRIALS, RANDOM_STATE, NUM_STARTUP_TRIALS, NUM_WARMUP_STEPS
from dataset import SarcasmDatasetWithContext
from preprocess import generate_splits

# Available models
ORIGINAL_NO_TUNING = 'outputs/checkpoints/original_no-tuning'
ORIGINAL_WITH_TUNING = 'outputs/checkpoints/original_with_tuning'
IMPROVED_NO_TUNING = 'outputs/checkpoints/improved_no_tuning'

MODEL = IMPROVED_NO_TUNING
CONTEXT_DATA_PATH = Path('data/sarcasm_with_context.csv')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='binary'),
        'macro_f1': f1_score(labels, preds, average='macro')
    }

class OptunaHFT:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.tokenizer = RobertaTokenizer.from_pretrained(MODEL)

        # Load and split dataset
        df = pd.read_csv(CONTEXT_DATA_PATH)
        self.train_df, self.val_df, _ = generate_splits(df)

        # Load datasets
        self.train_dataset = SarcasmDatasetWithContext(self.train_df, self.tokenizer)
        self.val_dataset = SarcasmDatasetWithContext(self.val_df, self.tokenizer)
    
    def objective(self, trial: optuna.Trial) -> float:
        # Hyperparamters to fine-tune
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True)

        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])

        num_warmup_steps = trial.suggest_int('num_warmup_steps', 0, 1000, step=100)

        weight_decay = trial.suggest_float('weight_decay', 0.0, 0.3)

        label_smoothing_factor = trial.suggest_float('label_smoothing_factor', 0.0, 0.2)

        num_epochs = trial.suggest_int('num_epochs', 2, 5)

        lr_scheduler_type = trial.suggest_categorical(
            "lr_scheduler_type",
            ["linear", "cosine", "constant_with_warmup"]
        )

        # DataLoader
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Model
        model = RobertaForSequenceClassification.from_pretrained(
            MODEL,
            num_labels=NUM_LABELS
        )

        training_args = TrainingArguments(
            output_dir=f'outputs/hft_optuna/roberta_improved/trial_{trial.number}',
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=64,
            warmup_steps=num_warmup_steps,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            label_smoothing_factor=label_smoothing_factor,
            eval_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model='macro_f1',
            greater_is_better=True,
            logging_steps=50,
            fp16=torch.cuda.is_available(),
            report_to='none'
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[OptunaPruningCallback(trial)]
        )

        trainer.train()

        best_metric = trainer.state.best_metric
        return best_metric if best_metric is not None else 0.0
    
class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial: optuna.Trial):
        self.trial = trial

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch = int(state.epoch) if state.epoch is not None else 0
        macro_f1 = metrics.get('eval_macro_f1', 0.0)

        self.trial.report(macro_f1, step=epoch)

        if self.trial.should_prune():
            print(f"Trial {self.trial.number} pruned at epoch {epoch + 1}")
            raise optuna.TrialPruned()

def optimize_hyperparameters(
  n_trials: int = NUM_TRIALS,
  timeout: int = None,
  study_name: str = 'sarcasm_detection',
  direction: str = 'maximize',
  save_path: str = 'outputs/hft_optuna/roberta_improved'
):
    trainer = OptunaHFT()

    sampler = TPESampler(seed=RANDOM_STATE)

    pruner = MedianPruner(
        n_startup_trials=NUM_STARTUP_TRIALS,
        n_warmup_steps=NUM_WARMUP_STEPS
    )

    study = optuna.create_study(
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        storage=f"sqlite:///{save_path}.db",
        load_if_exists=True
    )

    print(f"num of trials: {n_trials}")
    study.optimize(
        trainer.objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    # Results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    # Best trial
    best_trial = study.best_trial
    
    print(f"\nBest Trial: {best_trial.number}")
    print(f"Best F1 Score: {best_trial.value:.4f}")
    print("\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Trial statistics
    print(f"\nTotal trials: {len(study.trials)}")
    print(f"Completed trials: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")
    print(f"Pruned trials: {len([t for t in study.trials if t.state == TrialState.PRUNED])}")
    print(f"Failed trials: {len([t for t in study.trials if t.state == TrialState.FAIL])}")
    
    # Top 5 trials
    print("\nTop 5 Trials:")
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)
    for i, trial in enumerate(sorted_trials[:5], 1):
        print(f"  {i}. Trial {trial.number}: F1={trial.value:.4f}")
    
    # Save results to JSON
    results = {
        'best_trial': best_trial.number,
        'best_value': best_trial.value,
        'best_params': best_trial.params,
        'total_trials': len(study.trials),
        'completed_trials': len([t for t in study.trials if t.state == TrialState.COMPLETE]),
        'pruned_trials': len([t for t in study.trials if t.state == TrialState.PRUNED]),
    }

    output_dir = 'outputs/hft_optuna/roberta_improved'
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'optuna_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to 'optuna_results.json'")
    print(f"Study database saved to '{save_path}'")
    
    return study, best_trial

if __name__ == "__main__":
    study, best_trial = optimize_hyperparameters()