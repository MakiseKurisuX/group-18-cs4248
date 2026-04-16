# Standard libraries
from pathlib import Path

# Custom libraries
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import numpy as np
from sklearn.metrics import accuracy_score

import torch
from sklearn.metrics import f1_score

# Local imports
from config import MODEL_NAME, NUM_LABELS
from dataset import SarcasmDataset
from preprocess import generate_splits

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'augmentation_output_roberta' / 'round_3' / 'train.csv'
AUG_PATH = PROJECT_ROOT / 'augmentation_output_roberta' / 'round_3' / 'augmentation_candidates.csv'

OUTPUT_DIR = 'outputs/checkpoints/augmented_with_tuning_rd3'

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='binary'),
        'macro_f1': f1_score(labels, preds, average='macro')
    }

def train():
    # Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    # Load dataset
    df = pd.read_csv(DATA_PATH)
    aug_df = pd.read_csv(AUG_PATH)
    aug_df = aug_df[['headline', 'is_sarcastic', 'article_link']]
    df = pd.concat([df, aug_df], ignore_index=True)
    print(f"Combined dataset size: {len(df)}")
    print(f"Combined label split: {df['is_sarcastic'].value_counts().to_dict()}")
    train_df, val_df, _ = generate_splits(df)
    train_dataset = SarcasmDataset(train_df, tokenizer)
    val_dataset = SarcasmDataset(val_df, tokenizer)

    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_LABELS
    )

    # DataLoader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    training_args = TrainingArguments(
        output_dir='outputs/checkpoints/augmented_with_tuning_rd3',
        num_train_epochs=4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64, # evaluation does not require gradient recalculation, hence we use higher batch size
        lr_scheduler_type='cosine',
        warmup_steps=300, # Learning rate gradually increases from 0 to 1e-5
        weight_decay=0.2594131160070775, # helps prevent overfitting
        learning_rate=1.070817464451154e-05,
        label_smoothing_factor=0.08950252644277484,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        logging_steps=50,
        fp16=True,
        report_to='none'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == '__main__':
    train()