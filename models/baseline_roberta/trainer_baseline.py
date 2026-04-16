# Standard libraries
import os

# Custom libraries
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Local imports
from config import DATA_PATH, MODEL_NAME, NUM_LABELS
from dataset import load_json_dataset, SarcasmDataset
from preprocess import generate_splits

OUTPUT_DIR = 'outputs/checkpoints/improved_no_tuning'

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='binary')
    }

def train():
    # Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    # Load dataset
    data = load_json_dataset([DATA_PATH])
    df = pd.DataFrame(data)
    train_df, val_df, _ = generate_splits(df)
    train_dataset = SarcasmDataset(train_df, tokenizer)
    val_dataset = SarcasmDataset(val_df, tokenizer)

    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_LABELS
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    training_args = TrainingArguments(
        output_dir='outputs/checkpoints/roberta-no-tuning',
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64, # evaluation does not require gradient recalculation, hence we use higher batch size
        warmup_steps=0, # Learning rate gradually increases from 0 to 1e-5
        weight_decay=0.01, # helps prevent overfitting
        learning_rate=1e-5,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        logging_steps=50,
        fp16=True,
        report_to='none'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == '__main__':
    train()