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
from dataset import SarcasmDatasetWithContext
from preprocess import generate_splits

'''
Research Paper used: LLM-as-a-judge for sarcasm detection using supervised fine-tuning of transformers (https://link.springer.com/article/10.1007/s44443-025-00379-7#Fn1)

Proposed changes:
- Using a cosine LR scheduler instead of linear
- Adding LABEL_SMOOTHING to improve generalisation and training stability
- Using Macro-F1 as the best metric

Research Paper used: A contextual-based approach for sarcasm detection (https://www.nature.com/articles/s41598-024-65217-8#Sec13)

Proposed changes:
- Including context of the article. We utilise both description and article section for training

'''

RAW_DATA_PATH = Path('data/Sarcasm_Headline_Dataset_v2.json')
CONTEXT_DATA_PATH = Path('data/sarcasm_with_context.csv')
USE_CONTEXT = True

OUTPUT_DIR = 'outputs/checkpoints/improved_with_tuning'

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='binary'),
        'macro_f1': f1_score(labels, preds, average='macro')
    }

def load_data():
    if USE_CONTEXT and CONTEXT_DATA_PATH.exists():
        print('Using context dataset')
        df = pd.read_csv(CONTEXT_DATA_PATH)
    else:
        print('Using raw dataset')
        df = pd.read_json(RAW_DATA_PATH, lines=True)
    
    return df

def train():
    # Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    # Load dataset
    df = load_data()
    train_df, val_df, _ = generate_splits(df)
    train_dataset = SarcasmDatasetWithContext(train_df, tokenizer)
    val_dataset = SarcasmDatasetWithContext(val_df, tokenizer)

    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_LABELS
    )

    # DataLoader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    training_args = TrainingArguments(
        output_dir='outputs/checkpoints/improved_with_tuning',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64, # evaluation does not require gradient recalculation, hence we use higher batch size
        lr_scheduler_type='linear',
        warmup_steps=300, # Learning rate gradually increases from 0 to 1e-5
        weight_decay=0.011953707930373357, # helps prevent overfitting
        learning_rate=1.8983017837543872e-05,
        label_smoothing_factor=0.00485547546035332,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        logging_steps=50,
        fp16=torch.cuda.is_available(),
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