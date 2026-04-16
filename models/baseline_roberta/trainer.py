# Standard libraries
import os

# Custom libraries
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding, get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
import torch
from tqdm import tqdm
import evaluate
from huggingface_hub import HfApi

# Local imports
from config import DATA_PATH, MODEL_NAME, NUM_LABELS, LEARNING_RATE, TRAIN_BATCH_SIZE, NUM_TRAIN_EPOCHS, NUM_WARMUP_STEPS, WEIGHT_DECAY
from dataset import load_json_dataset, SarcasmDataset
from preprocess import generate_splits

OUTPUT_DIR = 'outputs/checkpoints/roberta-optuna-hft'

def train():
    # Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    # Load dataset
    data = load_json_dataset([DATA_PATH])
    df = pd.DataFrame(data)
    train_df, val_df, _ = generate_splits(df)
    train_dataset = SarcasmDataset(train_df, tokenizer)
    val_dataset = SarcasmDataset(val_df, tokenizer)

    # DataLoader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=TRAIN_BATCH_SIZE
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=TRAIN_BATCH_SIZE
    )

    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_LABELS
    )
    optimizer = AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )

    num_train_epochs = NUM_TRAIN_EPOCHS
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    # Learning Rate Scheduler
    lr_scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=NUM_WARMUP_STEPS,
        num_training_steps=num_training_steps
    )

    '''
    Accelerator

    Keeps code consistent even if hardware is different
    '''
    accelerator = Accelerator()
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    progress_bar = tqdm(range(num_training_steps))
    metric = evaluate.load('glue', 'mrpc')

    best_f1 = 0.0
    best_checkpoint_dir = 'outputs/checkpoints/roberta-optuna-hft-best'
    os.makedirs(best_checkpoint_dir, exist_ok=True)

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        # Evaluation
        model.eval()
        for batch in val_dataloader:
            with torch.no_grad():
                outputs = model(**batch)
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch['labels'])
        
        results = metric.compute()

        print(
            f'epoch {epoch}:',
            {
                'accuracy': results['accuracy'],
                'f1': results['f1']
            }
        )

        if results['f1'] > best_f1:
            best_f1 = results['f1']
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(best_checkpoint_dir)
            tokenizer.save_pretrained(best_checkpoint_dir)
            print(f'New best model saved (f1={best_f1:.4f})')

    # Save model locally
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    best_model = RobertaForSequenceClassification.from_pretrained(best_checkpoint_dir)
    best_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f'Best model (f1={best_f1:.4f}) saved to {OUTPUT_DIR}')

    print(f'Model saved to {OUTPUT_DIR}')

if __name__ == '__main__':
    train()