# Standard libraries
import json
import os

# Custom libraries
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Local imports
from config import EVAL_BATCH_SIZE, MODEL_NAME
from dataset import SarcasmDataset

# Available models
AUGMENTED_WITH_TUNING_RD1 = 'outputs/checkpoints/augmented_with_tuning_rd1'
AUGMENTED_WITH_TUNING_RD2 = 'outputs/checkpoints/augmented_with_tuning_rd2'
ORIGINAL_WITH_TUNING = 'outputs/checkpoints/original_with_tuning'

PREDICTIONS_DATASET = 'data/diagnostic_val.csv'

def evaluation(
    pretrained
):
    if pretrained:
        print("Running pretrained model")
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
        model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME)
    else:
        model_path = ORIGINAL_WITH_TUNING
        print(f"Loading model from {model_path}")
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(model_path)

    # Load datasets
    df = pd.read_csv(PREDICTIONS_DATASET)
    dataset = SarcasmDataset(df, tokenizer)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # DataLoader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        collate_fn=data_collator,
        batch_size=EVAL_BATCH_SIZE
    )

    all_preds_list = []
    all_labels_list = []
    all_probs_list = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_probs_list.extend(probs.cpu().numpy())
            all_preds_list.extend(preds.cpu().numpy())
            all_labels_list.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs_list)
    all_preds = np.array(all_preds_list)
    all_labels = np.array(all_labels_list)

    accuracy  = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall    = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1        = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    macro_f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # confusion_matrix order:
    # [[tn, fp],
    #  [fn, tp]]
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

    print("\n=== Evaluation Results ===")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Macro F1  : {macro_f1:.4f}")
    print(f"Confusion Matrix:")
    print(f"  TN: {tn}  FP: {fp}")
    print(f"  FN: {fn}  TP: {tp}")

    headline_series = df['headline'].astype(str)

    results_df = df.copy().reset_index(drop=True)
    results_df['Probability of non sarcastic'] = np.round(all_probs[:, 0], 6)
    results_df['Probability of sarcastic']     = np.round(all_probs[:, 1], 6)
    results_df['Confidence']                   = np.round(np.max(all_probs, axis=1), 6)
    results_df['Predicted is sarcastic']       = all_preds.astype(int)
    results_df['Actual label']                 = all_labels.astype(int)
    results_df['Is correct?']                  = (all_preds == all_labels).astype(int)
    results_df['False +ve']                    = ((all_preds == 1) & (all_labels == 0)).astype(int)
    results_df['False -ve']                    = ((all_preds == 0) & (all_labels == 1)).astype(int)
    results_df['Approximate token length']     = headline_series.str.split().str.len()
    results_df['Is exclamation?']              = headline_series.str.endswith('!').astype(int)
    results_df['Is question?']                 = headline_series.str.endswith('?').astype(int)
    results_df['Is full stop?']                = headline_series.str.endswith('.').astype(int)
    results_df['Text length']                  = headline_series.str.len()

    results_df = results_df.rename(columns={
        'headline':     'Headline',
        'article_link': 'Article_Link',
    })

    predictions_csv_path = f'outputs/predictions/original_with_tuning_diagnostic_val_predictions.csv'
    os.makedirs(os.path.dirname(predictions_csv_path), exist_ok=True)
    results_df.to_csv(predictions_csv_path, index=False)
    print(f"\nPredictions saved to: {predictions_csv_path}")

if __name__ == '__main__':
    evaluation(
        pretrained=False
    )