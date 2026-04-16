# Standard libraries
import json
import os
from pathlib import Path

# Custom libraries
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score

# Local imports
from config import EVAL_BATCH_SIZE, MODEL_NAME
from dataset import SarcasmDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MASTER_COPY_DEDUP_V2_PATH = PROJECT_ROOT / 'data' / 'processed' / 'master' / 'master_copy_dedup_v2.csv'
ORIGINAL_TEST_PATH = PROJECT_ROOT / 'data' / 'processed' / 'original' / 'test.csv'
DIAGNOSTIC_VAL_PATH = PROJECT_ROOT / 'models' / 'baseline_roberta' / 'data' / 'diagnostic_val.csv'

# Available models
ORIGINAL_NO_TUNING = 'outputs/checkpoints/original_no_tuning'
ORIGNAL_WITH_TUNING = 'outputs/checkpoints/original_with_tuning'
IMPROVED_NO_TUNING = 'outputs/checkpoints/improved_no_tuning'
IMPROVED_WITH_TUNING = 'outputs/checkpoints/improved_with_tuning'
AUGMENTED_WITH_TUNING_RD1 = 'outputs/checkpoints/augmented_with_tuning_rd1'
AUGMENTED_WITH_TUNING_RD2 = 'outputs/checkpoints/augmented_with_tuning_rd2'
AUGMENTED_WITH_TUNING_RD3 = 'outputs/checkpoints/augmented_with_tuning_rd3'

def evaluation(
    pretrained,
    mode
):
    if pretrained:
        print(f"Running pretrained model {MODEL_NAME}")
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
        model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME)
    else:
        model_path = IMPROVED_WITH_TUNING
        print(f"Loading model from {model_path}")
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(model_path)

    # Load datasets
    master_copy_dedup_v2_df = pd.read_csv(MASTER_COPY_DEDUP_V2_PATH)
    master_copy_dedup_v2_dataset = SarcasmDataset(master_copy_dedup_v2_df, tokenizer)
    original_test_df = pd.read_csv(ORIGINAL_TEST_PATH)
    original_test_dataset = SarcasmDataset(original_test_df, tokenizer)
    diagnostic_val_df = pd.read_csv(DIAGNOSTIC_VAL_PATH)
    diagnostic_val_dataset = SarcasmDataset(diagnostic_val_df, tokenizer)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # DataLoader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    master_copy_dedup_v2_dataloader = DataLoader(
        master_copy_dedup_v2_dataset,
        collate_fn=data_collator,
        batch_size=EVAL_BATCH_SIZE
    )

    original_test_dataloader = DataLoader(
        original_test_dataset,
        collate_fn=data_collator,
        batch_size=EVAL_BATCH_SIZE
    )

    diagnostic_val_dataloader = DataLoader(
        diagnostic_val_dataset,
        collate_fn=data_collator,
        batch_size=EVAL_BATCH_SIZE
    )

    dataloaders = [master_copy_dedup_v2_dataloader, original_test_dataloader, diagnostic_val_dataloader]
    dfs = [master_copy_dedup_v2_df, original_test_df, diagnostic_val_df]
    datasets = ['master_copy_dedup_v2', 'original_test', 'diagnostic_val']

    for df, dataloader, dataset in zip(dfs, dataloaders, datasets):
        METRICS_JSON_PATH = f'outputs/metrics/{mode}_{dataset}_metrics.json'

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

        print(f"\n=== Test Set Results for {dataset} ===")
        print(f"Accuracy  : {accuracy:.4f}") 
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}") 

        results_df = df.copy().reset_index(drop=True)

        results_df["prob_non_sarcastic"] = all_probs[:, 0]
        results_df["prob_sarcastic"] = all_probs[:, 1]
        results_df["confidence"] = np.max(all_probs, axis=1)
        results_df["predicted_is_sarcastic"] = all_preds
        results_df["actual_label"] = all_labels
        results_df["is_correct"] = (all_preds == all_labels).astype(int)
        results_df["false_positive"] = ((all_preds == 1) & (all_labels == 0)).astype(int)
        results_df["false_negative"] = ((all_preds == 0) & (all_labels == 1)).astype(int)

        metrics_dict = {
            "model_type": 'roberta',
            "mode": mode,
            "dataset": dataset,
            "num_examples": int(len(all_labels)),
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "confusion_matrix": [
                [int(tn), int(fp)],
                [int(fn), int(tp)],
            ],
        }

        os.makedirs(os.path.dirname(METRICS_JSON_PATH), exist_ok=True)
        with open(METRICS_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=2)

if __name__ == '__main__':
    evaluation(
        pretrained=False,
        mode='improved_with_tuning'
    )