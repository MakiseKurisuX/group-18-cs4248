# Standard libraries
from pathlib import Path
import json
from typing import List

# Custom libraries
from torch.utils.data import Dataset
import torch

# Local imports
from config import MAX_LENGTH, MAX_LENGTH_CONTEXT

def load_json_dataset(paths: List[Path]) -> List:
    json_data = []

    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                json_data.append(json.loads(line))
    
    return json_data
    
class SarcasmDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=MAX_LENGTH):
        self.texts = df['headline'].tolist()
        self.labels = df['is_sarcastic'].tolist()
        self.encodings = tokenizer(
            self.texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx: int):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx])
        }

def build_model_input(row):
    headline = str(row.get("headline", "")).strip()
    description = str(row.get("description", "")).strip()

    if description and description.lower() not in {"nan", "none", ""}:
        return f"{headline} </s> {description}"

    return headline

class SarcasmDatasetWithContext(Dataset):
    def __init__(self, df, tokenizer, max_length=MAX_LENGTH_CONTEXT):
        self.texts = df.apply(build_model_input, axis=1).tolist()
        self.labels = df["is_sarcastic"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }