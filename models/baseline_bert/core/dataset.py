"""Data loading and preprocessing layer for the BERT sarcasm detection pipeline."""

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import (
    DATASET_COLUMN,
    FILE_SOURCE_COLUMN,
    INDEX_COLUMN,
    LABEL_COLUMN,
    LINK_COLUMN,
    PROCESSED_DATA_DIR,
    TEXT_COLUMN,
)


_COLUMN_NORM_RE = re.compile(r"[^a-z0-9]+")
_WWW_RE = re.compile(r"^www\.")


def extract_domain(url: str) -> str:
    """Return the bare registrable domain label from a URL."""
    try:
        netloc = urlparse(str(url)).netloc.lower()
        netloc = _WWW_RE.sub("", netloc)
        parts = [part for part in netloc.split(".") if part]
        if not parts:
            return ""
        if len(parts) >= 2:
            return parts[-2]
        return parts[0]
    except Exception:
        return ""


COLUMN_ALIASES = {
    "index": INDEX_COLUMN,
    "article_link": LINK_COLUMN,
    "articlelink": LINK_COLUMN,
    "headline": TEXT_COLUMN,
    "is_sarcastic": LABEL_COLUMN,
    "actual_label": LABEL_COLUMN,
    "actual_labels": LABEL_COLUMN,
    "actuallabel": LABEL_COLUMN,
    "is_sarcastic_actual_label": LABEL_COLUMN,
    "dataset": DATASET_COLUMN,
    "file_source": FILE_SOURCE_COLUMN,
    "filesource": FILE_SOURCE_COLUMN,
}


def normalize_column_name(column_name: str) -> str:
    return _COLUMN_NORM_RE.sub("_", str(column_name).strip().lower()).strip("_")


def resolve_split_path(split_or_path: str) -> Path:
    candidate = Path(split_or_path)
    if candidate.exists():
        return candidate
    csv_candidate = PROCESSED_DATA_DIR / f"{split_or_path}.csv"
    if csv_candidate.exists():
        return csv_candidate
    return PROCESSED_DATA_DIR / f"{split_or_path}.jsonl"


def infer_dataset_name(split_or_path: str | Path) -> str:
    path = Path(split_or_path)
    if path.suffix:
        return path.stem
    return str(split_or_path)


def _read_json_records(file_path: Path) -> pd.DataFrame:
    try:
        return pd.read_json(file_path, lines=True)
    except ValueError:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.DataFrame.from_records(payload)
        return pd.DataFrame(payload)


def _read_dataframe(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    if suffix in {".json", ".jsonl"}:
        return _read_json_records(file_path)
    raise ValueError(f"Unsupported file format: {file_path}")


def _canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for column_name in frame.columns:
        normalized_name = normalize_column_name(column_name)
        canonical_name = COLUMN_ALIASES.get(normalized_name)
        if canonical_name:
            rename_map[column_name] = canonical_name
    return frame.rename(columns=rename_map).copy()


def prepare_input_frame(frame: pd.DataFrame, dataset_name: str | None = None) -> pd.DataFrame:
    prepared = _canonicalize_columns(frame)
    if TEXT_COLUMN not in prepared.columns:
        raise ValueError(f"Input data must contain a '{TEXT_COLUMN}' column.")

    prepared[TEXT_COLUMN] = prepared[TEXT_COLUMN].astype(str).str.strip()
    prepared = prepared[prepared[TEXT_COLUMN] != ""].reset_index(drop=True)

    if INDEX_COLUMN not in prepared.columns:
        prepared[INDEX_COLUMN] = range(len(prepared))
    if LINK_COLUMN not in prepared.columns:
        prepared[LINK_COLUMN] = ""
    if DATASET_COLUMN not in prepared.columns:
        prepared[DATASET_COLUMN] = dataset_name or "unknown"
    else:
        prepared[DATASET_COLUMN] = prepared[DATASET_COLUMN].fillna(dataset_name or "unknown")

    if LABEL_COLUMN in prepared.columns:
        prepared[LABEL_COLUMN] = pd.to_numeric(prepared[LABEL_COLUMN], errors="coerce").astype("Int64")
    else:
        prepared[LABEL_COLUMN] = pd.Series([pd.NA] * len(prepared), dtype="Int64")

    return prepared


def load_input_dataframe(split_or_path: str, dataset_name: str | None = None) -> pd.DataFrame:
    resolved_path = resolve_split_path(split_or_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Could not locate input data: {split_or_path}")

    dataset_label = dataset_name or infer_dataset_name(resolved_path if Path(split_or_path).suffix else split_or_path)
    frame = _read_dataframe(resolved_path)
    return prepare_input_frame(frame, dataset_name=dataset_label)


class SarcasmDataset(Dataset):
    """PyTorch Dataset for tokenized sarcasm detection headlines."""

    def __init__(self, texts, labels, tokenizer, max_length: int, text_pairs=None):
        texts_list = list(texts)
        pairs_list = list(text_pairs) if text_pairs is not None else None
        encoded = tokenizer(
            texts_list,
            text_pair=pairs_list,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.input_ids = [torch.tensor(ids, dtype=torch.long) for ids in encoded["input_ids"]]
        self.attention_mask = [torch.tensor(mask, dtype=torch.long) for mask in encoded["attention_mask"]]
        raw_token_type_ids = encoded.get("token_type_ids")
        self.token_type_ids = (
            [torch.tensor(token_type_ids, dtype=torch.long) for token_type_ids in raw_token_type_ids]
            if raw_token_type_ids is not None
            else None
        )
        self.labels = (
            [torch.tensor(int(label), dtype=torch.long) for label in labels]
            if labels is not None
            else None
        )
        self.token_lengths = [len(ids) for ids in self.input_ids]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int):
        item = {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[index]
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item


def build_dataset_from_frame(
    frame: pd.DataFrame,
    tokenizer,
    max_length: int,
    require_labels: bool = True,
    use_domain_context: bool = False,
) -> SarcasmDataset:
    labels = None
    if LABEL_COLUMN in frame.columns and not frame[LABEL_COLUMN].isna().any():
        labels = frame[LABEL_COLUMN].astype(int).tolist()
    elif require_labels:
        raise ValueError("Labels are required for this dataset but were missing from the provided frame.")

    text_pairs = None
    if use_domain_context and LINK_COLUMN in frame.columns:
        text_pairs = frame[LINK_COLUMN].apply(extract_domain).tolist()

    return SarcasmDataset(
        texts=frame[TEXT_COLUMN].tolist(),
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length,
        text_pairs=text_pairs,
    )

