"""Shared model loading and inference helpers.

This module centralizes the logic for:

1. Resolving a mode name to either a local checkpoint directory or the
   HuggingFace ``bert-base-uncased`` model ID.
2. Loading a tokenizer/model pair onto the appropriate device.
3. Running single-example or batched prediction with the same tokenization
   behavior used during evaluation.

Keeping inference in one place reduces the risk of train/evaluate/predict
drifting apart after future refactors.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from .config import (
    BATCH_SIZE,
    DEFAULT_BERT_MODE,
    MAX_LENGTH,
    PAD_TO_MULTIPLE_OF,
    PRETRAINED_MODE_MODEL_NAMES,
    get_mode_model_dir,
)
from .dataset import build_dataset_from_frame


def resolve_bert_model_reference(mode: str | None = None, model_path: str | Path | None = None) -> str | Path:
    """Resolve a caller-supplied mode or explicit path to a model reference.

    ``model_path`` takes precedence over ``mode``. Pretrained modes resolve to
    public HuggingFace model IDs, while all fine-tuned modes resolve to a local
    checkpoint directory under ``outputs/models``.
    """
    if model_path is not None:
        return model_path

    selected_mode = mode or DEFAULT_BERT_MODE
    if selected_mode in PRETRAINED_MODE_MODEL_NAMES:
        return PRETRAINED_MODE_MODEL_NAMES[selected_mode]

    mode_model_dir = get_mode_model_dir(selected_mode)
    if mode_model_dir.exists():
        return mode_model_dir

    raise FileNotFoundError(
        f"No saved BERT model found for mode '{selected_mode}'. Expected a checkpoint under {mode_model_dir}."
    )


def load_sequence_classifier(
    model_reference: str | Path,
    device: torch.device | None = None,
    use_fast: bool = True,
):
    """Load a sequence-classification model and tokenizer onto the target device."""
    if isinstance(model_reference, Path) and not model_reference.exists():
        raise FileNotFoundError(f"BERT model directory not found: {model_reference}")
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_reference, use_fast=use_fast)
    model = AutoModelForSequenceClassification.from_pretrained(model_reference)
    model.to(resolved_device).eval()
    return tokenizer, model, resolved_device


def predict_single_with_components(
    text: str,
    tokenizer,
    model,
    device: torch.device,
    max_length: int = MAX_LENGTH,
) -> tuple[int, float]:
    """Predict a single text example using already-loaded model components."""
    encoded = tokenizer(text, truncation=True, max_length=max_length, padding=False, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        probabilities = torch.softmax(model(**encoded).logits, dim=-1)[0]
    return int(torch.argmax(probabilities).item()), float(probabilities[1].item())


def predict_single(
    text: str,
    model_reference: str | Path,
    max_length: int = MAX_LENGTH,
) -> tuple[int, float]:
    """Convenience wrapper for one-off single-example prediction."""
    tokenizer, model, device = load_sequence_classifier(model_reference=model_reference)
    return predict_single_with_components(text, tokenizer, model, device, max_length=max_length)


def predict_batches(
    frame: pd.DataFrame,
    model_reference: str | Path,
    max_length: int = MAX_LENGTH,
    batch_size: int = BATCH_SIZE,
    progress_description: str | None = None,
    use_domain_context: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Run batched inference and return predictions, probabilities, and token lengths.

    Returns
    -------
    tuple
        ``(predictions, probabilities, token_lengths)`` where ``predictions`` is
        the argmax class label, ``probabilities`` is the softmax output for both
        classes, and ``token_lengths`` stores each example's pre-padding token
        count for downstream reporting.
    """
    tokenizer, model, device = load_sequence_classifier(model_reference=model_reference)
    dataset = build_dataset_from_frame(
        frame,
        tokenizer=tokenizer,
        max_length=max_length,
        require_labels=False,
        use_domain_context=use_domain_context,
    )
    token_lengths = dataset.token_lengths
    # Padding to a multiple of 8 is useful on CUDA for Tensor Core alignment,
    # but offers no real benefit on CPU inference.
    pad_to_multiple_of = PAD_TO_MULTIPLE_OF if torch.cuda.is_available() else None
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple_of)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    all_predictions = []
    all_probabilities = []
    batch_iterator = tqdm(dataloader, desc=progress_description, leave=False) if progress_description else dataloader
    with torch.inference_mode():
        for batch in batch_iterator:
            batch = {key: value.to(device) for key, value in batch.items()}
            probabilities = torch.softmax(model(**batch).logits, dim=-1)
            all_predictions.append(torch.argmax(probabilities, dim=-1).cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())

    return np.concatenate(all_predictions), np.concatenate(all_probabilities), token_lengths
