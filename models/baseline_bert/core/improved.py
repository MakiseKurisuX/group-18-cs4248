"""Improved-model helpers: topic balancing, contrastive loss, and special splits.

This module holds the pieces that are unique to the improved BERT variants:

- supervised contrastive loss on CLS embeddings
- topic-balanced sampling weights built from TF-IDF + KMeans clusters
- the custom Trainer subclass that combines cross-entropy and contrastive loss
- augmentation-specific data loading and persistence helpers
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import WeightedRandomSampler
from transformers import Trainer

from .artifacts import save_csv
from .config import (
    AUGMENTATION_CANDIDATES_PATH,
    AUGMENTED_TRAIN_PATH,
    BERT_MODEL_NAME,
    LABEL_COLUMN,
    SEED,
    TEXT_COLUMN,
)
from .dataset import load_input_dataframe
from .training import get_default_hyperparameters, log_step, run_training


CONTRASTIVE_WEIGHT = 0.1
CONTRASTIVE_TEMPERATURE = 0.07
N_TOPIC_CLUSTERS = 8


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = CONTRASTIVE_TEMPERATURE,
) -> torch.Tensor:
    """Compute supervised contrastive loss over normalized CLS embeddings."""
    batch_size = embeddings.size(0)
    device = embeddings.device

    similarities = torch.mm(embeddings, embeddings.T) / temperature
    similarities = similarities - similarities.detach().max(dim=1, keepdim=True).values

    not_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=device)
    same_label = (labels.unsqueeze(0) == labels.unsqueeze(1)) & not_diagonal
    has_positive = same_label.any(dim=1)
    if not has_positive.any():
        return embeddings.new_zeros(())

    exp_similarities = torch.exp(similarities) * not_diagonal.float()
    numerator = (exp_similarities * same_label.float()).sum(dim=1)
    denominator = exp_similarities.sum(dim=1)
    loss = -torch.log(numerator[has_positive] / (denominator[has_positive] + 1e-8))
    return loss.mean()


def build_topic_balanced_weights(
    texts: list,
    labels: list,
    n_clusters: int = N_TOPIC_CLUSTERS,
    seed: int = SEED,
) -> np.ndarray:
    """Assign inverse-frequency sampling weights within topic/label buckets.

    The intent is to reduce over-sampling of large topical clusters while still
    preserving label balance inside each topic bucket.
    """
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    features = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_ids = kmeans.fit_predict(features)

    label_array = np.array(labels)
    weights = np.zeros(len(texts), dtype=np.float64)
    for cluster_id in range(n_clusters):
        for label in (0, 1):
            mask = (cluster_ids == cluster_id) & (label_array == label)
            count = int(mask.sum())
            if count > 0:
                weights[mask] = 1.0 / count

    total = weights.sum()
    if total > 0:
        weights /= total

    log_step(
        f"Topic-balanced weights built: {n_clusters} clusters, min_w={weights.min():.2e}, max_w={weights.max():.2e}"
    )
    return weights


class ContrastiveTrainer(Trainer):
    """Trainer that adds supervised contrastive loss and optional weighted sampling."""
    def __init__(
        self,
        contrastive_weight: float = CONTRASTIVE_WEIGHT,
        contrastive_temperature: float = CONTRASTIVE_TEMPERATURE,
        topic_balanced_weights: np.ndarray | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.topic_balanced_weights = topic_balanced_weights

    def _get_train_sampler(self, dataset=None):
        if self.topic_balanced_weights is None:
            return super()._get_train_sampler(dataset)

        weights = torch.tensor(self.topic_balanced_weights, dtype=torch.float)
        return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Blend cross-entropy with supervised contrastive loss during training."""
        labels = inputs.pop("labels")
        need_hidden = model.training
        outputs = model(**inputs, output_hidden_states=need_hidden)
        logits = outputs.logits

        label_smoothing = getattr(self.args, "label_smoothing_factor", 0.0)
        ce_loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)

        if need_hidden:
            cls_embedding = outputs.hidden_states[-1][:, 0, :]
            cls_embedding = F.normalize(cls_embedding, dim=-1)
            contrastive_loss = supervised_contrastive_loss(
                cls_embedding,
                labels,
                temperature=self.contrastive_temperature,
            )
            loss = ce_loss + self.contrastive_weight * contrastive_loss
        else:
            loss = ce_loss

        return (loss, outputs) if return_outputs else loss


def get_improved_default_hyperparameters() -> dict:
    """Extend baseline defaults with improved-mode hyperparameters."""
    hyperparameters = get_default_hyperparameters()
    hyperparameters.update(
        {
            "contrastive_weight": CONTRASTIVE_WEIGHT,
            "contrastive_temperature": CONTRASTIVE_TEMPERATURE,
            "n_topic_clusters": N_TOPIC_CLUSTERS,
        }
    )
    return hyperparameters


def get_improved_large_default_hyperparameters() -> dict:
    """Return safer default hyperparameters for ``bert-large-uncased``.

    The large backbone is substantially more memory-hungry than
    ``bert-base-uncased``, so the physical batch size is reduced and gradient
    accumulation is increased to preserve a similar effective batch size.
    """
    hyperparameters = get_improved_default_hyperparameters()
    hyperparameters.update(
        {
            "batch_size": 8,
            "grad_accum": 2,
        }
    )
    return hyperparameters


def build_improved_trial_hyperparameters(trial, tuning_grid: dict) -> dict:
    """Extend the baseline Optuna search space with improved-only knobs."""
    from .tuning import build_trial_hyperparameters

    hyperparameters = build_trial_hyperparameters(trial, tuning_grid)
    cw_min, cw_max = sorted(float(value) for value in tuning_grid["contrastive_weight"])
    ct_min, ct_max = sorted(float(value) for value in tuning_grid["contrastive_temperature"])
    hyperparameters["contrastive_weight"] = float(trial.suggest_float("contrastive_weight", cw_min, cw_max))
    hyperparameters["contrastive_temperature"] = float(
        trial.suggest_float("contrastive_temperature", ct_min, ct_max)
    )
    hyperparameters["n_topic_clusters"] = int(
        trial.suggest_categorical("n_topic_clusters", [int(value) for value in tuning_grid["n_topic_clusters"]])
    )
    return hyperparameters


def train_improved_model(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    hyperparameters: dict,
    output_dir,
    save_artifacts: bool,
    model_name: str = BERT_MODEL_NAME,
    trial=None,
    tokenizer=None,
    use_domain_context: bool = False,
) -> dict:
    """Train the improved model using contrastive loss and topic-balanced sampling."""
    contrastive_weight = float(hyperparameters.get("contrastive_weight", CONTRASTIVE_WEIGHT))
    contrastive_temperature = float(
        hyperparameters.get("contrastive_temperature", CONTRASTIVE_TEMPERATURE)
    )
    n_topic_clusters = int(hyperparameters.get("n_topic_clusters", N_TOPIC_CLUSTERS))
    topic_weights = build_topic_balanced_weights(
        texts=train_frame[TEXT_COLUMN].tolist(),
        labels=train_frame[LABEL_COLUMN].astype(int).tolist(),
        n_clusters=n_topic_clusters,
    )

    return run_training(
        train_frame=train_frame,
        val_frame=val_frame,
        hyperparameters=hyperparameters,
        output_dir=output_dir,
        save_artifacts=save_artifacts,
        model_name=model_name,
        trial=trial,
        tokenizer=tokenizer,
        use_domain_context=use_domain_context,
        trainer_class=ContrastiveTrainer,
        trainer_init_kwargs={
            "contrastive_weight": contrastive_weight,
            "contrastive_temperature": contrastive_temperature,
            "topic_balanced_weights": topic_weights,
        },
        extra_log_fields={
            "contrastive_weight": contrastive_weight,
            "contrastive_temperature": contrastive_temperature,
            "n_topic_clusters": n_topic_clusters,
        },
        run_label="improved training run",
    )


def load_augmented_candidates():
    """Load augmentation candidates produced by the external augmentation pipeline."""
    if not AUGMENTATION_CANDIDATES_PATH.exists():
        raise FileNotFoundError(
            f"Augmentation candidates not found at {AUGMENTATION_CANDIDATES_PATH}. "
            "Run the data augmentation pipeline first to generate this file."
        )
    return load_input_dataframe(str(AUGMENTATION_CANDIDATES_PATH), dataset_name="augmentation_candidates")


def persist_augmented_train(frame: pd.DataFrame) -> None:
    """Persist the merged augmented training split for inspection and reuse."""
    save_csv(frame, AUGMENTED_TRAIN_PATH)
