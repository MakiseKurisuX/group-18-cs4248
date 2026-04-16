"""
Standalone error diagnostic script for sarcasm detection models.

Reproduces the full error_diagnostic_pipeline.ipynb analysis and saves all
outputs (CSV, PNG plots, cluster report text) to disk without any Jupyter
dependency.

The script is model-agnostic: pass --model to point it at any model folder
under models/. Predictions, model weights, and output results are all
resolved relative to that folder.

Usage
-----
  # Single mode / single dataset (defaults to baseline_bert)
  python error_diagnostic.py --mode original_with_tuning --dataset original_test

  # Explicit model
  python error_diagnostic.py --model baseline_bert --mode original_with_tuning --dataset original_test

  # All discovered modes for a model on a specific dataset
  python error_diagnostic.py --model baseline_bert --mode all --dataset original_test

  # All modes, all datasets
  python error_diagnostic.py --model baseline_bert --mode all --dataset all

  # Custom output directory
  python error_diagnostic.py --model baseline_bert --mode all --dataset all --output-dir my_results
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModel, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Constants ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # one level up from error_diagnostic/
_SCRIPT_DIR  = Path(__file__).resolve().parent
DATASET_JSON = PROJECT_ROOT / "data" / "src" / "Sarcasm_Headlines_Dataset_v2.json"

DEFAULT_MODEL = "baseline_bert"
ALL_DATASETS  = ["original_test", "master_copy_dedup_v2"]


# ── Path helpers ───────────────────────────────────────────────────────────────

def _pred_root(model: str) -> Path:
    """Directory that holds {mode}_{dataset}_predictions.csv for a model."""
    return PROJECT_ROOT / "models" / model / "outputs" / "predictions"


def _model_root(model: str) -> Path:
    """Directory that holds per-mode checkpoint sub-folders for a model."""
    return PROJECT_ROOT / "models" / model / "outputs" / "models"


def _default_output_dir(model: str) -> Path:
    """Default root for error-diagnostic results, organised by model."""
    return _SCRIPT_DIR / "results" / model


def _discover_modes(pred_root: Path, datasets: list[str]) -> list[str]:
    """Scan the predictions directory and return all mode names that have at
    least one predictions CSV for any of the supplied dataset names.

    This replaces a hardcoded ALL_MODES list so the script works for any
    model without requiring code changes.
    """
    seen: set[str] = set()
    for dataset in datasets:
        suffix = f"_{dataset}_predictions.csv"
        for f in sorted(pred_root.glob(f"*{suffix}")):
            seen.add(f.name[: -len(suffix)])
    return sorted(seen)


def _discover_datasets(pred_root: Path, modes: list[str]) -> list[str]:
    """For a set of known modes, return every dataset name that has a
    predictions CSV in the predictions directory.

    Works by stripping the known mode prefix and the _predictions.csv
    suffix from each matching filename, so it handles any dataset name
    including custom ones like 'diagnostic_val'.
    """
    seen: set[str] = set()
    for mode in modes:
        prefix = f"{mode}_"
        suffix = "_predictions.csv"
        for f in sorted(pred_root.glob(f"{mode}_*_predictions.csv")):
            name = f.name
            if name.startswith(prefix) and name.endswith(suffix):
                seen.add(name[len(prefix): -len(suffix)])
    return sorted(seen)


def _resolve_embedding_model(mode: str, model_root: Path) -> tuple[str, bool]:
    """Return (model_reference, local_files_only) for a given mode.

    Calibrated variants (e.g. improved_no_tuning_calibrated) share their
    weights with the base mode (improved_no_tuning), so we walk up the
    underscore-delimited segments until we find an existing model directory.

    For modes named "pretrained" the HuggingFace hub name is returned
    directly; all other modes are expected to have a local checkpoint.
    """
    if mode == "pretrained":
        return "bert-base-uncased", False

    # Try the full mode name first, then progressively strip the last segment.
    segments = mode.split("_")
    for n in range(len(segments), 0, -1):
        candidate = "_".join(segments[:n])
        candidate_dir = model_root / candidate
        if candidate_dir.exists():
            return str(candidate_dir.resolve()), True

    # Nothing found — return the full path so the caller gets a clear error.
    return str((model_root / mode).resolve()), True

_COLUMN_MAP = {
    "Headline":                     "headline",
    "Article_Link":                 "article_link",
    "Actual label":                 "true_label",
    "Predicted is sarcastic":       "pred_label",
    "Confidence":                   "confidence",
    "Is correct?":                  "correct",
    "False +ve":                    "false_positive",
    "False -ve":                    "false_negative",
    "Approximate token length":     "token_len_approx",
    "Is exclamation?":              "exclamation",
    "Is question?":                 "question",
    "Probability of non sarcastic": "prob_non_sarcastic",
    "Probability of sarcastic":     "prob_sarcastic",
}


# ── Embedding helpers ──────────────────────────────────────────────────────────

def _encode_texts(texts: list[str], tokenizer, model, device: str, batch_size: int = 64) -> np.ndarray:
    """Mean-pool + L2-normalise BERT hidden states for a list of texts."""
    vectors = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            mask    = encoded["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            summed  = (outputs.last_hidden_state * mask).sum(dim=1)
            denom   = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled  = summed / denom
            vectors.append(F.normalize(pooled, p=2, dim=1).cpu().numpy())
    return np.vstack(vectors)


# ── Per-run pipeline ───────────────────────────────────────────────────────────

def run_diagnostic(
    mode: str,
    dataset_name: str,
    output_dir: Path,
    pred_root: Path,
    model_root: Path,
    k_neighbors: int,
    n_clusters: int,
    seed: int,
    batch_size: int,
) -> None:
    run_label   = f"{mode}_{dataset_name}"
    predictions_path = pred_root / f"{run_label}_predictions.csv"
    run_out_dir = output_dir / run_label
    run_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Mode: {mode}   Dataset: {dataset_name}")
    print(f"  Predictions : {predictions_path}")
    print(f"  Output dir  : {run_out_dir}")
    print(f"{'='*70}")

    if not predictions_path.exists():
        print(f"  [SKIP] Predictions file not found: {predictions_path}")
        return

    # ── 1. Load predictions ────────────────────────────────────────────────
    df_preds = pd.read_csv(predictions_path)
    df_preds = df_preds.rename(columns={k: v for k, v in _COLUMN_MAP.items() if k in df_preds.columns})

    if "error_type" not in df_preds.columns:
        df_preds["error_type"] = "correct"
        if "false_positive" in df_preds.columns:
            df_preds.loc[df_preds["false_positive"] == 1, "error_type"] = "false_positive"
        if "false_negative" in df_preds.columns:
            df_preds.loc[df_preds["false_negative"] == 1, "error_type"] = "false_negative"

    errors_df = df_preds[df_preds["correct"] == 0].reset_index(drop=True)
    print(f"  Loaded {len(df_preds)} predictions — {len(errors_df)} errors to analyse.")

    if len(errors_df) == 0:
        print("  [SKIP] No errors found; nothing to analyse.")
        return

    # ── 2. Load reference dataset ──────────────────────────────────────────
    df_ref = pd.read_json(DATASET_JSON, lines=True)
    print(f"  Loaded {len(df_ref)} reference samples.")

    # ── 3. Load model + compute embeddings ────────────────────────────────
    # Calibrated variants (e.g. improved_no_tuning_calibrated) reuse the
    # weights of their base mode — _resolve_embedding_model handles the lookup.
    embedding_model_ref, local_files_only = _resolve_embedding_model(mode, model_root)
    if local_files_only and not Path(embedding_model_ref).exists():
        print(f"  [SKIP] Local checkpoint not found: {embedding_model_ref}")
        return

    print(f"  Loading model: {embedding_model_ref} ...")
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_ref, local_files_only=local_files_only, use_fast=True)
    model     = AutoModel.from_pretrained(embedding_model_ref, local_files_only=local_files_only).to(device)

    print("  Encoding reference dataset ...")
    ref_embeddings   = _encode_texts(df_ref["headline"].astype(str).tolist(), tokenizer, model, device, batch_size)
    print("  Encoding error samples ...")
    error_embeddings = _encode_texts(errors_df["headline"].astype(str).tolist(), tokenizer, model, device, batch_size)

    # Free GPU memory early
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── 4. KNN search ─────────────────────────────────────────────────────
    print(f"  Fitting KNN (k={k_neighbors}) ...")
    knn = NearestNeighbors(n_neighbors=k_neighbors, metric="cosine")
    knn.fit(ref_embeddings)
    distances, indices = knn.kneighbors(error_embeddings)

    # ── 5. Heuristic angle categorisation ─────────────────────────────────
    records = []
    for i, row in errors_df.iterrows():
        neighbor_idx  = indices[i]
        neighbor_dist = distances[i]
        neighbors     = df_ref.iloc[neighbor_idx]

        sarcastic_ratio = (neighbors["is_sarcastic"] == 1).mean()
        closest_dist    = neighbor_dist[0]
        closest_label   = int(neighbors.iloc[0]["is_sarcastic"])

        is_fp       = row["error_type"] == "false_positive"
        actual_label = row["true_label"]
        pred_label  = row["pred_label"]
        conf        = row["confidence"]
        token_len   = row.get("token_len_approx", len(str(row["headline"]).split()))
        has_excl    = bool(row.get("exclamation", "!" in str(row["headline"])))
        has_quest   = bool(row.get("question",    "?" in str(row["headline"])))
        has_full    = "." in str(row["headline"])

        is_outlier        = closest_dist > 0.15
        is_conflict       = (not is_outlier) and (closest_label != actual_label)
        is_systematic_bias = (not is_outlier) and (closest_label == pred_label) and (conf > 0.90)
        is_ambiguous      = (conf < 0.60) and (token_len < 6)
        is_syntactic_bias = is_fp and (has_excl or has_quest) and (closest_dist > 0.10)
        is_mixed          = 0.3 <= sarcastic_ratio <= 0.7

        if is_outlier:
            issue_cat = "Angle 2: Outlier / Zero-Shot Zone"
        elif is_ambiguous:
            issue_cat = "Angle 4: Low-Signal Ambiguity (Too short/vague)"
        elif is_systematic_bias:
            issue_cat = "Angle 3: Systematic Bias / Learned Spurious Rule"
        elif is_syntactic_bias:
            issue_cat = "Angle 5: Punctuation Spurious Correlation"
        elif is_conflict:
            issue_cat = "Angle 1: Label Conflict (Fuzzy Boundary)"
        elif is_mixed:
            issue_cat = "Angle 6: Mixed Neighborhood (Tangled Region)"
        else:
            issue_cat = "Uncategorized Error"

        records.append({
            "Index":                      row.name,
            "Article_Link":               row.get("article_link", ""),
            "Headline":                   row["headline"],
            "Dataset":                    dataset_name,
            "Probability of non sarcastic": row.get("prob_non_sarcastic", 1 - row.get("prob_sarcastic", 0)),
            "Probability of sarcastic":   row.get("prob_sarcastic", 0),
            "Confidence":                 conf,
            "Predicted is sarcastic":     pred_label,
            "Actual label":               actual_label,
            "Is correct?":                row["correct"],
            "False +ve":                  1 if is_fp else 0,
            "False -ve":                  0 if is_fp else 1,
            "Text length":                len(str(row["headline"])),
            "Approximate token length":   token_len,
            "Is exclamation?":            1 if has_excl else 0,
            "Is question?":               1 if has_quest else 0,
            "Is full stop?":              1 if has_full else 0,
            "Distance to closest neighbor": closest_dist,
            "Closest neighbor label":     closest_label,
            "Neighbor sarcastic ratio":   sarcastic_ratio,
            "Identified Category":        issue_cat,
        })

    results_df = pd.DataFrame(records)

    # ── 6. KMeans cluster analysis ─────────────────────────────────────────
    actual_k = min(n_clusters, len(errors_df))
    print(f"  Fitting KMeans (k={actual_k}) on {len(error_embeddings)} error embeddings ...")
    kmeans        = KMeans(n_clusters=actual_k, random_state=seed, n_init="auto")
    cluster_labels = kmeans.fit_predict(error_embeddings)

    errors_clustered          = errors_df.copy()
    errors_clustered["cluster_id"] = cluster_labels

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    tfidf.fit(errors_clustered["headline"].astype(str))
    feature_names = np.array(tfidf.get_feature_names_out())

    N_KEYWORDS = 5
    cluster_themes: dict[int, str] = {}
    for cid in range(actual_k):
        mask_c = errors_clustered["cluster_id"] == cid
        c_headlines = errors_clustered.loc[mask_c, "headline"].astype(str).tolist()
        if not c_headlines:
            cluster_themes[cid] = f"Cluster {cid} (empty)"
            continue
        tfidf_matrix = tfidf.transform(c_headlines)
        mean_scores  = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        top_idx      = mean_scores.argsort()[::-1][:N_KEYWORDS]
        keywords     = ", ".join(feature_names[top_idx])
        cluster_themes[cid] = f"C{cid}: {keywords}"

    errors_clustered["cluster_theme"] = errors_clustered["cluster_id"].map(cluster_themes)

    # Attach cluster columns to results
    results_df["Cluster ID"]    = cluster_labels
    results_df["Cluster Theme"] = errors_clustered["cluster_theme"].values

    # ── 7. Save CSV ────────────────────────────────────────────────────────
    csv_path = run_out_dir / "error_root_causes.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # ── 8. Cluster report text ─────────────────────────────────────────────
    report_lines = [
        f"MESO-LEVEL CLUSTER REPORT  (K={actual_k})",
        f"Mode: {mode}   Dataset: {dataset_name}",
        "=" * 70,
    ]
    N_SAMPLES = 4
    for cid in range(actual_k):
        mask_c = errors_clustered["cluster_id"] == cid
        sub    = errors_clustered[mask_c]
        n_fp   = (sub["error_type"] == "false_positive").sum()
        n_fn   = (sub["error_type"] == "false_negative").sum()
        report_lines += [
            "",
            "─" * 70,
            f"  {cluster_themes[cid]}",
            f"  Size: {len(sub)}  │  FP: {n_fp}  │  FN: {n_fn}",
            "─" * 70,
        ]
        samples = sub["headline"].sample(min(N_SAMPLES, len(sub)), random_state=seed)
        for j, headline in enumerate(samples, 1):
            report_lines.append(f"  [{j}] {headline}")
    report_lines += ["", "=" * 70]
    report_text = "\n".join(report_lines)

    report_path = run_out_dir / "cluster_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"  Saved: {report_path}")
    print(report_text)

    # ── 9. t-SNE projection ────────────────────────────────────────────────
    print("  Computing t-SNE (may take a minute) ...")
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30, max_iter=1000)
    all_embeddings = np.vstack([ref_embeddings, error_embeddings])
    all_2d = tsne.fit_transform(all_embeddings)
    n_ref  = len(ref_embeddings)
    ref_2d   = all_2d[:n_ref]
    error_2d = all_2d[n_ref:]

    is_fp_mask = errors_clustered["error_type"] == "false_positive"
    is_fn_mask = errors_clustered["error_type"] == "false_negative"
    fp_2d = error_2d[is_fp_mask.values]
    fn_2d = error_2d[is_fn_mask.values]

    # ── 10. Plot 1: Error distribution bar chart ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    import seaborn as sns
    order = results_df["Identified Category"].value_counts().index.tolist()
    sns.countplot(
        data=results_df,
        y="Identified Category",
        order=order,
        palette="viridis",
        hue="Identified Category",
        legend=False,
        ax=ax,
    )
    ax.set_title(f"Error Root Cause Distribution\n({mode} / {dataset_name})", fontsize=14)
    ax.set_xlabel("Number of Errors", fontsize=11)
    ax.set_ylabel("Root Cause Strategy / Angle", fontsize=11)
    plt.tight_layout()
    dist_path = run_out_dir / "error_distribution.png"
    fig.savefig(dist_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {dist_path}")

    # ── 11. Plot 2: t-SNE side-by-side (unclustered + FP/FN density) ──────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"t-SNE Error Analysis — {mode} / {dataset_name}", fontsize=14, fontweight="bold")

    axes[0].set_title("Unclustered Manifold", fontsize=13)
    axes[0].scatter(ref_2d[:, 0], ref_2d[:, 1], c="blue", alpha=0.3, s=10, edgecolors="none")
    axes[0].scatter(error_2d[:, 0], error_2d[:, 1], c="blue", alpha=0.9, s=15, edgecolors="none")
    axes[0].set_xlabel("t-SNE Dim 1"); axes[0].set_ylabel("t-SNE Dim 2")
    axes[0].tick_params(direction="in", top=True, right=True)

    axes[1].set_title("Error Density (FP / FN)", fontsize=13)
    axes[1].scatter(ref_2d[:, 0], ref_2d[:, 1], c="green", alpha=0.15, s=10, edgecolors="none", label="Reference")
    axes[1].scatter(fp_2d[:, 0], fp_2d[:, 1], c="red",  alpha=0.9, s=25, edgecolors="white", linewidth=0.5, label="False Positive")
    axes[1].scatter(fn_2d[:, 0], fn_2d[:, 1], c="blue", alpha=0.9, s=25, edgecolors="white", linewidth=0.5, label="False Negative")
    axes[1].set_xlabel("t-SNE Dim 1"); axes[1].set_ylabel("t-SNE Dim 2")
    axes[1].tick_params(direction="in", top=True, right=True)
    axes[1].legend(loc="best", fontsize=10)

    plt.tight_layout()
    tsne_path = run_out_dir / "tsne_analysis.png"
    fig.savefig(tsne_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {tsne_path}")

    # ── 12. Plot 3: KMeans clusters on t-SNE ──────────────────────────────
    cmap = matplotlib.colormaps["tab10"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Meso-Level Cluster Analysis — {mode} / {dataset_name}", fontsize=15, fontweight="bold")

    # Left: colour by cluster
    ax = axes[0]
    ax.set_title("Error Clusters (KMeans → t-SNE)", fontsize=13)
    ax.scatter(ref_2d[:, 0], ref_2d[:, 1], c="lightgray", s=6, alpha=0.25, edgecolors="none")
    scatter = ax.scatter(
        error_2d[:, 0], error_2d[:, 1],
        c=cluster_labels, cmap="tab10", vmin=0, vmax=actual_k - 1,
        s=35, alpha=0.9, edgecolors="white", linewidth=0.4,
    )
    plt.colorbar(scatter, ax=ax, label="Cluster ID", ticks=range(actual_k))
    ax.set_xlabel("t-SNE Dim 1"); ax.set_ylabel("t-SNE Dim 2")
    ax.tick_params(direction="in", top=True, right=True)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=cluster_themes[cid],
                   markerfacecolor=cmap(cid / max(actual_k - 1, 1)), markersize=8)
        for cid in range(actual_k)
    ]
    ax.legend(handles=handles, title="Cluster Theme", bbox_to_anchor=(1.02, 1),
              loc="upper left", fontsize=7.5, title_fontsize=9, framealpha=0.9)

    # Right: cluster + FP (■) / FN (▲) shape coding
    ax2 = axes[1]
    ax2.set_title("Clusters + Error Type  (■ FP  ▲ FN)", fontsize=13)
    ax2.scatter(ref_2d[:, 0], ref_2d[:, 1], c="lightgray", s=6, alpha=0.25, edgecolors="none")
    for cid in range(actual_k):
        c_mask = cluster_labels == cid
        fp_c   = c_mask & is_fp_mask.values
        fn_c   = c_mask & is_fn_mask.values
        colour = cmap(cid / max(actual_k - 1, 1))
        if fp_c.any():
            ax2.scatter(error_2d[fp_c, 0], error_2d[fp_c, 1],
                        color=colour, marker="s", s=40, alpha=0.85, edgecolors="white", linewidth=0.4)
        if fn_c.any():
            ax2.scatter(error_2d[fn_c, 0], error_2d[fn_c, 1],
                        color=colour, marker="^", s=50, alpha=0.85, edgecolors="white", linewidth=0.4)
    ax2.set_xlabel("t-SNE Dim 1"); ax2.set_ylabel("t-SNE Dim 2")
    ax2.tick_params(direction="in", top=True, right=True)
    fp_patch = plt.Line2D([0], [0], marker="s", color="w", label="False Positive",  markerfacecolor="gray", markersize=9)
    fn_patch = plt.Line2D([0], [0], marker="^", color="w", label="False Negative",  markerfacecolor="gray", markersize=9)
    ax2.legend(handles=[fp_patch, fn_patch], loc="lower right", fontsize=10)

    plt.tight_layout()
    cluster_path = run_out_dir / "cluster_analysis.png"
    fig.savefig(cluster_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {cluster_path}")

    print(f"\n  Done: {run_label}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the sarcasm detection error diagnostic pipeline and save all outputs to disk."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Model folder name under models/ (default: %(default)s). "
            "Determines where predictions and checkpoints are read from, "
            "and where results are written to."
        ),
    )
    parser.add_argument(
        "--mode",
        default="original_with_tuning",
        help=(
            "Experiment mode to diagnose, or 'all' to auto-discover every mode "
            "that has a predictions CSV in the model's outputs/predictions/ folder. "
            "Any string is accepted so custom suffixes work without code changes."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="original_test",
        help=(
            "Dataset name to analyse, or 'all' to auto-discover every dataset "
            "that has a predictions CSV for the selected mode(s). "
            "Any string is accepted — e.g. 'original_test', "
            "'master_copy_dedup_v2', 'diagnostic_val'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Root output directory. "
            "Default: error_diagnostic/results/{model}/ next to this script."
        ),
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=10,
        help="Number of nearest neighbours for KNN search (default: 10).",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=8,
        help="Number of KMeans clusters (default: 8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for t-SNE, KMeans, and cluster sampling (default: 42).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for model encoding (default: 64).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pred_root  = _pred_root(args.model)
    model_root = _model_root(args.model)

    # Resolve modes first — needed for dataset auto-discovery.
    if args.mode == "all":
        # Use ALL_DATASETS as a seed so _discover_modes has something to scan against;
        # the full dataset list is resolved afterwards.
        seed_datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]
        modes = _discover_modes(pred_root, seed_datasets)
        if not modes:
            print(f"[WARN] No predictions CSVs found under {pred_root} — nothing to run.")
            return
    else:
        modes = [args.mode]

    # Resolve datasets — 'all' discovers every dataset that has predictions for
    # the selected modes, including custom ones like 'diagnostic_val'.
    if args.dataset == "all":
        datasets = _discover_datasets(pred_root, modes)
        if not datasets:
            print(f"[WARN] No prediction files found for modes {modes} under {pred_root}.")
            return
    else:
        datasets = [args.dataset]

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args.model)

    print(f"Model            : {args.model}")
    print(f"Predictions dir  : {pred_root}")
    print(f"Model weights dir: {model_root}")
    print(f"Output directory : {output_dir}")
    print(f"Modes            : {modes}")
    print(f"Datasets         : {datasets}")
    print(f"K neighbors      : {args.k_neighbors}")
    print(f"N clusters       : {args.n_clusters}")
    print(f"Seed             : {args.seed}")

    for mode in modes:
        for dataset in datasets:
            run_diagnostic(
                mode=mode,
                dataset_name=dataset,
                output_dir=output_dir,
                pred_root=pred_root,
                model_root=model_root,
                k_neighbors=args.k_neighbors,
                n_clusters=args.n_clusters,
                seed=args.seed,
                batch_size=args.batch_size,
            )

    print(f"\nAll runs complete. Results in: {output_dir}")


if __name__ == "__main__":
    main()
