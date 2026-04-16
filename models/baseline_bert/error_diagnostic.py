"""
Standalone error diagnostic script for the baseline BERT pipeline.

This script reads prediction CSVs from models/baseline_bert/outputs/predictions,
loads the corresponding checkpoint from outputs/models when needed, and writes
all diagnostic artifacts under outputs/error_diagnostic_results.
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModel, AutoTokenizer

from core.config import (
    BERT_LARGE_MODEL_NAME,
    BERT_MODEL_NAME,
    ERROR_DIAGNOSTIC_RESULTS_DIR,
    MODEL_OUTPUT_DIR,
    PREDICTIONS_DIR,
    PROJECT_ROOT,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATASET_JSON = PROJECT_ROOT / "data" / "src" / "Sarcasm_Headlines_Dataset_v2.json"
ALL_DATASETS = ["original_test", "master_copy_dedup_v2", "diagnostic_val"]
PRETRAINED_MODE_MODEL_NAMES = {
    "pretrained": BERT_MODEL_NAME,
    "pretrained_large": BERT_LARGE_MODEL_NAME,
}

_COLUMN_MAP = {
    "Headline": "headline",
    "Article_Link": "article_link",
    "Actual label": "true_label",
    "Predicted is sarcastic": "pred_label",
    "Confidence": "confidence",
    "Is correct?": "correct",
    "False +ve": "false_positive",
    "False -ve": "false_negative",
    "Approximate token length": "token_len_approx",
    "Is exclamation?": "exclamation",
    "Is question?": "question",
    "Probability of non sarcastic": "prob_non_sarcastic",
    "Probability of sarcastic": "prob_sarcastic",
}


def _discover_modes(pred_root: Path, datasets: list[str]) -> list[str]:
    seen: set[str] = set()
    for dataset in datasets:
        suffix = f"_{dataset}_predictions.csv"
        for file_path in sorted(pred_root.glob(f"*{suffix}")):
            seen.add(file_path.name[: -len(suffix)])
    return sorted(seen)


def _discover_datasets(pred_root: Path, modes: list[str]) -> list[str]:
    seen: set[str] = set()
    for mode in modes:
        prefix = f"{mode}_"
        suffix = "_predictions.csv"
        for file_path in sorted(pred_root.glob(f"{mode}_*_predictions.csv")):
            name = file_path.name
            if name.startswith(prefix) and name.endswith(suffix):
                seen.add(name[len(prefix) : -len(suffix)])
    return sorted(seen)


def _resolve_embedding_model(mode: str) -> tuple[str, bool]:
    if mode in PRETRAINED_MODE_MODEL_NAMES:
        return PRETRAINED_MODE_MODEL_NAMES[mode], False

    segments = mode.split("_")
    for n_segments in range(len(segments), 0, -1):
        candidate = "_".join(segments[:n_segments])
        candidate_dir = MODEL_OUTPUT_DIR / candidate
        if candidate_dir.exists():
            return str(candidate_dir.resolve()), True

    return str((MODEL_OUTPUT_DIR / mode).resolve()), True


def _encode_texts(texts: list[str], tokenizer, model, device: str, batch_size: int = 64) -> np.ndarray:
    vectors = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            summed = (outputs.last_hidden_state * mask).sum(dim=1)
            denom = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = summed / denom
            vectors.append(F.normalize(pooled, p=2, dim=1).cpu().numpy())
    return np.vstack(vectors)


def run_diagnostic(
    mode: str,
    dataset_name: str,
    output_dir: Path,
    k_neighbors: int,
    n_clusters: int,
    seed: int,
    batch_size: int,
) -> None:
    run_label = f"{mode}_{dataset_name}"
    predictions_path = PREDICTIONS_DIR / f"{run_label}_predictions.csv"
    run_out_dir = output_dir / run_label
    run_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  Mode: {mode}   Dataset: {dataset_name}")
    print(f"  Predictions : {predictions_path}")
    print(f"  Output dir  : {run_out_dir}")
    print(f"{'=' * 70}")

    if not predictions_path.exists():
        print(f"  [SKIP] Predictions file not found: {predictions_path}")
        return

    if not DATASET_JSON.exists():
        print(f"  [SKIP] Reference dataset not found: {DATASET_JSON}")
        return

    df_preds = pd.read_csv(predictions_path)
    df_preds = df_preds.rename(columns={key: value for key, value in _COLUMN_MAP.items() if key in df_preds.columns})

    if "error_type" not in df_preds.columns:
        df_preds["error_type"] = "correct"
        if "false_positive" in df_preds.columns:
            df_preds.loc[df_preds["false_positive"] == 1, "error_type"] = "false_positive"
        if "false_negative" in df_preds.columns:
            df_preds.loc[df_preds["false_negative"] == 1, "error_type"] = "false_negative"

    errors_df = df_preds[df_preds["correct"] == 0].reset_index(drop=True)
    print(f"  Loaded {len(df_preds)} predictions - {len(errors_df)} errors to analyse.")

    if len(errors_df) == 0:
        print("  [SKIP] No errors found; nothing to analyse.")
        return

    df_ref = pd.read_json(DATASET_JSON, lines=True)
    print(f"  Loaded {len(df_ref)} reference samples.")

    embedding_model_ref, local_files_only = _resolve_embedding_model(mode)
    if local_files_only and not Path(embedding_model_ref).exists():
        print(f"  [SKIP] Local checkpoint not found: {embedding_model_ref}")
        return

    print(f"  Loading model: {embedding_model_ref}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        embedding_model_ref,
        local_files_only=local_files_only,
        use_fast=True,
    )
    model = AutoModel.from_pretrained(
        embedding_model_ref,
        local_files_only=local_files_only,
    ).to(device)

    print("  Encoding reference dataset...")
    ref_embeddings = _encode_texts(df_ref["headline"].astype(str).tolist(), tokenizer, model, device, batch_size)
    print("  Encoding error samples...")
    error_embeddings = _encode_texts(errors_df["headline"].astype(str).tolist(), tokenizer, model, device, batch_size)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"  Fitting KNN (k={k_neighbors})...")
    knn = NearestNeighbors(n_neighbors=k_neighbors, metric="cosine")
    knn.fit(ref_embeddings)
    distances, indices = knn.kneighbors(error_embeddings)

    records = []
    for i, row in errors_df.iterrows():
        neighbor_idx = indices[i]
        neighbor_dist = distances[i]
        neighbors = df_ref.iloc[neighbor_idx]

        sarcastic_ratio = (neighbors["is_sarcastic"] == 1).mean()
        closest_dist = neighbor_dist[0]
        closest_label = int(neighbors.iloc[0]["is_sarcastic"])

        is_fp = row["error_type"] == "false_positive"
        actual_label = row["true_label"]
        pred_label = row["pred_label"]
        confidence = row["confidence"]
        token_len = row.get("token_len_approx", len(str(row["headline"]).split()))
        has_exclamation = bool(row.get("exclamation", "!" in str(row["headline"])))
        has_question = bool(row.get("question", "?" in str(row["headline"])))
        has_full_stop = "." in str(row["headline"])

        is_outlier = closest_dist > 0.15
        is_conflict = (not is_outlier) and (closest_label != actual_label)
        is_systematic_bias = (not is_outlier) and (closest_label == pred_label) and (confidence > 0.90)
        is_ambiguous = (confidence < 0.60) and (token_len < 6)
        is_syntactic_bias = is_fp and (has_exclamation or has_question) and (closest_dist > 0.10)
        is_mixed = 0.3 <= sarcastic_ratio <= 0.7

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

        records.append(
            {
                "Index": row.name,
                "Article_Link": row.get("article_link", ""),
                "Headline": row["headline"],
                "Dataset": dataset_name,
                "Probability of non sarcastic": row.get(
                    "prob_non_sarcastic",
                    1 - row.get("prob_sarcastic", 0),
                ),
                "Probability of sarcastic": row.get("prob_sarcastic", 0),
                "Confidence": confidence,
                "Predicted is sarcastic": pred_label,
                "Actual label": actual_label,
                "Is correct?": row["correct"],
                "False +ve": 1 if is_fp else 0,
                "False -ve": 0 if is_fp else 1,
                "Text length": len(str(row["headline"])),
                "Approximate token length": token_len,
                "Is exclamation?": 1 if has_exclamation else 0,
                "Is question?": 1 if has_question else 0,
                "Is full stop?": 1 if has_full_stop else 0,
                "Distance to closest neighbor": closest_dist,
                "Closest neighbor label": closest_label,
                "Neighbor sarcastic ratio": sarcastic_ratio,
                "Identified Category": issue_cat,
            }
        )

    results_df = pd.DataFrame(records)

    actual_k = min(n_clusters, len(errors_df))
    print(f"  Fitting KMeans (k={actual_k}) on {len(error_embeddings)} error embeddings...")
    kmeans = KMeans(n_clusters=actual_k, random_state=seed, n_init="auto")
    cluster_labels = kmeans.fit_predict(error_embeddings)

    errors_clustered = errors_df.copy()
    errors_clustered["cluster_id"] = cluster_labels

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    tfidf.fit(errors_clustered["headline"].astype(str))
    feature_names = np.array(tfidf.get_feature_names_out())

    cluster_themes: dict[int, str] = {}
    n_keywords = 5
    for cid in range(actual_k):
        mask = errors_clustered["cluster_id"] == cid
        cluster_headlines = errors_clustered.loc[mask, "headline"].astype(str).tolist()
        if not cluster_headlines:
            cluster_themes[cid] = f"Cluster {cid} (empty)"
            continue
        tfidf_matrix = tfidf.transform(cluster_headlines)
        mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        top_idx = mean_scores.argsort()[::-1][:n_keywords]
        keywords = ", ".join(feature_names[top_idx])
        cluster_themes[cid] = f"C{cid}: {keywords}"

    errors_clustered["cluster_theme"] = errors_clustered["cluster_id"].map(cluster_themes)
    results_df["Cluster ID"] = cluster_labels
    results_df["Cluster Theme"] = errors_clustered["cluster_theme"].values

    csv_path = run_out_dir / "error_root_causes.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    report_lines = [
        f"MESO-LEVEL CLUSTER REPORT  (K={actual_k})",
        f"Mode: {mode}   Dataset: {dataset_name}",
        "=" * 70,
    ]
    n_samples = 4
    for cid in range(actual_k):
        mask = errors_clustered["cluster_id"] == cid
        sub = errors_clustered[mask]
        n_fp = (sub["error_type"] == "false_positive").sum()
        n_fn = (sub["error_type"] == "false_negative").sum()
        report_lines.extend(
            [
                "",
                "-" * 70,
                f"  {cluster_themes[cid]}",
                f"  Size: {len(sub)}  |  FP: {n_fp}  |  FN: {n_fn}",
                "-" * 70,
            ]
        )
        samples = sub["headline"].sample(min(n_samples, len(sub)), random_state=seed)
        for j, headline in enumerate(samples, 1):
            report_lines.append(f"  [{j}] {headline}")
    report_lines.extend(["", "=" * 70])
    report_text = "\n".join(report_lines)

    report_path = run_out_dir / "cluster_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"  Saved: {report_path}")

    print("  Computing t-SNE (may take a minute)...")
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30, max_iter=1000)
    all_embeddings = np.vstack([ref_embeddings, error_embeddings])
    all_2d = tsne.fit_transform(all_embeddings)
    n_ref = len(ref_embeddings)
    ref_2d = all_2d[:n_ref]
    error_2d = all_2d[n_ref:]

    is_fp_mask = errors_clustered["error_type"] == "false_positive"
    is_fn_mask = errors_clustered["error_type"] == "false_negative"
    fp_2d = error_2d[is_fp_mask.values]
    fn_2d = error_2d[is_fn_mask.values]

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

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"t-SNE Error Analysis - {mode} / {dataset_name}", fontsize=14, fontweight="bold")

    axes[0].set_title("Unclustered Manifold", fontsize=13)
    axes[0].scatter(ref_2d[:, 0], ref_2d[:, 1], c="blue", alpha=0.3, s=10, edgecolors="none")
    axes[0].scatter(error_2d[:, 0], error_2d[:, 1], c="blue", alpha=0.9, s=15, edgecolors="none")
    axes[0].set_xlabel("t-SNE Dim 1")
    axes[0].set_ylabel("t-SNE Dim 2")
    axes[0].tick_params(direction="in", top=True, right=True)

    axes[1].set_title("Error Density (FP / FN)", fontsize=13)
    axes[1].scatter(ref_2d[:, 0], ref_2d[:, 1], c="green", alpha=0.15, s=10, edgecolors="none", label="Reference")
    axes[1].scatter(
        fp_2d[:, 0],
        fp_2d[:, 1],
        c="red",
        alpha=0.9,
        s=25,
        edgecolors="white",
        linewidth=0.5,
        label="False Positive",
    )
    axes[1].scatter(
        fn_2d[:, 0],
        fn_2d[:, 1],
        c="blue",
        alpha=0.9,
        s=25,
        edgecolors="white",
        linewidth=0.5,
        label="False Negative",
    )
    axes[1].set_xlabel("t-SNE Dim 1")
    axes[1].set_ylabel("t-SNE Dim 2")
    axes[1].tick_params(direction="in", top=True, right=True)
    axes[1].legend(loc="best", fontsize=10)

    plt.tight_layout()
    tsne_path = run_out_dir / "tsne_analysis.png"
    fig.savefig(tsne_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {tsne_path}")

    cmap = matplotlib.colormaps["tab10"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Meso-Level Cluster Analysis - {mode} / {dataset_name}", fontsize=15, fontweight="bold")

    ax = axes[0]
    ax.set_title("Error Clusters (KMeans -> t-SNE)", fontsize=13)
    ax.scatter(ref_2d[:, 0], ref_2d[:, 1], c="lightgray", s=6, alpha=0.25, edgecolors="none")
    scatter = ax.scatter(
        error_2d[:, 0],
        error_2d[:, 1],
        c=cluster_labels,
        cmap="tab10",
        vmin=0,
        vmax=actual_k - 1,
        s=35,
        alpha=0.9,
        edgecolors="white",
        linewidth=0.4,
    )
    plt.colorbar(scatter, ax=ax, label="Cluster ID", ticks=range(actual_k))
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.tick_params(direction="in", top=True, right=True)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=cluster_themes[cid],
            markerfacecolor=cmap(cid / max(actual_k - 1, 1)),
            markersize=8,
        )
        for cid in range(actual_k)
    ]
    ax.legend(
        handles=handles,
        title="Cluster Theme",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=7.5,
        title_fontsize=9,
        framealpha=0.9,
    )

    ax2 = axes[1]
    ax2.set_title("Clusters + Error Type  ([square] FP  [triangle] FN)", fontsize=13)
    ax2.scatter(ref_2d[:, 0], ref_2d[:, 1], c="lightgray", s=6, alpha=0.25, edgecolors="none")
    for cid in range(actual_k):
        cluster_mask = cluster_labels == cid
        fp_cluster = cluster_mask & is_fp_mask.values
        fn_cluster = cluster_mask & is_fn_mask.values
        colour = cmap(cid / max(actual_k - 1, 1))
        if fp_cluster.any():
            ax2.scatter(
                error_2d[fp_cluster, 0],
                error_2d[fp_cluster, 1],
                color=colour,
                marker="s",
                s=40,
                alpha=0.85,
                edgecolors="white",
                linewidth=0.4,
            )
        if fn_cluster.any():
            ax2.scatter(
                error_2d[fn_cluster, 0],
                error_2d[fn_cluster, 1],
                color=colour,
                marker="^",
                s=50,
                alpha=0.85,
                edgecolors="white",
                linewidth=0.4,
            )
    ax2.set_xlabel("t-SNE Dim 1")
    ax2.set_ylabel("t-SNE Dim 2")
    ax2.tick_params(direction="in", top=True, right=True)
    fp_patch = plt.Line2D([0], [0], marker="s", color="w", label="False Positive", markerfacecolor="gray", markersize=9)
    fn_patch = plt.Line2D([0], [0], marker="^", color="w", label="False Negative", markerfacecolor="gray", markersize=9)
    ax2.legend(handles=[fp_patch, fn_patch], loc="lower right", fontsize=10)

    plt.tight_layout()
    cluster_path = run_out_dir / "cluster_analysis.png"
    fig.savefig(cluster_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {cluster_path}")

    print(f"\n  Done: {run_label}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the baseline BERT error diagnostic pipeline and save all outputs to disk."
    )
    parser.add_argument(
        "--mode",
        default="original_with_tuning",
        help=(
            "Experiment mode to diagnose, or 'all' to auto-discover every mode "
            "that has a predictions CSV in outputs/predictions."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="original_test",
        help=(
            "Dataset name to analyse, or 'all' to auto-discover every dataset "
            "that has a predictions CSV for the selected mode(s)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root output directory. Default: models/baseline_bert/outputs/error_diagnostic_results",
    )
    parser.add_argument("--k-neighbors", type=int, default=10, help="Number of nearest neighbours for KNN search.")
    parser.add_argument("--n-clusters", type=int, default=8, help="Number of KMeans clusters.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for t-SNE, KMeans, and cluster sampling.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for model encoding.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "all":
        seed_datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]
        modes = _discover_modes(PREDICTIONS_DIR, seed_datasets)
        if not modes:
            print(f"[WARN] No predictions CSVs found under {PREDICTIONS_DIR}.")
            return
    else:
        modes = [args.mode]

    if args.dataset == "all":
        datasets = _discover_datasets(PREDICTIONS_DIR, modes)
        if not datasets:
            print(f"[WARN] No prediction files found for modes {modes} under {PREDICTIONS_DIR}.")
            return
    else:
        datasets = [args.dataset]

    output_dir = Path(args.output_dir) if args.output_dir else ERROR_DIAGNOSTIC_RESULTS_DIR

    print(f"Predictions dir  : {PREDICTIONS_DIR}")
    print(f"Model weights dir: {MODEL_OUTPUT_DIR}")
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
                k_neighbors=args.k_neighbors,
                n_clusters=args.n_clusters,
                seed=args.seed,
                batch_size=args.batch_size,
            )

    print(f"\nAll runs complete. Results in: {output_dir}")


if __name__ == "__main__":
    main()
