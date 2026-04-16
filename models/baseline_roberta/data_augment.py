"""
Data Augmentation Pipeline — BERT Edition
==========================================
Adapted from data_augment.py to use the fine-tuned BERT checkpoint
(improved_with_tuning) for semantic embeddings instead of deBERTa.

The pipeline is identical in structure but differs in three ways:
  1. Embedding model  : fine-tuned BERT (improved_with_tuning checkpoint)
  2. Master pool      : single master_copy_dedup_v2.csv (already built)
  3. Output schema    : matches data/processed/original/train.csv
                        (is_sarcastic, headline, article_link, source_domain)

Pipeline Steps
--------------
  1. Load master pool and exclude training / val / test headlines
  2. Embed the master pool and error headlines with BERT
  2.5 Cluster error embeddings (KMeans + silhouette auto-tuning)
  3. Cluster-aware, angle-specific KNN candidate selection
  4. Cross-cluster deduplication and export

Input
-----
  ERROR_ROOT_CAUSES_CSV  : Diagnostic output (see config below)
  MASTER_POOL_PATH       : data/processed/master/master_copy_dedup_v2.csv
  TRAIN/VAL/TEST paths   : data/processed/original/{train,val,test}.csv

Output
------
  data/augmentation_output/baseline_roberta/round_3/augmentation_candidates.csv  — append to train.csv
  data/augmentation_output/baseline_roberta/round_3/augmentation_report.csv      — human-review report
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Diagnostic output — using the held-out validation set errors.
# These 156 errors represent genuine failure modes on unseen data and
# are more representative than master_copy errors for targeted augmentation.
ERROR_ROOT_CAUSES_CSV = Path("error_diagnostic/results/baseline_roberta/augmented_with_tuning_rd2_validation_set/error_root_causes.csv")

# Pool to pull augmentation candidates from
MASTER_POOL_PATH = Path("data/processed/master/master_copy_dedup_v2.csv")

# Exclusion sets: any headline already in training, val, or test is blocked
TRAIN_PATH = Path("data/augmentation_output/baseline_roberta/round_3/train.csv")
VAL_PATH   = Path("data/processed/original/val.csv")
TEST_PATH  = Path("data/processed/original/test.csv")

# Fine-tuned BERT checkpoint used for semantic embeddings.
# Using the task-specific checkpoint means the embedding space already
# separates sarcastic / non-sarcastic headlines — better KNN signal than
# a generic sentence encoder.
EMBEDDING_MODEL = Path(
    "models/baseline_roberta/outputs/checkpoints/augmented_with_tuning_rd2"
)

# KNN / distance hyperparameters (mirror data_augment.py)
BATCH_SIZE   = 64
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
K_MASTER     = 20   # neighbours to retrieve per error point

# Distance thresholds (cosine space, L2-normalised embeddings)
DIST_OUTLIER_FILL    = 0.30   # Angle 2: wide net to fill coverage voids
DIST_COUNTER_EXAMPLE = 0.20   # Angle 3 & 5: tight, must be on-topic
DIST_MIXED_REGION    = 0.15   # Angle 6: very tight to sharpen boundary
DIST_AMBIGUITY       = 0.25   # Angle 4: moderate radius

# Meso-level clustering
CLUSTER_K_MIN      = 4
CLUSTER_K_MAX      = 12
CLUSTER_N_KEYWORDS = 5

# Per-cluster criticality caps (max candidates kept per cluster)
CLUSTER_CRITICALITY_CAPS = {
    "Angle 2: Outlier / Zero-Shot Zone":                100,
    "Angle 3: Systematic Bias / Learned Spurious Rule":  60,
    "Angle 4: Low-Signal Ambiguity (Too short/vague)":   40,
    "Angle 5: Punctuation Spurious Correlation":         60,
    "Angle 6: Mixed Neighborhood (Tangled Region)":      80,
}
DEFAULT_CLUSTER_CAP = 50

OUTPUT_DIR = Path("data/augmentation_output/baseline_roberta/round_3")


# ══════════════════════════════════════════════════════════════════════════════
# Angle routing
# ══════════════════════════════════════════════════════════════════════════════

AUGMENTABLE_ANGLES = {
    "Angle 2: Outlier / Zero-Shot Zone",
    "Angle 3: Systematic Bias / Learned Spurious Rule",
    "Angle 4: Low-Signal Ambiguity (Too short/vague)",
    "Angle 5: Punctuation Spurious Correlation",
    "Angle 6: Mixed Neighborhood (Tangled Region)",
}

SKIP_ANGLES = {
    "Angle 1: Label Conflict (Fuzzy Boundary)",
    "Uncategorized Error",
}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Load master pool and deduplicate
# ══════════════════════════════════════════════════════════════════════════════

def load_master_pool() -> pd.DataFrame:
    print("=" * 70)
    print("  STEP 1: Loading master pool & deduplicating")
    print("=" * 70)

    # Build exclusion set from all three splits
    exclude_headlines: set[str] = set()
    for label, path in [("train", TRAIN_PATH), ("val", VAL_PATH), ("test", TEST_PATH)]:
        if path.exists():
            df = pd.read_csv(path)
            exclude_headlines |= set(df["headline"].str.strip().str.lower())
            print(f"  {label} split loaded: {len(df)} headlines excluded")
        else:
            print(f"  WARNING: {path} not found — skipping exclusion for {label}")

    # Also exclude the diagnosed error headlines themselves — when errors come
    # from the master copy (as here), they live inside the pool and would be
    # retrieved at distance=0, producing candidates that are the errors themselves.
    if ERROR_ROOT_CAUSES_CSV.exists():
        err_df = pd.read_csv(ERROR_ROOT_CAUSES_CSV)
        headline_col = "Headline" if "Headline" in err_df.columns else "headline"
        error_headlines = set(err_df[headline_col].str.strip().str.lower())
        exclude_headlines |= error_headlines
        print(f"  Error headlines excluded: {len(error_headlines)}")

    print(f"  Total unique headlines to exclude: {len(exclude_headlines)}")

    # Load master pool
    if not MASTER_POOL_PATH.exists():
        raise FileNotFoundError(f"Master pool not found: {MASTER_POOL_PATH}")
    master_df = pd.read_csv(MASTER_POOL_PATH)
    print(f"\n  Master pool loaded: {len(master_df)} rows")

    # Ensure required columns
    if "headline" not in master_df.columns or "is_sarcastic" not in master_df.columns:
        raise ValueError("master_copy_dedup_v2.csv must have 'headline' and 'is_sarcastic' columns")

    # Fill optional columns
    if "article_link" not in master_df.columns:
        master_df["article_link"] = "N/A"
    if "source_domain" not in master_df.columns:
        master_df["source_domain"] = master_df["article_link"].apply(_extract_domain)

    # Remove internal duplicates
    before = len(master_df)
    master_df = master_df.drop_duplicates(subset="headline", keep="first")
    print(f"  Internal duplicates removed: {before - len(master_df)}")

    # Remove training / val / test leakage
    master_df["_hl_lower"] = master_df["headline"].str.strip().str.lower()
    n_leaked = master_df["_hl_lower"].isin(exclude_headlines).sum()
    master_df = master_df[~master_df["_hl_lower"].isin(exclude_headlines)].reset_index(drop=True)
    master_df = master_df.drop(columns=["_hl_lower"])
    print(f"  Training/test leakage removed: {n_leaked}")
    print(f"  Remaining pool: {len(master_df)}")

    label_counts = master_df["is_sarcastic"].value_counts()
    print(f"  Pool label split: {label_counts.get(0, 0)} non-sarcastic, "
          f"{label_counts.get(1, 0)} sarcastic")

    return master_df


def _extract_domain(url: str) -> str:
    """Best-effort domain extraction from a URL string."""
    if not isinstance(url, str) or url == "N/A":
        return "N/A"
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc or "N/A"
    except Exception:
        return "N/A"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Embed with BERT
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_path: Path, device: str):
    print(f"\n  Loading ROBERTA checkpoint from: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True).to(device)
    model.eval()
    return model, tokenizer


def encode_texts(texts: list[str], model, tokenizer, device: str,
                 batch_size: int = 64, desc: str = "Encoding") -> np.ndarray:
    """Mean-pooled, L2-normalised embeddings (identical to data_augment.py)."""
    texts = [str(t) if t is not None and t == t else "" for t in texts]
    vectors = []
    batches = range(0, len(texts), batch_size)
    with torch.no_grad():
        for i in tqdm(batches, desc=f"    {desc}", unit="batch"):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True,
                max_length=128, return_tensors="pt"
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            mask = (
                encoded["attention_mask"]
                .unsqueeze(-1)
                .expand(outputs.last_hidden_state.size())
                .float()
            )
            summed = (outputs.last_hidden_state * mask).sum(dim=1)
            denom  = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = summed / denom
            normalized = F.normalize(pooled, p=2, dim=1)
            vectors.append(normalized.cpu().numpy())
    return np.vstack(vectors)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2.5: Meso-level clustering of error embeddings
# ══════════════════════════════════════════════════════════════════════════════

def cluster_errors(errors_df: pd.DataFrame, error_embeddings: np.ndarray):
    print("\n" + "=" * 70)
    print("  STEP 2.5: Meso-Level Clustering of Error Embeddings")
    print("=" * 70)

    n_errors = len(error_embeddings)
    if n_errors < CLUSTER_K_MIN + 1:
        print(f"  Too few errors ({n_errors}) to cluster. Assigning all to cluster 0.")
        out = errors_df.copy()
        out["cluster_id"]    = 0
        out["cluster_theme"] = "C0: (all errors)"
        return out, 1

    # Auto-tune K via silhouette score
    k_range   = range(CLUSTER_K_MIN, min(CLUSTER_K_MAX + 1, n_errors))
    best_k, best_score = CLUSTER_K_MIN, -1.0
    print(f"  Auto-tuning k in [{CLUSTER_K_MIN}, {min(CLUSTER_K_MAX, n_errors - 1)}]...")
    for k in tqdm(k_range, desc="    Silhouette search", unit="k"):
        km     = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(error_embeddings)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(
            error_embeddings, labels, metric="cosine",
            sample_size=min(5000, n_errors)
        )
        tqdm.write(f"    k={k:2d}  silhouette={score:.4f}")
        if score > best_score:
            best_score, best_k = score, k

    print(f"  Best k = {best_k}  (silhouette = {best_score:.4f})")

    kmeans         = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(error_embeddings)

    out = errors_df.copy()
    out["cluster_id"] = cluster_labels

    # Auto-name clusters with TF-IDF keywords
    # Error CSV from the diagnostic pipeline uses capital 'Headline'
    headline_col = "Headline" if "Headline" in out.columns else "headline"
    tfidf        = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    tfidf.fit(out[headline_col].astype(str))
    feature_names = np.array(tfidf.get_feature_names_out())

    cluster_themes: dict[int, str] = {}
    for cid in range(best_k):
        mask  = out["cluster_id"] == cid
        texts = out.loc[mask, headline_col].astype(str).tolist()
        if not texts:
            cluster_themes[cid] = f"C{cid}: (empty)"
            continue
        mat        = tfidf.transform(texts)
        mean_scores = np.asarray(mat.mean(axis=0)).flatten()
        top_idx    = mean_scores.argsort()[::-1][:CLUSTER_N_KEYWORDS]
        keywords   = ", ".join(feature_names[top_idx])
        cluster_themes[cid] = f"C{cid}: {keywords}"

    out["cluster_theme"] = out["cluster_id"].map(cluster_themes)

    print(f"\n  Cluster summary ({best_k} clusters, {n_errors} errors):")
    for cid in range(best_k):
        sub = out[out["cluster_id"] == cid]
        fp  = (sub.get("False +ve", pd.Series(dtype=int)) == 1).sum()
        fn  = (sub.get("False -ve", pd.Series(dtype=int)) == 1).sum()
        print(f"    {cluster_themes[cid]}")
        print(f"      Size: {len(sub)}  |  FP: {fp}  |  FN: {fn}")

    return out, best_k


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Cluster-aware, angle-specific candidate selection
# ══════════════════════════════════════════════════════════════════════════════

def select_candidates_for_error(row, pool_df, pool_knn, error_emb,
                                 target_label=None):
    """Return nearby master-pool candidates for a single diagnosed error."""
    category = row["Identified Category"]
    if category not in AUGMENTABLE_ANGLES:
        return None

    dists, idxs = pool_knn.kneighbors(error_emb.reshape(1, -1))
    candidates  = pool_df.iloc[idxs[0]].copy()
    candidates["_distance"]             = dists[0]
    candidates["_source_error_headline"] = row["Headline"]
    candidates["_source_category"]      = category

    if target_label is not None:
        label_mask = candidates["is_sarcastic"] == target_label
    else:
        label_mask = candidates["is_sarcastic"] == row["Actual label"]

    if category == "Angle 2: Outlier / Zero-Shot Zone":
        candidates = candidates[candidates["_distance"] < DIST_OUTLIER_FILL]

    elif category == "Angle 3: Systematic Bias / Learned Spurious Rule":
        candidates = candidates[
            label_mask & (candidates["_distance"] < DIST_COUNTER_EXAMPLE)
        ]

    elif category == "Angle 4: Low-Signal Ambiguity (Too short/vague)":
        candidates = candidates[
            label_mask &
            (candidates["_distance"] < DIST_AMBIGUITY) &
            (candidates["headline"].str.split().str.len() >= 6)
        ]

    elif category == "Angle 5: Punctuation Spurious Correlation":
        candidates = candidates[
            label_mask & (candidates["_distance"] < DIST_COUNTER_EXAMPLE)
        ]

    elif category == "Angle 6: Mixed Neighborhood (Tangled Region)":
        candidates = candidates[candidates["_distance"] < DIST_MIXED_REGION]

    return candidates if len(candidates) > 0 else None


def run_cluster_augmentation_selection(
    errors_df, master_df, master_knn,
    master_0_df, master_0_knn, master_0_embeddings,
    master_1_df, master_1_knn, master_1_embeddings,
    error_embeddings, master_embeddings, n_clusters
) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("  STEP 3: Cluster-aware candidate selection")
    print("=" * 70)

    all_cluster_candidates = []
    cluster_stats = []

    for cid in range(n_clusters):
        cluster_mask    = errors_df["cluster_id"] == cid
        cluster_errors  = errors_df[cluster_mask]
        cluster_indices = np.where(cluster_mask.values)[0]
        theme = (
            cluster_errors["cluster_theme"].iloc[0]
            if len(cluster_errors) > 0
            else f"C{cid}"
        )

        if len(cluster_errors) == 0:
            continue

        # Determine dominant error type → select label-specific pool
        if "False +ve" in cluster_errors.columns and "False -ve" in cluster_errors.columns:
            n_fp = (cluster_errors["False +ve"] == 1).sum()
            n_fn = (cluster_errors["False -ve"] == 1).sum()
            dominant_error_type = "false_positive" if n_fp >= n_fn else "false_negative"
        else:
            dominant_error_type = "unknown"

        if dominant_error_type == "false_negative":
            pool_df, pool_knn, pool_emb, target_label = (
                master_1_df, master_1_knn, master_1_embeddings, 1
            )
        elif dominant_error_type == "false_positive":
            pool_df, pool_knn, pool_emb, target_label = (
                master_0_df, master_0_knn, master_0_embeddings, 0
            )
        else:
            pool_df, pool_knn, pool_emb, target_label = (
                master_df, master_knn, master_embeddings, None
            )

        # Per-point KNN queries
        cluster_candidates = []
        skipped_count      = 0

        rows = list(cluster_errors.iterrows())
        for local_i, (global_i, row) in enumerate(
            tqdm(rows, desc=f"    C{cid} KNN queries", unit="err", leave=False)
        ):
            if row["Identified Category"] in SKIP_ANGLES:
                skipped_count += 1
                continue
            emb_idx = cluster_indices[local_i]
            result  = select_candidates_for_error(
                row, pool_df, pool_knn,
                error_embeddings[emb_idx], target_label
            )
            if result is not None:
                result["_cluster_id"]    = cid
                result["_cluster_theme"] = theme
                cluster_candidates.append(result)

        if not cluster_candidates:
            cluster_stats.append({
                "cluster": theme, "errors": len(cluster_errors),
                "skipped": skipped_count, "candidates_raw": 0,
                "candidates_dedup": 0, "cap_applied": "—",
                "candidates_final": 0,
            })
            continue

        # Deduplicate within cluster (keep closest match)
        cluster_df  = pd.concat(cluster_candidates, ignore_index=True)
        raw_count   = len(cluster_df)
        cluster_df  = (
            cluster_df.sort_values("_distance")
            .drop_duplicates(subset="headline", keep="first")
            .reset_index(drop=True)
        )
        dedup_count = len(cluster_df)

        # Criticality cap based on dominant angle
        angle_counts   = cluster_errors["Identified Category"].value_counts()
        dominant_angle = angle_counts.index[0] if len(angle_counts) > 0 else ""
        cap            = CLUSTER_CRITICALITY_CAPS.get(dominant_angle, DEFAULT_CLUSTER_CAP)

        if len(cluster_df) > cap:
            cluster_df = cluster_df.nsmallest(cap, "_distance").reset_index(drop=True)

        final_count = len(cluster_df)
        all_cluster_candidates.append(cluster_df)

        cluster_stats.append({
            "cluster": theme, "errors": len(cluster_errors),
            "skipped": skipped_count, "candidates_raw": raw_count,
            "candidates_dedup": dedup_count,
            "cap_applied": f"{cap} ({dominant_angle})",
            "candidates_final": final_count,
        })

    # Report
    print("\n  Per-cluster selection results:\n")
    for s in cluster_stats:
        print(f"    {s['cluster']}")
        print(f"      Errors: {s['errors']}  |  Skipped: {s['skipped']}  |  "
              f"Raw: {s['candidates_raw']}  →  Dedup: {s['candidates_dedup']}  "
              f"→  Final: {s['candidates_final']}")
        print(f"      Cap: {s['cap_applied']}")

    if not all_cluster_candidates:
        print("\n  WARNING: No augmentation candidates found.")
        return pd.DataFrame()

    return pd.concat(all_cluster_candidates, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Cross-cluster deduplication and export
# ══════════════════════════════════════════════════════════════════════════════

def finalize_and_export(aug_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  STEP 4: Cross-cluster deduplication & export")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if aug_df.empty:
        print("  No candidates to export.")
        return

    before  = len(aug_df)
    aug_df  = (
        aug_df.sort_values("_distance")
        .drop_duplicates(subset="headline", keep="first")
        .reset_index(drop=True)
    )
    print(f"  After cross-cluster dedup: {len(aug_df)} (removed {before - len(aug_df)})")

    # ── Detailed report for human review ─────────────────────────────────────
    report_cols = [
        "headline", "is_sarcastic", "article_link", "source_domain",
        "_source_error_headline", "_source_category", "_distance",
    ]
    if "_cluster_id" in aug_df.columns:
        report_cols += ["_cluster_id", "_cluster_theme"]

    report_df = aug_df[[c for c in report_cols if c in aug_df.columns]].copy()
    report_df.columns = [c.lstrip("_") for c in report_df.columns]
    report_path = OUTPUT_DIR / "augmentation_report.csv"
    report_df.to_csv(report_path, index=False)

    # ── Training-ready CSV (matches data/processed/original/train.csv schema) ─
    # Columns: is_sarcastic, headline, article_link, source_domain
    if "source_domain" not in aug_df.columns:
        aug_df["source_domain"] = aug_df["article_link"].apply(_extract_domain)

    train_df = aug_df[["is_sarcastic", "headline", "article_link", "source_domain"]].copy()
    candidates_path = OUTPUT_DIR / "augmentation_candidates.csv"
    train_df.to_csv(candidates_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  Final augmentation pool: {len(aug_df)} candidates")
    print(f"  Label split: "
          f"{(aug_df['is_sarcastic'] == 0).sum()} non-sarcastic, "
          f"{(aug_df['is_sarcastic'] == 1).sum()} sarcastic")

    print(f"\n  Breakdown by diagnostic angle:")
    for cat, group in aug_df.groupby("_source_category"):
        print(f"    {cat}: {len(group)} candidates "
              f"(mean dist: {group['_distance'].mean():.4f})")

    if "_cluster_id" in aug_df.columns:
        print(f"\n  Breakdown by cluster:")
        for (cid, theme), group in aug_df.groupby(["_cluster_id", "_cluster_theme"]):
            print(f"    {theme}: {len(group)} candidates "
                  f"(mean dist: {group['_distance'].mean():.4f})")

    print(f"\n  Files written:")
    print(f"    {report_path}    (detailed report for human review)")
    print(f"    {candidates_path}  (training-ready, matches train.csv schema)")
    print(f"\n  To augment your training set:")
    print(f"    import pandas as pd")
    print(f"    df_train = pd.concat([")
    print(f"        pd.read_csv('{TRAIN_PATH}'),")
    print(f"        pd.read_csv('{candidates_path}')")
    print(f"    ]).reset_index(drop=True)")
    print(f"    df_train.to_csv('{TRAIN_PATH}', index=False)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║        DATA AUGMENTATION PIPELINE — ROBERTA EDITION              ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device    : {DEVICE}")
    print(f"  Errors CSV: {ERROR_ROOT_CAUSES_CSV}")

    # ── Load diagnostic errors ────────────────────────────────────────────────
    if not ERROR_ROOT_CAUSES_CSV.exists():
        raise FileNotFoundError(
            f"Diagnostic output not found: {ERROR_ROOT_CAUSES_CSV}\n"
            "Run the error diagnostic pipeline first."
        )
    errors_df = pd.read_csv(ERROR_ROOT_CAUSES_CSV)
    print(f"\n  Loaded {len(errors_df)} diagnosed errors")
    print("  Angle distribution:")
    for cat, count in errors_df["Identified Category"].value_counts().items():
        print(f"    {cat}: {count}")

    # ── Step 1: Master pool ───────────────────────────────────────────────────
    master_df = load_master_pool()

    # ── Step 2: Embeddings ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 2: Computing embeddings")
    print("=" * 70)

    model, tokenizer = load_model_and_tokenizer(EMBEDDING_MODEL, DEVICE)

    print("  Encoding master pool...")
    master_embeddings = encode_texts(
        master_df["headline"].astype(str).tolist(),
        model, tokenizer, DEVICE, BATCH_SIZE,
        desc="Master pool"
    )
    print(f"    Master embeddings shape: {master_embeddings.shape}")

    headline_col = "Headline" if "Headline" in errors_df.columns else "headline"
    print("  Encoding error headlines...")
    error_embeddings = encode_texts(
        errors_df[headline_col].astype(str).tolist(),
        model, tokenizer, DEVICE, BATCH_SIZE,
        desc="Error headlines"
    )
    print(f"    Error embeddings shape:  {error_embeddings.shape}")

    # ── Build label-specific KNN indices ─────────────────────────────────────
    print(f"  Fitting label-specific KNN indices (k={K_MASTER})...")
    mask_0 = master_df["is_sarcastic"] == 0
    mask_1 = master_df["is_sarcastic"] == 1

    master_0_df         = master_df[mask_0].reset_index(drop=True)
    master_1_df         = master_df[mask_1].reset_index(drop=True)
    master_0_embeddings = master_embeddings[mask_0.values]
    master_1_embeddings = master_embeddings[mask_1.values]

    master_0_knn = NearestNeighbors(n_neighbors=K_MASTER, metric="cosine")
    master_1_knn = NearestNeighbors(n_neighbors=K_MASTER, metric="cosine")
    master_knn   = NearestNeighbors(n_neighbors=K_MASTER, metric="cosine")

    if len(master_0_embeddings) > 0:
        master_0_knn.fit(master_0_embeddings)
    if len(master_1_embeddings) > 0:
        master_1_knn.fit(master_1_embeddings)
    master_knn.fit(master_embeddings)

    # ── Step 2.5: Cluster errors ──────────────────────────────────────────────
    errors_df, n_clusters = cluster_errors(errors_df, error_embeddings)

    # ── Step 3: Candidate selection ───────────────────────────────────────────
    aug_df = run_cluster_augmentation_selection(
        errors_df, master_df, master_knn,
        master_0_df, master_0_knn, master_0_embeddings,
        master_1_df, master_1_knn, master_1_embeddings,
        error_embeddings, master_embeddings, n_clusters
    )

    # ── Step 4: Export ────────────────────────────────────────────────────────
    finalize_and_export(aug_df)

    print("\n" + "═" * 70)
    print("  Pipeline complete.")
    print("═" * 70)


if __name__ == "__main__":
    main()
