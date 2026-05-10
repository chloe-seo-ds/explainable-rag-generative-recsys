"""
Shared utilities for RecSys26 Tutorial:
Choosing Between Explainable, Retrieval-Augmented, and LLM-Native Recommenders in Text-Rich Domains

Anchor papers:
  - XRec (Ma et al., EMNLP 2024) — Explainable paradigm
  - K-RagRec (Wang et al., ACL 2025) — RAG paradigm
  - GenRec (Ji et al., ECIR 2024) — Generative paradigm
"""

import json
import math
import os
import pickle
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
CACHE_DIR = ROOT_DIR / "cache"
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# Data loading and preprocessing
# ──────────────────────────────────────────────
def load_amazon_books(
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    max_users: int = 20_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Load and preprocess Amazon Books dataset (2023 version, Hou et al.).

    Downloads from McAuley-Lab/Amazon-Reviews-2023 on HuggingFace.
    Returns DataFrame with columns: user_id, item_id, rating, timestamp, title.
    """
    processed_path = DATA_DIR / "amazon_books_processed.parquet"
    if processed_path.exists():
        print(f"Loading cached data from {processed_path}")
        return pd.read_parquet(processed_path)

    print("Downloading Amazon Books dataset...")
    from datasets import load_dataset

    reviews = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_review_Books",
        split="full",
        trust_remote_code=True,
    )
    meta = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        "raw_meta_Books",
        split="full",
        trust_remote_code=True,
    )

    print("Processing reviews...")
    df_reviews = reviews.to_pandas()[["user_id", "parent_asin", "rating", "timestamp"]]
    df_reviews.columns = ["user_id", "item_id", "rating", "timestamp"]
    df_reviews = df_reviews.dropna(subset=["user_id", "item_id", "rating", "timestamp"])

    print("Processing metadata...")
    df_meta = meta.to_pandas()[["parent_asin", "title"]].drop_duplicates(subset=["parent_asin"])
    df_meta.columns = ["item_id", "title"]
    df_meta = df_meta.dropna(subset=["title"])
    df_meta = df_meta[df_meta["title"].str.len() > 0]

    # Merge
    df = df_reviews.merge(df_meta, on="item_id", how="inner")

    # K-core filtering
    print("Applying k-core filtering...")
    for _ in range(10):
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df["user_id"].isin(valid_users)]

        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= min_item_interactions].index
        df = df[df["item_id"].isin(valid_items)]

    # Subsample users for tractability
    rng = np.random.RandomState(seed)
    unique_users = df["user_id"].unique()
    if len(unique_users) > max_users:
        sampled_users = rng.choice(unique_users, max_users, replace=False)
        df = df[df["user_id"].isin(sampled_users)]

    # Re-filter items after user subsampling
    item_counts = df["item_id"].value_counts()
    valid_items = item_counts[item_counts >= min_item_interactions].index
    df = df[df["item_id"].isin(valid_items)]

    # Sort by timestamp within each user
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Re-index
    user_map = {u: i for i, u in enumerate(df["user_id"].unique())}
    item_map = {it: i for i, it in enumerate(df["item_id"].unique())}
    df["user_idx"] = df["user_id"].map(user_map)
    df["item_idx"] = df["item_id"].map(item_map)

    df.to_parquet(processed_path)
    print(f"Saved to {processed_path}")
    print(f"Users: {df['user_idx'].nunique()}, Items: {df['item_idx'].nunique()}, Interactions: {len(df)}")
    return df


def create_splits(df: pd.DataFrame) -> dict:
    """Leave-last-two-out split: last item = test, second-to-last = val, rest = train."""
    split_path = DATA_DIR / "splits.pkl"
    if split_path.exists():
        with open(split_path, "rb") as f:
            return pickle.load(f)

    train_data, val_data, test_data = [], [], []
    for user_idx, group in df.groupby("user_idx"):
        items = group.sort_values("timestamp")["item_idx"].tolist()
        if len(items) < 3:
            train_data.extend([(user_idx, it) for it in items])
            continue
        train_data.extend([(user_idx, it) for it in items[:-2]])
        val_data.append((user_idx, items[-2]))
        test_data.append((user_idx, items[-1]))

    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }
    with open(split_path, "wb") as f:
        pickle.dump(splits, f)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    return splits


def get_user_histories(train_data: list) -> dict:
    """Build user -> list of item_idx mapping from training data."""
    histories = defaultdict(list)
    for user_idx, item_idx in train_data:
        histories[user_idx].append(item_idx)
    return dict(histories)


def get_item_titles(df: pd.DataFrame) -> dict:
    """Build item_idx -> title mapping."""
    return df.drop_duplicates("item_idx").set_index("item_idx")["title"].to_dict()


def get_all_items(df: pd.DataFrame) -> set:
    """Get set of all item indices."""
    return set(df["item_idx"].unique())


# ──────────────────────────────────────────────
# Ranking metrics
# ──────────────────────────────────────────────
def hit_at_k(ranked_list: list, ground_truth: int, k: int = 10) -> float:
    """1 if ground_truth is in top-k of ranked_list, else 0."""
    return 1.0 if ground_truth in ranked_list[:k] else 0.0


def ndcg_at_k(ranked_list: list, ground_truth: int, k: int = 10) -> float:
    """NDCG@k for a single ground-truth item."""
    for i, item in enumerate(ranked_list[:k]):
        if item == ground_truth:
            return 1.0 / math.log2(i + 2)
    return 0.0


def recall_at_k(ranked_list: list, ground_truth: int, k: int = 10) -> float:
    """Recall@k for single ground-truth item (same as hit@k in this case)."""
    return hit_at_k(ranked_list, ground_truth, k)


def evaluate_ranking(predictions: dict, ground_truth: dict, k_values: list = None) -> dict:
    """Evaluate ranking predictions across all users.

    Args:
        predictions: {user_idx: [ranked_item_idx_list]}
        ground_truth: {user_idx: item_idx}
        k_values: list of k values to evaluate (default [5, 10, 20])

    Returns:
        dict of metric_name -> value
    """
    if k_values is None:
        k_values = [5, 10, 20]

    results = {}
    for k in k_values:
        hits, ndcgs = [], []
        for user_idx, true_item in ground_truth.items():
            if user_idx not in predictions:
                continue
            pred_list = predictions[user_idx]
            hits.append(hit_at_k(pred_list, true_item, k))
            ndcgs.append(ndcg_at_k(pred_list, true_item, k))

        results[f"HR@{k}"] = np.mean(hits) if hits else 0.0
        results[f"NDCG@{k}"] = np.mean(ndcgs) if ndcgs else 0.0

    return results


# ──────────────────────────────────────────────
# Explanation quality metrics
# ──────────────────────────────────────────────
def explanation_relevance(explanation: str, user_history_titles: list, item_title: str) -> float:
    """Does the explanation reference items/attributes the user actually interacted with?

    Simple lexical overlap measure: fraction of history titles mentioned in explanation.
    """
    if not explanation or not user_history_titles:
        return 0.0
    explanation_lower = explanation.lower()
    mentioned = sum(1 for t in user_history_titles if t.lower() in explanation_lower)
    return mentioned / len(user_history_titles)


def explanation_specificity(explanation: str, item_title: str) -> float:
    """Does the explanation go beyond generic praise to reference concrete item features?

    Heuristic: penalize short/generic explanations, reward specific entity mentions.
    """
    if not explanation:
        return 0.0
    words = explanation.split()
    if len(words) < 10:
        return 0.1
    # Check if the recommended item is mentioned
    item_mentioned = 1.0 if item_title.lower() in explanation.lower() else 0.0
    # Longer, more specific explanations score higher (capped)
    length_score = min(len(words) / 50.0, 1.0)
    return 0.5 * item_mentioned + 0.5 * length_score


def explanation_consistency(explanations: list) -> float:
    """Do repeated runs produce semantically stable explanations?

    Measures pairwise lexical overlap (Jaccard) among multiple explanations for the same input.
    For a proper implementation, use BERTScore or sentence-transformers cosine similarity.
    """
    if len(explanations) < 2:
        return 1.0
    scores = []
    for i in range(len(explanations)):
        for j in range(i + 1, len(explanations)):
            set_a = set(explanations[i].lower().split())
            set_b = set(explanations[j].lower().split())
            if not set_a or not set_b:
                continue
            jaccard = len(set_a & set_b) / len(set_a | set_b)
            scores.append(jaccard)
    return np.mean(scores) if scores else 0.0


MIN_TITLE_LENGTH = 4  # skip titles too short to match reliably


def explanation_hallucination_rate(
    explanation: str, item_title: str, user_history_titles: list, all_titles: set
) -> float:
    """Does the explanation assert facts not supported by the input data?

    Checks if the explanation mentions book titles that are neither the recommended item
    nor in the user's history. Returns fraction of mentioned titles that are hallucinated.
    Uses word-boundary matching to avoid false positives from short titles.
    """
    if not explanation:
        return 0.0
    explanation_lower = explanation.lower()
    valid_titles = set(t.lower() for t in user_history_titles) | {item_title.lower()}

    mentioned_count = 0
    hallucinated_count = 0
    for title in all_titles:
        title_lower = title.lower()
        if len(title_lower) < MIN_TITLE_LENGTH:
            continue
        if re.search(r'\b' + re.escape(title_lower) + r'\b', explanation_lower):
            mentioned_count += 1
            if title_lower not in valid_titles:
                hallucinated_count += 1

    return hallucinated_count / mentioned_count if mentioned_count > 0 else 0.0


def _cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def explanation_relevance_embedding(
    explanation: str, user_history_titles: list, item_title: str, encoder
) -> float:
    """Embedding-based relevance: cosine similarity between the explanation
    and the user's reading history (concatenated titles)."""
    if not explanation or not user_history_titles:
        return 0.0
    history_text = "; ".join(user_history_titles)
    embs = encoder.encode([explanation, history_text])
    return _cosine_sim(embs[0], embs[1])


def explanation_consistency_embedding(texts: list, encoder) -> float:
    """Embedding-based consistency: mean pairwise cosine similarity."""
    if len(texts) < 2:
        return 1.0
    embs = encoder.encode(texts)
    scores = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            scores.append(_cosine_sim(embs[i], embs[j]))
    return np.mean(scores) if scores else 0.0


def evaluate_explanations(
    explanations: dict,
    user_histories: dict,
    item_titles: dict,
    all_titles: Optional[set] = None,
    consistency_texts: Optional[dict] = None,
    encoder=None,
) -> dict:
    """Evaluate explanation quality across all users.

    Args:
        explanations: {user_idx: {"item_idx": int, "text": str}}
        user_histories: {user_idx: [item_idx_list]}
        item_titles: {item_idx: title_str}
        all_titles: set of all item titles (for hallucination check)
        consistency_texts: {user_idx: [text1, text2, ...]} — multiple explanations
            for the same input, used to measure cross-run stability
        encoder: a SentenceTransformer instance for embedding-based metrics;
            if None, only lexical metrics are computed

    Returns:
        dict of metric_name -> value
    """
    if all_titles is None:
        all_titles = set(item_titles.values())

    relevances, specificities, hallucinations = [], [], []
    emb_relevances = []
    for user_idx, expl_data in explanations.items():
        item_idx = expl_data["item_idx"]
        text = expl_data["text"]
        history = user_histories.get(user_idx, [])
        history_titles = [item_titles.get(it, "") for it in history if it in item_titles]
        item_title = item_titles.get(item_idx, "")

        relevances.append(explanation_relevance(text, history_titles, item_title))
        specificities.append(explanation_specificity(text, item_title))
        hallucinations.append(
            explanation_hallucination_rate(text, item_title, history_titles, all_titles)
        )
        if encoder is not None:
            emb_relevances.append(
                explanation_relevance_embedding(text, history_titles, item_title, encoder)
            )

    # Lexical consistency
    consistency_val = 0.0
    consistency_emb_val = 0.0
    if consistency_texts:
        cons_scores = [
            explanation_consistency(texts)
            for texts in consistency_texts.values()
            if len(texts) >= 2
        ]
        consistency_val = np.mean(cons_scores) if cons_scores else 0.0

        if encoder is not None:
            emb_cons = [
                explanation_consistency_embedding(texts, encoder)
                for texts in consistency_texts.values()
                if len(texts) >= 2
            ]
            consistency_emb_val = np.mean(emb_cons) if emb_cons else 0.0

    results = {
        "relevance": np.mean(relevances) if relevances else 0.0,
        "specificity": np.mean(specificities) if specificities else 0.0,
        "consistency": consistency_val,
        "hallucination_rate": np.mean(hallucinations) if hallucinations else 0.0,
    }
    if encoder is not None:
        results["relevance_embedding"] = np.mean(emb_relevances) if emb_relevances else 0.0
        results["consistency_embedding"] = consistency_emb_val

    return results


# ──────────────────────────────────────────────
# System metrics
# ──────────────────────────────────────────────
class LatencyTracker:
    """Track inference latency for system-level metrics."""

    def __init__(self):
        self.times = []

    def start(self):
        self._start = time.time()

    def stop(self):
        self.times.append(time.time() - self._start)

    def summary(self) -> dict:
        if not self.times:
            return {"mean_latency_ms": 0, "p50_latency_ms": 0, "p95_latency_ms": 0}
        arr = np.array(self.times) * 1000  # convert to ms
        return {
            "mean_latency_ms": float(np.mean(arr)),
            "p50_latency_ms": float(np.median(arr)),
            "p95_latency_ms": float(np.percentile(arr, 95)),
            "total_seconds": float(np.sum(self.times)),
        }


# ──────────────────────────────────────────────
# Results I/O
# ──────────────────────────────────────────────
def save_results(paradigm_name: str, results: dict):
    """Save paradigm results to JSON."""
    path = RESULTS_DIR / f"{paradigm_name}_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {path}")


def load_results(paradigm_name: str) -> dict:
    """Load paradigm results from JSON."""
    path = RESULTS_DIR / f"{paradigm_name}_results.json"
    with open(path) as f:
        return json.load(f)


def save_predictions(paradigm_name: str, predictions: dict):
    """Save raw predictions for later analysis."""
    path = RESULTS_DIR / f"{paradigm_name}_predictions.pkl"
    with open(path, "wb") as f:
        pickle.dump(predictions, f)


def load_predictions(paradigm_name: str) -> dict:
    """Load raw predictions."""
    path = RESULTS_DIR / f"{paradigm_name}_predictions.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────
# Cache I/O (pre-computed LLM outputs, embeddings)
# ──────────────────────────────────────────────
def save_cache(name: str, data):
    """Save pre-computed outputs to cache directory."""
    path = CACHE_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Cached: {path}")


def load_cache(name: str):
    """Load pre-computed outputs from cache. Returns None if not found."""
    path = CACHE_DIR / f"{name}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded cache: {path}")
        return data
    return None
