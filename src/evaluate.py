
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.nlp_pipeline import StartupNLPPipeline
from src.recommender import StartupRecommender


def train_test_split_ratings(
    ratings_df: pd.DataFrame,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out ~20% of ratings per user (at least 1 rating when user has >= 3 ratings)
    so every active user retains training interactions.
    """
    rng = np.random.default_rng(random_state)
    train_parts = []
    test_parts = []

    for _, group in ratings_df.groupby("user_id"):
        n = len(group)
        n_test = max(1, int(round(n * test_ratio))) if n >= 3 else 0
        if n_test >= n:
            n_test = max(0, n - 1)
        perm = rng.permutation(group.index.to_numpy())
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        train_parts.append(ratings_df.loc[train_idx])
        if len(test_idx) > 0:
            test_parts.append(ratings_df.loc[test_idx])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else ratings_df.copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=ratings_df.columns)
    return train_df, test_df


def precision_at_k(recommended_ids: List[str], relevant_ids: set[str], k: int = 5) -> float:
    if k <= 0:
        return 0.0
    top_k = recommended_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for iid in top_k if iid in relevant_ids)
    return hits / float(k)


def evaluate_models(
    items_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    nlp: StartupNLPPipeline,
    relevance_threshold: int = 4,
    k: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compare Popularity Baseline, NLP Content-Based, Item-Based CF, and Hybrid
    using Precision@K on held-out relevant items, plus MAE/RMSE for CF predictions.
    """
    train_df, test_df = train_test_split_ratings(ratings_df, test_ratio=0.2, random_state=random_state)
    train_rec = StartupRecommender(items_df, train_df, nlp, min_ratings=2)

    method_precisions: Dict[str, List[float]] = {
        "Popularity Baseline": [],
        "NLP Content-Based": [],
        "Item-Based CF": [],
        "Hybrid (NLP + CF + Pop)": [],
    }

    for user_id, u_test in test_df.groupby("user_id"):
        relevant = set(
            u_test[u_test["interest_rating"] >= relevance_threshold]["item_id"].astype(str)
        )
        if not relevant:
            continue

        # Khi đánh giá trên test split, không loại trừ test items khỏi ứng viên gợi ý
        # chỉ loại trừ các items user đã thấy trong train_df
        train_seen = set(train_df[train_df["user_id"] == user_id]["item_id"].astype(str))

        pop_all = train_rec.recommend_popularity(user_id=None, top_k=len(items_df), enforce_min_ratings=False)
        pop_recs = [iid for iid in pop_all["item_id"] if iid not in train_seen][:k]

        cb_all = train_rec.recommend_content_for_user(
            user_id=user_id, top_k=len(items_df), ratings_subset=train_df, exclude_rated=True
        )
        cb_recs = list(cb_all["item_id"])[:k]

        cf_all = train_rec.recommend_cf(
            user_id=user_id, top_k=len(items_df), ratings_subset=train_df, exclude_rated=True
        )
        cf_recs = list(cf_all["item_id"])[:k]

        hyb_all = train_rec.recommend_hybrid(
            user_id=user_id, top_k=len(items_df), ratings_subset=train_df, exclude_rated=True
        )
        hyb_recs = list(hyb_all["item_id"])[:k]

        method_precisions["Popularity Baseline"].append(precision_at_k(pop_recs, relevant, k=k))
        method_precisions["NLP Content-Based"].append(precision_at_k(cb_recs, relevant, k=k))
        method_precisions["Item-Based CF"].append(precision_at_k(cf_recs, relevant, k=k))
        method_precisions["Hybrid (NLP + CF + Pop)"].append(precision_at_k(hyb_recs, relevant, k=k))

    # Rating prediction metrics (MAE & RMSE) on test set using Item-Based CF
    y_true: List[float] = []
    y_pred_cf: List[float] = []
    for _, row in test_df.iterrows():
        y_true.append(float(row["interest_rating"]))
        y_pred_cf.append(train_rec.predict_cf_rating(str(row["user_id"]), str(row["item_id"]), ratings_subset=train_df))

    mae = float(mean_absolute_error(y_true, y_pred_cf)) if y_true else 0.0
    rmse = float(root_mean_squared_error(y_true, y_pred_cf)) if y_true else 0.0

    summary_rows = []
    for method_name, scores in method_precisions.items():
        mean_p = float(np.mean(scores)) if scores else 0.0
        summary_rows.append(
            {
                "Method": method_name,
                f"Precision@{k}": round(mean_p, 4),
                "Evaluated_Users": len(scores),
                "CF_MAE": round(mae, 4) if "CF" in method_name else None,
                "CF_RMSE": round(rmse, 4) if "CF" in method_name else None,
            }
        )

    metrics_df = pd.DataFrame(summary_rows)
    return metrics_df, {"mae": mae, "rmse": rmse, "n_train": len(train_df), "n_test": len(test_df)}


def generate_visualizations(
    items_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    results_dir: Path,
) -> Tuple[Path, Path]:
    """Generate and save 2 clean charts inside results/."""
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # Chart 1: Rating Distribution & Startup Sector Distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    rating_counts = ratings_df["interest_rating"].value_counts().sort_index()
    axes[0].bar(
        rating_counts.index.astype(str),
        rating_counts.values,
        color="#2B6CB0",
        edgecolor="black",
        alpha=0.9,
    )
    axes[0].set_title("Mentor Interest Rating Distribution (1-5)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Interest Rating")
    axes[0].set_ylabel("Number of Ratings")
    for idx, val in enumerate(rating_counts.values):
        axes[0].text(idx, val + 0.4, str(val), ha="center", va="bottom", fontsize=10)

    sector_counts = items_df["sector"].value_counts().sort_values(ascending=True)
    axes[1].barh(
        sector_counts.index,
        sector_counts.values,
        color="#2F855A",
        edgecolor="black",
        alpha=0.9,
    )
    axes[1].set_title("Startups by Technology Sector", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Number of Startups")
    axes[1].set_ylabel("Sector")
    for idx, val in enumerate(sector_counts.values):
        axes[1].text(val + 0.25, idx, str(val), va="center", fontsize=9)

    plt.tight_layout()
    chart1_path = results_dir / "rating_and_sector_distribution.png"
    plt.savefig(chart1_path, dpi=180)
    plt.close(fig)

    # Chart 2: Precision@5 Comparison Across Recommendation Methods
    fig, ax = plt.subplots(figsize=(9, 5))
    methods = metrics_df["Method"].tolist()
    prec_col = [c for c in metrics_df.columns if c.startswith("Precision@")][0]
    values = metrics_df[prec_col].tolist()
    colors = ["#718096", "#3182CE", "#D69E2E", "#38A169"]

    bars = ax.bar(methods, values, color=colors[: len(methods)], edgecolor="black", width=0.55)
    ax.set_title(f"Offline Evaluation: {prec_col} Across Methods", fontsize=12, fontweight="bold")
    ax.set_ylabel(prec_col)
    ax.set_ylim(0, max(values + [0.5]) * 1.2)
    plt.xticks(rotation=15, ha="right")

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.015,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    chart2_path = results_dir / "model_comparison_precision_at_5.png"
    plt.savefig(chart2_path, dpi=180)
    plt.close(fig)

    return chart1_path, chart2_path


def run_required_test_cases(
    base_items_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
) -> str:
    """
    Execute the 3 required test cases and format a clear report string:
      1. Idea query: 'AI assistant for clinic appointment scheduling and triage'
      2. Similar-startup test: Given 1 bookmarked startup (e.g., S01 MedPulse AI), find Top-5 similar startups
      3. Cold-start test: Add 1 newly submitted student startup pitch with zero ratings and verify matching
    """
    nlp = StartupNLPPipeline().fit(base_items_df)
    rec = StartupRecommender(base_items_df, ratings_df, nlp, min_ratings=2)

    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("CS467 MINI-PROJECT (MP-C11) - REQUIRED TEST CASES OUTPUT")
    lines.append("=" * 80)

    # Test Case 1: Idea Query
    query_1 = "AI assistant for clinic appointment scheduling and triage"
    tc1_df = rec.recommend_for_query(query_1, top_k=5)
    lines.append("\n[TEST CASE 1] Idea Query Recommendation")
    lines.append(f"Query: '{query_1}'")
    lines.append("-" * 60)
    for rank, row in tc1_df.iterrows():
        lines.append(
            f"  #{rank + 1} | {row['item_id']} - {row['startup_name']} ({row['sector']}, {row['stage']}) "
            f"| Sim Score: {row['score']:.4f} | Shared Keywords: {row['shared_keywords']}"
        )

    # Test Case 2: Similar-Startup Test
    bookmark_id = "S01"
    bookmark_row = base_items_df[base_items_df["item_id"] == bookmark_id].iloc[0]
    tc2_df = rec.recommend_similar_items(bookmark_id, top_k=5)
    lines.append("\n[TEST CASE 2] Similar-Startup Test (Bookmarked Item)")
    lines.append(f"Bookmarked Startup: {bookmark_id} - {bookmark_row['startup_name']} ({bookmark_row['sector']})")
    lines.append("-" * 60)
    for rank, row in tc2_df.iterrows():
        lines.append(
            f"  #{rank + 1} | {row['item_id']} - {row['startup_name']} ({row['sector']}, {row['stage']}) "
            f"| Sim Score: {row['score']:.4f} | Shared Keywords: {row['shared_keywords']}"
        )

    # Test Case 3: Cold-Start Item Test
    cold_item = {
        "item_id": "S41",
        "startup_name": "ClinicTriage Lite",
        "sector": "HealthTech",
        "stage": "Pre-Seed",
        "pitch_summary": (
            "Lightweight AI assistant for clinic appointment scheduling, symptom triage, "
            "and automated patient intake reminders."
        ),
        "tech_tags": "healthtech triage clinic scheduling nlp",
    }
    cold_nlp = StartupNLPPipeline().fit(base_items_df)
    updated_items = cold_nlp.add_cold_start_item(**cold_item)
    cold_rec = StartupRecommender(updated_items, ratings_df, cold_nlp, min_ratings=2)

    # Kiểm tra cả 2 chiều cho item cold-start (0 ratings):
    # (a) Truy vấn khớp với pitch của item mới
    tc3_query_df = cold_rec.recommend_for_query(cold_item["pitch_summary"], top_k=5)
    # (b) Các startup tương tự nhất với item mới S41
    tc3_sim_df = cold_rec.recommend_similar_items(cold_item["item_id"], top_k=5)

    lines.append("\n[TEST CASE 3] Cold-Start Startup Test (Zero Ratings)")
    lines.append(
        f"New Startup: {cold_item['item_id']} - {cold_item['startup_name']} "
        f"({cold_item['sector']}, {cold_item['stage']}) | Ratings count: 0"
    )
    lines.append(f"Pitch: {cold_item['pitch_summary']}")
    lines.append("  (a) Top-5 startups matching the new pitch query:")
    for rank, row in tc3_query_df.iterrows():
        lines.append(
            f"      #{rank + 1} | {row['item_id']} - {row['startup_name']} "
            f"| Sim Score: {row['score']:.4f} | Shared Keywords: {row['shared_keywords']}"
        )
    lines.append("  (b) Top-5 startups most similar to the zero-rating startup S41:")
    for rank, row in tc3_sim_df.iterrows():
        lines.append(
            f"      #{rank + 1} | {row['item_id']} - {row['startup_name']} "
            f"| Sim Score: {row['score']:.4f} | Shared Keywords: {row['shared_keywords']}"
        )

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)