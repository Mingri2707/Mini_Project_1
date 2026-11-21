from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluate import evaluate_models, generate_visualizations, run_required_test_cases
from src.nlp_pipeline import StartupNLPPipeline
from src.recommender import StartupRecommender


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tech Startup and Product Idea Recommender (CS467 MP-C11)"
    )
    parser.add_argument(
        "--user",
        type=str,
        default="M01",
        help="Mentor user_id for personalized/hybrid recommendations (e.g., M01..M14)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="AI assistant for clinic appointment scheduling and triage",
        help="Free-text startup idea query to test NLP content-based matching",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    items_df = pd.read_csv(data_dir / "items.csv")
    ratings_df = pd.read_csv(data_dir / "ratings.csv")

    print(f"Loaded {len(items_df)} startups and {len(ratings_df)} mentor ratings.")

    # Fit NLP pipeline & Recommender
    nlp = StartupNLPPipeline().fit(items_df)
    rec = StartupRecommender(items_df, ratings_df, nlp, min_ratings=2)

    # 1. Popularity Baseline for user
    print(f"\n--- Top-5 Popularity Baseline (User: {args.user}) ---")
    pop_df = rec.recommend_popularity(user_id=args.user, top_k=5)
    print(pop_df.to_string(index=False))

    # 2. NLP Content-Based for custom query
    print(f"\n--- Top-5 NLP Content-Based (Query: '{args.query}') ---")
    cb_df = rec.recommend_for_query(args.query, top_k=5)
    print(cb_df.to_string(index=False))

    # 3. Hybrid Recommender for user
    print(f"\n--- Top-5 Hybrid Recommender (User: {args.user}) ---")
    hyb_df = rec.recommend_hybrid(user_id=args.user, top_k=5)
    print(hyb_df.to_string(index=False))

    # 4. Offline Evaluation & Charts
    metrics_df, extra_stats = evaluate_models(items_df, ratings_df, nlp, relevance_threshold=4, k=5)
    print("\n--- Offline Evaluation Summary ---")
    print(metrics_df.to_string(index=False))
    print(
        f"Split sizes -> Train: {extra_stats['n_train']} ratings | "
        f"Test: {extra_stats['n_test']} ratings | "
        f"CF MAE: {extra_stats['mae']:.4f} | CF RMSE: {extra_stats['rmse']:.4f}"
    )

    chart1, chart2 = generate_visualizations(items_df, ratings_df, metrics_df, results_dir)
    print(f"\nSaved charts to: {chart1.name}, {chart2.name}")

    # 5. Run & Save Required 3 Test Cases
    test_report = run_required_test_cases(items_df, ratings_df)
    sample_out_path = results_dir / "sample_outputs.txt"
    sample_out_path.write_text(test_report, encoding="utf-8")
    print(f"Saved required test cases output to: {sample_out_path}")


if __name__ == "__main__":
    main()