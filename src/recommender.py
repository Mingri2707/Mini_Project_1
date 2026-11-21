from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.nlp_pipeline import StartupNLPPipeline


class StartupRecommender:
    """
    Combines:
      1. Popularity Baseline (weighted rating with minimum rating count threshold)
      2. NLP Content-Based Recommender (TF-IDF cosine similarity + shared keyword explanation)
      3. Item-Based Collaborative Filtering + Hybrid Recommender
    """

    def __init__(
        self,
        items_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        nlp_pipeline: StartupNLPPipeline,
        min_ratings: int = 2,
    ) -> None:
        self.items_df = items_df.reset_index(drop=True).copy()
        self.ratings_df = ratings_df.copy()
        self.nlp = nlp_pipeline
        self.min_ratings = min_ratings

        self._user_item_matrix: pd.DataFrame | None = None
        self._item_sim_df: pd.DataFrame | None = None
        self._global_mean: float = 3.5
        self._prepare_collaborative_matrices()

    def _prepare_collaborative_matrices(self) -> None:
        if self.ratings_df.empty:
            self._user_item_matrix = pd.DataFrame()
            self._item_sim_df = pd.DataFrame()
            self._global_mean = 3.5
            return

        self._global_mean = float(self.ratings_df["interest_rating"].mean())
        piv = self.ratings_df.pivot_table(
            index="user_id",
            columns="item_id",
            values="interest_rating",
            aggfunc="mean",
        )
        all_item_ids = list(self.items_df["item_id"])
        piv = piv.reindex(columns=all_item_ids)
        self._user_item_matrix = piv

        # Item-item cosine similarity over centered/filled ratings (0 for unrated)
        filled = piv.fillna(0.0).values  # shape: (n_users, n_items)
        sim_matrix = cosine_similarity(filled.T)  # shape: (n_items, n_items)
        np.fill_diagonal(sim_matrix, 0.0)
        self._item_sim_df = pd.DataFrame(sim_matrix, index=all_item_ids, columns=all_item_ids)

    def update_data(self, items_df: pd.DataFrame, ratings_df: pd.DataFrame) -> None:
        self.items_df = items_df.reset_index(drop=True).copy()
        self.ratings_df = ratings_df.copy()
        self._prepare_collaborative_matrices()

    def get_rated_items(self, user_id: str) -> set[str]:
        if self.ratings_df.empty:
            return set()
        user_rows = self.ratings_df[self.ratings_df["user_id"] == user_id]
        return set(user_rows["item_id"].astype(str))

    # ------------------------------------------------------------------
    # 1. Popularity Baseline
    # ------------------------------------------------------------------
    def recommend_popularity(
        self,
        user_id: Optional[str] = None,
        top_k: int = 5,
        enforce_min_ratings: bool = True,
    ) -> pd.DataFrame:
        """
        Rank startups by weighted rating (IMDB-style prior) using minimum rating count threshold.
        Filters out items already rated by user_id if provided.
        """
        stats = (
            self.ratings_df.groupby("item_id")["interest_rating"]
            .agg(["count", "mean"])
            .reset_index()
        )
        merged = self.items_df.merge(stats, on="item_id", how="left")
        merged["count"] = merged["count"].fillna(0).astype(int)
        merged["mean"] = merged["mean"].fillna(self._global_mean)

        if enforce_min_ratings:
            eligible = merged[merged["count"] >= self.min_ratings].copy()
            if eligible.empty:
                eligible = merged.copy()
        else:
            eligible = merged.copy()

        m = float(max(self.min_ratings, 1))
        c = self._global_mean
        v = eligible["count"].astype(float)
        r = eligible["mean"].astype(float)
        eligible["score"] = (v / (v + m)) * r + (m / (v + m)) * c

        if user_id:
            seen = self.get_rated_items(user_id)
            eligible = eligible[~eligible["item_id"].isin(seen)]

        eligible = eligible.sort_values(
            by=["score", "count", "mean"], ascending=[False, False, False]
        ).head(top_k)

        return eligible[
            ["item_id", "startup_name", "sector", "stage", "count", "mean", "score"]
        ].rename(columns={"count": "rating_count", "mean": "avg_rating"})

    # ------------------------------------------------------------------
    # 2. NLP Content-Based Recommender
    # ------------------------------------------------------------------
    def recommend_for_query(self, query_text: str, top_k: int = 5) -> pd.DataFrame:
        """Recommend Top-K startups for a free-text query using TF-IDF cosine similarity."""
        query_vec = self.nlp.transform_query(query_text)
        sims = cosine_similarity(query_vec, self.nlp.tfidf_matrix).ravel()

        rows = []
        for idx, score in enumerate(sims):
            item_row = self.items_df.iloc[idx]
            item_vec = self.nlp.tfidf_matrix[idx]
            shared_kw = self.nlp.get_shared_keywords(query_vec, item_vec, top_k=5)
            kw_str = ", ".join([f"{kw} ({w:.2f})" for kw, w in shared_kw]) if shared_kw else "general domain match"
            rows.append(
                {
                    "item_id": item_row["item_id"],
                    "startup_name": item_row["startup_name"],
                    "sector": item_row["sector"],
                    "stage": item_row["stage"],
                    "score": float(score),
                    "shared_keywords": kw_str,
                }
            )

        res_df = pd.DataFrame(rows).sort_values(by="score", ascending=False).head(top_k)
        return res_df.reset_index(drop=True)

    def recommend_similar_items(self, item_id: str, top_k: int = 5) -> pd.DataFrame:
        """Given 1 startup item_id, find the Top-K most similar startups via TF-IDF."""
        if item_id not in self.nlp.item_id_to_idx:
            raise ValueError(f"Unknown item_id: {item_id}")
        target_idx = self.nlp.item_id_to_idx[item_id]
        target_vec = self.nlp.tfidf_matrix[target_idx]
        sims = cosine_similarity(target_vec, self.nlp.tfidf_matrix).ravel()

        rows = []
        for idx, score in enumerate(sims):
            if idx == target_idx:
                continue
            item_row = self.items_df.iloc[idx]
            item_vec = self.nlp.tfidf_matrix[idx]
            shared_kw = self.nlp.get_shared_keywords(target_vec, item_vec, top_k=5)
            kw_str = ", ".join([f"{kw} ({w:.2f})" for kw, w in shared_kw]) if shared_kw else "related sector"
            rows.append(
                {
                    "item_id": item_row["item_id"],
                    "startup_name": item_row["startup_name"],
                    "sector": item_row["sector"],
                    "stage": item_row["stage"],
                    "score": float(score),
                    "shared_keywords": kw_str,
                }
            )

        res_df = pd.DataFrame(rows).sort_values(by="score", ascending=False).head(top_k)
        return res_df.reset_index(drop=True)

    def _build_user_profile_vector(self, user_id: str, ratings_subset: Optional[pd.DataFrame] = None):
        """Weighted average of TF-IDF vectors for items rated by user_id."""
        df_use = ratings_subset if ratings_subset is not None else self.ratings_df
        user_ratings = df_use[df_use["user_id"] == user_id]
        if user_ratings.empty:
            return None

        weights = []
        indices = []
        for _, r_row in user_ratings.iterrows():
            iid = str(r_row["item_id"])
            if iid in self.nlp.item_id_to_idx:
                # Trọng số lấy rating - 2.0 (tối thiểu 0.5) để ưu tiên item được thích cao
                w = max(float(r_row["interest_rating"]) - 2.0, 0.5)
                indices.append(self.nlp.item_id_to_idx[iid])
                weights.append(w)

        if not indices:
            return None

        w_arr = np.array(weights, dtype=float)
        sub_mat = self.nlp.tfidf_matrix[indices].toarray()
        profile = np.average(sub_mat, axis=0, weights=w_arr)
        return profile.reshape(1, -1)

    def recommend_content_for_user(
        self,
        user_id: str,
        top_k: int = 5,
        ratings_subset: Optional[pd.DataFrame] = None,
        exclude_rated: bool = True,
    ) -> pd.DataFrame:
        """Content-based recommendations tailored to a user's rated item profile."""
        profile_vec = self._build_user_profile_vector(user_id, ratings_subset=ratings_subset)
        df_use = ratings_subset if ratings_subset is not None else self.ratings_df
        seen = set(df_use[df_use["user_id"] == user_id]["item_id"].astype(str)) if exclude_rated else set()

        if profile_vec is None:
            return self.recommend_popularity(user_id=user_id, top_k=top_k)

        sims = cosine_similarity(profile_vec, self.nlp.tfidf_matrix).ravel()
        rows = []
        for idx, score in enumerate(sims):
            item_row = self.items_df.iloc[idx]
            iid = str(item_row["item_id"])
            if iid in seen:
                continue
            item_vec = self.nlp.tfidf_matrix[idx]
            shared_kw = self.nlp.get_shared_keywords(profile_vec, item_vec, top_k=4)
            kw_str = ", ".join([kw for kw, _ in shared_kw]) if shared_kw else "profile match"
            rows.append(
                {
                    "item_id": iid,
                    "startup_name": item_row["startup_name"],
                    "sector": item_row["sector"],
                    "stage": item_row["stage"],
                    "score": float(score),
                    "shared_keywords": kw_str,
                }
            )

        res_df = pd.DataFrame(rows).sort_values(by="score", ascending=False).head(top_k)
        return res_df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Item-Based Collaborative Filtering & Hybrid Recommender
    # ------------------------------------------------------------------
    def predict_cf_rating(
        self,
        user_id: str,
        item_id: str,
        ratings_subset: Optional[pd.DataFrame] = None,
    ) -> float:
        """Predict rating [1-5] for (user_id, item_id) using Item-Based Collaborative Filtering."""
        if ratings_subset is not None:
            temp_rec = StartupRecommender(
                self.items_df, ratings_subset, self.nlp, min_ratings=self.min_ratings
            )
            return temp_rec.predict_cf_rating(user_id, item_id, ratings_subset=None)

        if self._user_item_matrix is None or self._item_sim_df is None:
            return self._global_mean

        if user_id not in self._user_item_matrix.index or item_id not in self._item_sim_df.index:
            return self._global_mean

        user_series = self._user_item_matrix.loc[user_id]
        rated_series = user_series.dropna()
        if rated_series.empty:
            return self._global_mean

        sims_to_target = self._item_sim_df.loc[item_id, rated_series.index]
        pos_mask = sims_to_target > 0
        if not np.any(pos_mask):
            # Fallback to user's average rating if no positive item similarity exists
            return float(rated_series.mean())

        num = float(np.sum(sims_to_target[pos_mask].values * rated_series[pos_mask].values))
        den = float(np.sum(np.abs(sims_to_target[pos_mask].values)))
        pred = num / den if den > 1e-9 else float(rated_series.mean())
        return float(np.clip(pred, 1.0, 5.0))

    def recommend_cf(
        self,
        user_id: str,
        top_k: int = 5,
        ratings_subset: Optional[pd.DataFrame] = None,
        exclude_rated: bool = True,
    ) -> pd.DataFrame:
        """Recommend Top-K startups using Item-Based Collaborative Filtering predicted ratings."""
        df_use = ratings_subset if ratings_subset is not None else self.ratings_df
        seen = set(df_use[df_use["user_id"] == user_id]["item_id"].astype(str)) if exclude_rated else set()

        rows = []
        for _, item_row in self.items_df.iterrows():
            iid = str(item_row["item_id"])
            if iid in seen:
                continue
            pred_rating = self.predict_cf_rating(user_id, iid, ratings_subset=ratings_subset)
            rows.append(
                {
                    "item_id": iid,
                    "startup_name": item_row["startup_name"],
                    "sector": item_row["sector"],
                    "stage": item_row["stage"],
                    "score": pred_rating,
                    "shared_keywords": "item-based CF rating prediction",
                }
            )

        res_df = pd.DataFrame(rows).sort_values(by="score", ascending=False).head(top_k)
        return res_df.reset_index(drop=True)

    def recommend_hybrid(
        self,
        user_id: str,
        top_k: int = 5,
        alpha_content: float = 0.45,
        beta_cf: float = 0.40,
        gamma_pop: float = 0.15,
        ratings_subset: Optional[pd.DataFrame] = None,
        exclude_rated: bool = True,
    ) -> pd.DataFrame:
        """
        Hybrid recommendation combining normalized NLP Content-Based similarity,
        Item-Based CF predicted rating, and Popularity prior.
        """
        df_use = ratings_subset if ratings_subset is not None else self.ratings_df
        seen = set(df_use[df_use["user_id"] == user_id]["item_id"].astype(str)) if exclude_rated else set()

        profile_vec = self._build_user_profile_vector(user_id, ratings_subset=ratings_subset)
        if profile_vec is not None:
            content_sims = cosine_similarity(profile_vec, self.nlp.tfidf_matrix).ravel()
        else:
            content_sims = np.zeros(len(self.items_df), dtype=float)

        pop_df = self.recommend_popularity(user_id=None, top_k=len(self.items_df), enforce_min_ratings=False)
        pop_map: Dict[str, float] = dict(zip(pop_df["item_id"], pop_df["score"]))
        max_pop = max(pop_map.values()) if pop_map else 5.0
        min_pop = min(pop_map.values()) if pop_map else 1.0
        pop_range = max(max_pop - min_pop, 1e-9)

        # Nếu user chưa có lịch sử rating (cold-start user), dồn trọng số sang content + popularity
        if profile_vec is None:
            w_c, w_cf, w_p = 0.70, 0.0, 0.30
        else:
            w_c, w_cf, w_p = alpha_content, beta_cf, gamma_pop

        rows = []
        for idx, item_row in self.items_df.iterrows():
            iid = str(item_row["item_id"])
            if iid in seen:
                continue

            c_score = float(content_sims[idx])  # already in [0, 1]
            cf_pred = self.predict_cf_rating(user_id, iid, ratings_subset=ratings_subset)
            cf_norm = float(np.clip((cf_pred - 1.0) / 4.0, 0.0, 1.0))  # map [1, 5] -> [0, 1]
            p_raw = pop_map.get(iid, self._global_mean)
            p_norm = float(np.clip((p_raw - min_pop) / pop_range, 0.0, 1.0))

            hybrid_score = w_c * c_score + w_cf * cf_norm + w_p * p_norm

            if profile_vec is not None:
                shared_kw = self.nlp.get_shared_keywords(profile_vec, self.nlp.tfidf_matrix[idx], top_k=3)
                kw_text = ", ".join([kw for kw, _ in shared_kw]) if shared_kw else "hybrid signal"
            else:
                kw_text = "popularity + sector prior"

            rows.append(
                {
                    "item_id": iid,
                    "startup_name": item_row["startup_name"],
                    "sector": item_row["sector"],
                    "stage": item_row["stage"],
                    "score": float(hybrid_score),
                    "cf_pred_rating": round(cf_pred, 2),
                    "content_sim": round(c_score, 3),
                    "shared_keywords": kw_text,
                }
            )

        res_df = pd.DataFrame(rows).sort_values(by="score", ascending=False).head(top_k)
        return res_df.reset_index(drop=True)