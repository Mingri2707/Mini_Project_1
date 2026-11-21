from __future__ import annotations

import re
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

# Giữ lại một số từ miền công nghệ/y tế/kinh doanh hữu ích khi cần
CUSTOM_STOP_WORDS = set(ENGLISH_STOP_WORDS) - {
    "AI",
    "ai",
    "system",
    "systems",
    "platform",
    "app",
    "apps",
    "tool",
    "tools",
}


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/special chars, and normalize whitespace."""
    if not isinstance(text, str) or not text.strip():
        return ""
    lowered = text.lower()
    # Thay thế dấu gạch nối trong tags như 'nlp-llm' hoặc 'zero-trust' bằng khoảng trắng để giữ cả token rời & ghép
    normalized = re.sub(r"[-_/]+", " ", lowered)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", normalized)
    collapsed = re.sub(r"\s+", " ", cleaned).strip()
    return collapsed


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize cleaned text and optionally remove English stop words."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    tokens = cleaned.split()
    if remove_stopwords:
        tokens = [tok for tok in tokens if tok not in CUSTOM_STOP_WORDS and len(tok) > 1]
    return tokens


def preprocess_for_tfidf(text: str) -> str:
    """Return a clean, stop-word-filtered string ready for TF-IDF."""
    return " ".join(tokenize(text, remove_stopwords=True))


class StartupNLPPipeline:
    """TF-IDF (unigrams + bigrams) pipeline for startup pitches, tags, and queries."""

    def __init__(self, max_features: int = 1500) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
        )
        self.items_df: pd.DataFrame | None = None
        self.tfidf_matrix = None
        self.feature_names: np.ndarray = np.array([])
        self.item_id_to_idx: dict[str, int] = {}

    def build_corpus_texts(self, df: pd.DataFrame) -> List[str]:
        """Combine sector, pitch_summary, and tech_tags into a rich preprocessed text."""
        corpus: List[str] = []
        for _, row in df.iterrows():
            sector_str = str(row.get("sector", ""))
            pitch_str = str(row.get("pitch_summary", ""))
            tags_str = str(row.get("tech_tags", ""))
            # Lặp nhẹ sector và tech_tags để tăng trọng số chủ đề miền
            combined = f"{sector_str} {sector_str} {pitch_str} {tags_str} {tags_str}"
            corpus.append(preprocess_for_tfidf(combined))
        return corpus

    def fit(self, items_df: pd.DataFrame) -> "StartupNLPPipeline":
        """Fit TF-IDF vectorizer on the startup items dataset."""
        self.items_df = items_df.reset_index(drop=True).copy()
        self.items_df["processed_text"] = self.build_corpus_texts(self.items_df)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.items_df["processed_text"])
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.item_id_to_idx = {
            str(item_id): idx for idx, item_id in enumerate(self.items_df["item_id"])
        }
        return self

    def transform_query(self, query_text: str):
        """Transform a free-text query or new startup pitch using the fitted vectorizer."""
        if self.tfidf_matrix is None:
            raise RuntimeError("NLP pipeline must be fitted before transforming queries.")
        processed = preprocess_for_tfidf(query_text)
        return self.vectorizer.transform([processed])

    def get_shared_keywords(
        self,
        vec_a,
        vec_b,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Compute top shared domain terms/bigrams between two TF-IDF vectors
        (supports both sparse matrices and dense numpy arrays).
        """
        arr_a = vec_a.toarray().ravel() if hasattr(vec_a, "toarray") else np.asarray(vec_a).ravel()
        arr_b = vec_b.toarray().ravel() if hasattr(vec_b, "toarray") else np.asarray(vec_b).ravel()
        shared_scores = arr_a * arr_b
        nonzero_indices = np.flatnonzero(shared_scores > 0)
        if len(nonzero_indices) == 0:
            return []
        sorted_idx = nonzero_indices[np.argsort(shared_scores[nonzero_indices])[::-1][:top_k]]
        return [(str(self.feature_names[i]), float(shared_scores[i])) for i in sorted_idx]

    def get_top_terms_for_item(self, item_id: str, top_k: int = 5) -> List[str]:
        """Return highest TF-IDF terms for a given item_id."""
        if self.tfidf_matrix is None or item_id not in self.item_id_to_idx:
            return []
        idx = self.item_id_to_idx[item_id]
        row = self.tfidf_matrix[idx].toarray().ravel()
        top_idx = np.argsort(row)[::-1][:top_k]
        return [str(self.feature_names[i]) for i in top_idx if row[i] > 0]

    def add_cold_start_item(
        self,
        item_id: str,
        startup_name: str,
        sector: str,
        stage: str,
        pitch_summary: str,
        tech_tags: str,
    ) -> pd.DataFrame:
        """
        Append a new unrated startup pitch and re-fit the TF-IDF matrix so cold-start
        content-based matching works seamlessly.
        """
        if self.items_df is None:
            raise RuntimeError("NLP pipeline must be fitted first.")
        new_row = pd.DataFrame(
            [
                {
                    "item_id": item_id,
                    "startup_name": startup_name,
                    "sector": sector,
                    "stage": stage,
                    "pitch_summary": pitch_summary,
                    "tech_tags": tech_tags,
                }
            ]
        )
        updated_df = pd.concat([self.items_df.drop(columns=["processed_text"], errors="ignore"), new_row], ignore_index=True)
        self.fit(updated_df)
        return self.items_df.copy()