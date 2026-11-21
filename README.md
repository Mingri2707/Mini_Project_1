# Tech Startup and Product Idea Recommender (MP-C11)

- **Student Name:** Trần Bùi Minh Nhật
- **Student ID:** 29219021634
- **Section:** CS467C (No. 11)
- **Course:** CS467 Advanced Artificial Intelligence (2026–2027)
- **Project Title:** Tech Startup and Product Idea Recommender

## 1. Problem Statement

Mentors, students, and startup incubators browse innovation directories to discover relevant startups and validate product ideas across technology sectors. This project combines Natural Language Processing (TF-IDF over pitch summaries and technology tags) with recommendation techniques (Popularity Baseline, Item-Based Collaborative Filtering over mentor ratings, and a Hybrid ranker) to recommend startups and explain keyword matches.

## 2. Dataset Description

- `data/items.csv`: 40 technology startups across `HealthTech`, `FinTech`, `EdTech`, `AgriTech`, `CleanTech`, `Enterprise AI`, and `CyberSec` (`Pre-Seed`, `Seed`, `Series A`), with `pitch_summary` and `tech_tags`.
- `data/ratings.csv`: 14 mentors (`M01`–`M14`) providing interest ratings (`1`–`5`) and short review notes (`mentor_note`), yielding a sparse user-item interaction matrix.

## 3. NLP and Recommendation Methods

- **NLP Pipeline (`src/nlp_pipeline.py`):** Lowercases text, normalizes hyphens/punctuation, tokenizes, removes stop words, constructs TF-IDF (`ngram_range=(1, 2)`), transforms free-text queries, and extracts shared top domain keywords/bigrams.
- **Recommender Engine (`src/recommender.py`):**
  1. **Popularity Baseline:** Weighted rating score with a minimum rating count threshold (`min_ratings=2`), filtering out already-rated startups.
  2. **NLP Content-Based:** Cosine similarity on TF-IDF vectors for free-text idea queries, item-to-item similarity, and user profile matching.
  3. **Item-Based CF & Hybrid:** Item-to-item cosine similarity on mentor ratings to predict ratings, combined with NLP similarity and popularity prior in a hybrid score.

## 4. Installation and Run Commands

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
python -m src.main --user M01 --query "AI assistant for clinic appointment scheduling and triage"
```
