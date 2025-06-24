# 🎬 Movie Recommendation System

![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)
![License: CC0 1.0](https://img.shields.io/badge/license-CC0%201.0-lightgrey.svg)
![Status: production‑ready](https://img.shields.io/badge/status-production--ready-brightgreen)

A **two‑stage** movie recommender that first recalls candidates with FAISS (ALS + content hybrids) and then re‑ranks them with a LightGBM LambdaRank model.

---

## 🔍 Pipeline Overview

1. **Candidate Generation**
   • **ALS collaborative filtering** — captures user–item latent factors.
   • **TF‑IDF → SVD hybrids** — covers cold‑start items, indexed with **FAISS**.
3. **Re‑ranking**
   • Feature engineering on (user, item) pairs.
   • **LightGBM LambdaRank** scores each candidate list.

---

## 📂 Repository Structure (key bits)

```text
movie-recommender/
├─ src/
│   ├─ 00_prepare_dataset.py    # clean raw CSVs
│   ├─ 01_build_mf.py           # train ALS
│   ├─ 02_vectorize.py          # TF‑IDF + SVD
│   ├─ 03_index.py              # build FAISS index
│   ├─ 04_faiss_recall.py       # recall candidates
│   ├─ 05_build_features.py     # LightGBM feature tables
│   ├─ 06_train_lgbm.py         # train LambdaRank
│   └─ 07_eval_lgbm.py          # MAP/NDCG evaluation
│
├─ data/
│   ├─ raw/         # place original MovieLens CSV here
│   ├─ tmp/         # train/valid splits
│   ├─ processed/   # cleaned artefacts
│   └─ features/    # LightGBM matrices
├─ models/                      # Saved artefacts & model files
│   ├─ als_model.npz            # ALS factors
│   ├─ mf_mappings.json         # ID mappings
│   ├─ tfidf_vectorizer.pkl
│   ├─ svd_transformer.pkl
│   ├─ knn_hybrid.faiss
│   └─ lgb_ranker.txt           # LambdaRank models
│
├─ requirements.txt             # Python dependencies
└─ README.md                    # This file
```

---

## 🚀 Getting Started

### 1. Install

```bash
git clone https://github.com/atharva-m/movie-recommendation-system-v2.git
cd movie-recommendation-system-v2
pip install -r requirements.txt
```

### 2. Data

* Download **`ratings.csv`** from **MovieLens‑32M**: [https://grouplens.org/datasets/movielens/32m/](https://grouplens.org/datasets/movielens/32m/).
  Unzip the archive and place **`ratings.csv`** inside `data/raw/`.

### 3. End‑to‑End Commands

```bash
# 0️⃣  create raw/ and drop TMDB + MovieLens CSVs inside
python src/00_prepare_dataset.py

# 1️⃣  collaborative filter
python src/01_build_mf.py

# 2️⃣  content vectors + FAISS index
python src/02_vectorize.py
python src/03_index.py

# 3️⃣  candidate recall (run twice: train & valid)
python src/04_faiss_recall.py \
       --seen data/tmp/train_ratings.csv \
       --output data/processed/candidates_train.parquet
python src/04_faiss_recall.py \
       --seen data/tmp/valid_ratings.csv \
       --output data/processed/candidates_valid.parquet

# 4️⃣  supervised feature rows
python src/05_build_features.py --set train \
        --ratings     data/tmp/train_ratings.csv \
        --candidates  data/processed/candidates_train.parquet \
        --out-train   data/features/train_lgb.parquet
python src/05_build_features.py --set valid \
        --ratings     data/tmp/valid_ratings.csv \
        --candidates  data/processed/candidates_valid.parquet \
        --out-train   data/features/valid_lgb.parquet

# 5️⃣  LambdaRank re‑ranker
python src/06_train_lgbm.py

# 6️⃣  evaluate recommendations
python src/07_eval_lgbm.py \
        --features data/features/valid_lgb.parquet \
        --model    models/lgb_ranker.txt \
        --k 10
```

---

## 📈 Offline Metrics

* **MAP\@10:** 0.6490
* **NDCG\@10:** 0.7582  *(optimised target)*

### Why prefer NDCG on MovieLens‑32M?

MovieLens‑32M contains **many multi‑positive sessions** (users rate dozens of movies).  **MAP** treats all relevant items equally, ignoring their rank positions once they appear within the top‑k list.  **NDCG**, however, applies a logarithmic discount—rewarding systems that surface the *most* relevant (high‑rated) movies *earlier* in the list.  This better reflects real‑world UX where users are likelier to click the first few tiles rather than scroll through ten suggestions.

---

## ⏱️ Runtime

The full pipeline runs **CPU-only** in under **15 minutes** end-to-end.

---

## 📝 License

Released into the public domain under [CC0 1.0 Universal](http://creativecommons.org/publicdomain/zero/1.0/).

---

### Contact

Atharva Mokashi · atharvamokashi01@gmail.com · [LinkedIn](https://www.linkedin.com/in/atharva-m)
