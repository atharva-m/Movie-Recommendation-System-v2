# 🎬 Movie Recommendation System

A **two-stage** recommendation engine combining FAISS-based recall with LightGBM re-ranking to deliver high-quality movie suggestions.

## 🔍 Pipeline Overview

1. **Candidate Generation**

   - **Collaborative Filtering (ALS)** produces user/item latent factors.
   - **Content Hybrids** use TF-IDF → SVD → FAISS index for cold-start recall.

2. **Re-ranking**

   - **Feature Engineering** creates rich interaction-level features for each candidate set.
   - **LambdaRank (LightGBM)** orders the top-K list by predicted relevance.

---

## 📂 Repository Structure

```
movie-recommender/
├─ src/                          # Core pipeline scripts
│   ├─ 00_prepare_dataset.py     # Ingest & clean raw CSVs
│   ├─ 01_build_mf.py            # Train ALS; save factors & mappings
│   ├─ 02_vectorize.py           # Build & save TF-IDF + SVD pipelines
│   ├─ 03_index.py               # Build FAISS index
│   ├─ 04_faiss_recall.py        # Recall top-N candidates
│   ├─ 05_build_features.py      # Generate LightGBM feature tables
│   └─ 06_train_lgbm.py          # Train LambdaRank model
│
├─ data/
│   ├─ raw/                      # Original CSVs (movies.csv, ratings.csv, links.csv, credits.csv)
│   ├─ processed/                # Cleaned & mapped artifacts (.csv, .npy, .parquet)
│   ├─ tmp/                      # Splits: train/valid ratings (.csv)
│   └─ features/                 # Final LightGBM tables (.parquet)
│
├─ models/                      # Saved artefacts & model files
│   ├─ als_model.npz            # ALS factors
│   ├─ mf_mappings.json         # ID mappings
│   ├─ tfidf_vectorizer.pkl
│   ├─ svd_transformer.pkl
│   ├─ knn_hybrid.faiss
│   └─ lgb_ranker.txt           # LambdaRank model
│
├─ requirements.txt             # Python dependencies
├─ LICENSE                      # CC0 1.0
└─ README.md                    # This file
```

---

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.13
- **pip** or **poetry**

### Installation

```bash
git clone https://github.com/your-org/movie-recommender.git
cd movie-recommender
pip install -r requirements.txt
```

### Data Preparation

1. Place the following CSV files into `data/raw/`:
   - `movies.csv`, `ratings.csv`, `links.csv`, `credits.csv`
2. Run the prep script:
   ```bash
   python src/00_prepare_dataset.py \
     --raw-dir data/raw \
     --out-processed data/processed \
     --out-tmp data/tmp
   ```

### Full Pipeline

```bash
# 1️⃣ Collaborative filter
python src/01_build_mf.py --input data/processed/movies_processed.csv \
    --ratings data/tmp/train_ratings.csv \
    --output models/als_model.npz \
    --mappings models/mf_mappings.json

# 2️⃣ Content indexing
python src/02_vectorize.py --movies data/processed/movies_processed.csv \
    --out-tfidf models/tfidf_vectorizer.pkl \
    --out-svd models/svd_transformer.pkl
python src/03_index.py --tfidf models/tfidf_vectorizer.pkl \
    --svd models/svd_transformer.pkl \
    --out-index models/knn_hybrid.faiss

# 3️⃣ Recall candidates (train & valid)
python src/04_faiss_recall.py --index models/knn_hybrid.faiss \
    --ratings data/tmp/train_ratings.csv \
    --out data/processed/candidates_train.parquet
# repeat for valid set

# 4️⃣ Feature engineering
python src/05_build_features.py --ratings data/tmp/train_ratings.csv \
    --candidates data/processed/candidates_train.parquet \
    --out data/features/train_lgb.parquet
# repeat for valid set

# 5️⃣ Train LambdaRank model
python src/06_train_lgbm.py --train data/features/train_lgb.parquet \
    --valid data/features/valid_lgb.parquet \
    --output models/lgb_ranker.txt
```

---

## ⚙️ Configuration & Tips

- **Paths & hyperparameters** can be overridden via CLI flags in each script.
- **Cache** large artefacts with DVC or `.gitignore` folders: `data/tmp/`, `data/features/`, `models/`.
- **Automate** with Airflow, Step Functions, or CI pipelines for retraining and deployment.

---

## 📝 License

This work is dedicated to the public domain under [CC0 1.0 Universal](http://creativecommons.org/publicdomain/zero/1.0/).

---

*Happy recommending!*

