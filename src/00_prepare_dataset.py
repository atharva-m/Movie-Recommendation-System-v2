# prepare_dataset.py ─ build all first-stage artefacts in one pass
#
# Outputs
# ───────
#  • data/processed/movies_processed.csv      ← cleaned movie metadata
#  • models/lang_order.json                   ← language frequency order
#  • data/processed/ratings_mapped.csv        ← MovieLens ratings mapped to TMDB ids
#  • data/tmp/train_ratings.csv               ← training split for users
#  • data/tmp/valid_ratings.csv               ← validation split for users
#
# Guarantees
# ──────────
#  ✓ every user in train / valid has ≥1 positive *and* ≥1 negative rating
#  ✓ no user appears in both splits
#  ✓ all columns required by downstream scripts are present
# ---------------------------------------------------------------------------
import argparse, json, ast
from collections import Counter
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import spacy
from sklearn.preprocessing import StandardScaler

# ────────────────────────────────────────────────────────────────────────────
# Load a lightweight spaCy pipeline for tokenization + lemmatization (no parsing or NER)
NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Safely parse stringified lists (JSON or Python literals)
def _safe_loads(x: str):
    """json.loads that falls back to ast.literal_eval when quotes are wrong"""
    try:
        return json.loads(x)
    except Exception:
        try:
            return ast.literal_eval(x)
        except Exception:
            return []

# Extract names from a list of dictionaries (if any)
def _names(objlist) -> List[str]:
    return [o.get("name", "") for o in objlist if isinstance(o, dict)]

# ────────────────────────────────────────────────────────────────────────────
def build_movies(args) -> pd.DataFrame:
    """Preprocess movie and credits data, generate tags and features"""
    print("• preprocessing movie metadata …")

    # Load raw metadata
    movies  = pd.read_csv(args.movies_csv)
    credits = pd.read_csv(args.credits_csv)

    # Merge movie and credit data
    df = movies.merge(credits, left_on="id", right_on="movie_id", how="inner")
    df.drop(columns=["movie_id"], inplace=True)

    # Fix title columns if necessary
    if "title" not in df.columns and "original_title" in df.columns:
        df.rename(columns={"original_title": "title"}, inplace=True)
    elif "original_title" in df.columns:
        df["title"] = df["title"].fillna(df["original_title"])
        df.drop(columns=["original_title"], inplace=True)

    cast_n = args.cast_n  # limit number of cast members per movie

    # Extract relevant tag fields
    genres_list   = df["genres" ].apply(lambda x: _names(_safe_loads(x)))
    keywords_list = df["keywords"].apply(lambda x: _names(_safe_loads(x)))
    cast_list     = df["cast"].apply(lambda x: _names(_safe_loads(x))[:cast_n])
    director_list = df["crew"].apply(
        lambda x: [m.get("name") for m in _safe_loads(x) if m.get("job")=="Director"]
    )
    country_iso   = df["production_countries"].apply(
        lambda x: (_safe_loads(x)[0].get("iso_3166_1") if _safe_loads(x) else None)
    )

    # Add new feature columns to the dataframe
    df["genres_list"] = genres_list
    df["country_iso"] = country_iso
    df["year"]        = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    # Combine all tags into a raw space-delimited string
    df["tags_raw"] = (
        genres_list + keywords_list + cast_list + director_list +
        country_iso.fillna("").apply(lambda x: [x])
    ).apply(
        lambda lst: " ".join([t.lower().replace(" ", "") for t in lst if t])
    )

    # Lemmatize + remove stopwords, punctuations, etc.
    lemmas = [
        [t.lemma_ for t in doc if not (t.is_stop or t.is_punct or t.is_space)]
        for doc in NLP.pipe(df["tags_raw"].tolist(), batch_size=1000)
    ]

    # Drop tokens that appear in too many movies (stopword-like)
    df_counts = Counter()
    for l in lemmas:
        df_counts.update(set(l))
    keep = {tok for tok,c in df_counts.items() if c <= args.df_cut * len(df)}

    # Final cleaned tags (filtered by doc frequency)
    df["clean_tags"] = [" ".join([tok for tok in l if tok in keep]) for l in lemmas]

    # Normalize runtime (z-score); treat missing/zero runtimes as NaN
    rt = df["runtime"].fillna(0).replace(0, np.nan)
    scaler = StandardScaler()
    df["runtime_z"] = scaler.fit_transform(rt.values.reshape(-1,1)).ravel()

    # Create language index based on frequency
    lang_order = df["original_language"].fillna("xx").value_counts().index.tolist()
    df["lang_idx"] = df["original_language"].fillna("xx").map(
        {l:i for i,l in enumerate(lang_order)}
    )

    # Save language order mapping to JSON
    Path(args.lang_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.lang_json).write_text(json.dumps(lang_order, indent=2))

    # Select final columns to keep
    cols_keep = [
        "id", "title",
        "genres_list",
        "clean_tags",
        "lang_idx", "runtime_z",
        "popularity", "vote_average", "vote_count",
        "original_language",
    ]
    out_p = Path(args.movies_out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df[cols_keep].to_csv(out_p, index=False)
    print(f"  ↳ saved movies → {out_p}")

    return df[["id"]]  # Return only IDs for later filtering

# ────────────────────────────────────────────────────────────────────────────
def map_ratings(args, valid_movie_ids: set) -> pd.DataFrame:
    """Map MovieLens movieIds to TMDB ids and filter to valid movies"""
    print("• mapping MovieLens ids → TMDB …")

    ratings = pd.read_csv(args.ratings_csv)          # MovieLens user ratings
    links   = pd.read_csv(args.links_csv,
                          usecols=["movieId", "tmdbId"]).dropna()

    # Ensure IDs are integers
    links["movieId"] = links["movieId"].astype(int)
    links["tmdbId"]  = links["tmdbId"].astype(int)

    # Merge links with ratings
    merged = ratings.merge(links, on="movieId", how="inner")

    # Rename tmdbId to movieId and keep useful columns
    remapped = (
        merged[["userId", "tmdbId", "rating", "timestamp"]]
        .rename(columns={"tmdbId": "movieId"})
    )

    # Filter to movies that exist in the processed metadata
    remapped = remapped[remapped["movieId"].isin(valid_movie_ids)]

    # Save to output
    out_p = Path(args.ratings_out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    remapped.to_csv(out_p, index=False)
    print(f"  ↳ saved mapped ratings → {out_p}  ({len(remapped):,} rows)")

    return remapped

# ────────────────────────────────────────────────────────────────────────────
def split_users(args, ratings: pd.DataFrame):
    """Split users into train and validation sets with integrity checks"""
    print("• user train / valid split …")

    users = ratings["userId"].unique()
    rng   = np.random.default_rng(args.seed)
    rng.shuffle(users)

    # Determine cutoff index
    cut = int(len(users) * (1 - args.valid_frac))
    train_users = set(users[:cut])
    valid_users = set(users[cut:])

    train = ratings[ratings.userId.isin(train_users)].copy()
    valid = ratings[ratings.userId.isin(valid_users)].copy()

    # Filter users who have both positive and negative ratings
    def keep(df):
        g = df.groupby("userId")["rating"]
        pos = g.apply(lambda x: (x >= 4).sum())   # positive = 4 or 5
        neg = g.apply(lambda x: (x  < 4).sum())   # negative = 1 to 3
        ok  = pos[(pos>0) & (neg>0)].index
        return df[df.userId.isin(ok)]

    train = keep(train)
    valid = keep(valid)

    # Save the splits
    Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.train_out, index=False)
    valid.to_csv(args.valid_out, index=False)

    print(f"  ↳ train users = {train.userId.nunique():,}  "
          f"valid users = {valid.userId.nunique():,}")
    print(f"  ↳ wrote → {args.train_out} / {args.valid_out}")

# ────────────────────────────────────────────────────────────────────────────
def cli() -> argparse.Namespace:
    """Parse command-line arguments for input/output paths and hyperparameters"""
    p = argparse.ArgumentParser(
        description="One-shot preparation of movies, ratings mapping and user split"
    )
    # Input files
    p.add_argument("--movies-csv",   default="data/raw/movies.csv")
    p.add_argument("--credits-csv",  default="data/raw/credits.csv")
    p.add_argument("--ratings-csv",  default="data/raw/ratings.csv")
    p.add_argument("--links-csv",    default="data/raw/links.csv")
    # Output files
    p.add_argument("--movies-out",   default="data/processed/movies_processed.csv")
    p.add_argument("--lang-json",    default="models/lang_order.json")
    p.add_argument("--ratings-out",  default="data/processed/ratings_mapped.csv")
    p.add_argument("--train-out",    default="data/tmp/train_ratings.csv")
    p.add_argument("--valid-out",    default="data/tmp/valid_ratings.csv")
    # Hyperparameters
    p.add_argument("--cast-n",     type=int,   default=5)
    p.add_argument("--df-cut",     type=float, default=0.50,
                   help="drop tokens whose doc-freq > DF_CUT share")
    p.add_argument("--valid-frac", type=float, default=0.20)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()

# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """Main routine: run all steps in sequence"""
    args = cli()

    movies_df = build_movies(args)                                  # Step 1
    ratings   = map_ratings(args, set(movies_df["id"].values))      # Step 2
    split_users(args, ratings)                                      # Step 3

# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()  # Entry point
