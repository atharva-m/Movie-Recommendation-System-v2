#!/usr/bin/env python3
# prepare_dataset.py ─ build all first-stage artefacts in one pass
#
# Outputs
# ───────
#  • data/processed/movies_processed.csv
#  • models/lang_order.json
#  • data/processed/ratings_mapped.csv
#  • data/tmp/train_ratings.csv
#  • data/tmp/valid_ratings.csv
#
# Guarantees
# ──────────
#  ✓ every user in train / valid has ≥1 positive *and* ≥1 negative
#  ✓ no user appears in both splits
#  ✓ all columns required by downstream scripts are present
# ---------------------------------------------------------------------------
import argparse, json, ast, re
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import spacy
from sklearn.preprocessing import StandardScaler

# ────────────────────────────────────────────────────────────────────────────
# small helpers
# ────────────────────────────────────────────────────────────────────────────
NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def _safe_loads(x: str):
    """json.loads that falls back to ast.literal_eval when quotes are wrong"""
    try:
        return json.loads(x)
    except Exception:
        try:
            return ast.literal_eval(x)
        except Exception:
            return []

def _names(objlist) -> List[str]:
    return [o.get("name", "") for o in objlist if isinstance(o, dict)]

# ────────────────────────────────────────────────────────────────────────────
def build_movies(args) -> pd.DataFrame:
    print("• preprocessing movie metadata …")

    movies  = pd.read_csv(args.movies_csv)
    credits = pd.read_csv(args.credits_csv)

    df = movies.merge(credits, left_on="id", right_on="movie_id", how="inner")
    df.drop(columns=["movie_id"], inplace=True)

    if "title" not in df.columns and "original_title" in df.columns:
        df.rename(columns={"original_title": "title"}, inplace=True)
    elif "original_title" in df.columns:
        df["title"] = df["title"].fillna(df["original_title"])
        df.drop(columns=["original_title"], inplace=True)

    cast_n = args.cast_n

    genres_list   = df["genres" ].apply(lambda x: _names(_safe_loads(x)))
    keywords_list = df["keywords"].apply(lambda x: _names(_safe_loads(x)))
    cast_list     = df["cast"].apply(lambda x: _names(_safe_loads(x))[:cast_n])
    director_list = df["crew"].apply(
        lambda x: [m.get("name") for m in _safe_loads(x) if m.get("job")=="Director"]
    )
    country_iso   = df["production_countries"].apply(
        lambda x: (_safe_loads(x)[0].get("iso_3166_1") if _safe_loads(x) else None)
    )

    df["genres_list"] = genres_list
    df["country_iso"] = country_iso
    df["year"]        = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    # raw tags
    df["tags_raw"] = (
        genres_list + keywords_list + cast_list + director_list +
        country_iso.fillna("").apply(lambda x: [x])
    ).apply(
        lambda lst: " ".join([t.lower().replace(" ", "") for t in lst if t])
    )

    # tokenise / lemmatise
    lemmas = [
        [t.lemma_ for t in doc if not (t.is_stop or t.is_punct or t.is_space)]
        for doc in NLP.pipe(df["tags_raw"].tolist(), batch_size=1000)
    ]

    # drop super-frequent tokens
    df_counts = Counter()
    for l in lemmas:
        df_counts.update(set(l))
    keep = {tok for tok,c in df_counts.items() if c <= args.df_cut * len(df)}

    df["clean_tags"] = [" ".join([tok for tok in l if tok in keep]) for l in lemmas]

    # runtime z-score
    rt = df["runtime"].fillna(0).replace(0, np.nan)
    scaler = StandardScaler()
    df["runtime_z"] = scaler.fit_transform(rt.values.reshape(-1,1)).ravel()

    # language buckets
    lang_order = df["original_language"].fillna("xx").value_counts().index.tolist()
    df["lang_idx"] = df["original_language"].fillna("xx").map(
        {l:i for i,l in enumerate(lang_order)}
    )

    Path(args.lang_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.lang_json).write_text(json.dumps(lang_order, indent=2))

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

    return df[["id"]]   # just for id look-ups later

# ────────────────────────────────────────────────────────────────────────────
def map_ratings(args, valid_movie_ids: set) -> pd.DataFrame:
    print("• mapping MovieLens ids → TMDB …")

    ratings = pd.read_csv(args.ratings_csv)          # userId, movieId, rating …
    links   = pd.read_csv(args.links_csv,
                          usecols=["movieId", "tmdbId"]).dropna()

    links["movieId"] = links["movieId"].astype(int)
    links["tmdbId"]  = links["tmdbId"].astype(int)

    merged = ratings.merge(links, on="movieId", how="inner")

    # ── keep only the columns we really need ───────────────────
    remapped = (
        merged[["userId", "tmdbId", "rating", "timestamp"]]      # or whatever cols exist
        .rename(columns={"tmdbId": "movieId"})                   # <- single, clean column
    )

    # ── filter to movies that exist in the processed metadata ──
    remapped = remapped[remapped["movieId"].isin(valid_movie_ids)]

    out_p = Path(args.ratings_out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    remapped.to_csv(out_p, index=False)
    print(f"  ↳ saved mapped ratings → {out_p}  ({len(remapped):,} rows)")

    return remapped

# ────────────────────────────────────────────────────────────────────────────
def split_users(args, ratings: pd.DataFrame):
    print("• user train / valid split …")

    users = ratings["userId"].unique()
    rng   = np.random.default_rng(args.seed)
    rng.shuffle(users)

    cut = int(len(users) * (1 - args.valid_frac))
    train_users = set(users[:cut])
    valid_users = set(users[cut:])

    train = ratings[ratings.userId.isin(train_users)].copy()
    valid = ratings[ratings.userId.isin(valid_users)].copy()

    # -------- invariants: ≥1 positive & ≥1 negative -------------
    def keep(df):
        g = df.groupby("userId")["rating"]
        pos = g.apply(lambda x: (x >= 4).sum())   # consider 4-5 stars “positive”
        neg = g.apply(lambda x: (x  < 4).sum())
        ok  = pos[(pos>0) & (neg>0)].index
        return df[df.userId.isin(ok)]

    train = keep(train)
    valid = keep(valid)

    Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.train_out, index=False)
    valid.to_csv(args.valid_out, index=False)

    print(f"  ↳ train users = {train.userId.nunique():,}  "
          f"valid users = {valid.userId.nunique():,}")
    print(f"  ↳ wrote → {args.train_out} / {args.valid_out}")

# ────────────────────────────────────────────────────────────────────────────
def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-shot preparation of movies, ratings mapping and user split"
    )
    # raw inputs
    p.add_argument("--movies-csv",   default="data/raw/movies.csv")
    p.add_argument("--credits-csv",  default="data/raw/credits.csv")
    p.add_argument("--ratings-csv",  default="data/raw/ratings.csv")
    p.add_argument("--links-csv",    default="data/raw/links.csv")
    # outputs
    p.add_argument("--movies-out",   default="data/processed/movies_processed.csv")
    p.add_argument("--lang-json",    default="models/lang_order.json")
    p.add_argument("--ratings-out",  default="data/processed/ratings_mapped.csv")
    p.add_argument("--train-out",    default="data/tmp/train_ratings.csv")
    p.add_argument("--valid-out",    default="data/tmp/valid_ratings.csv")
    # knobs
    p.add_argument("--cast-n",     type=int,   default=5)
    p.add_argument("--df-cut",     type=float, default=0.50,
                   help="drop tokens whose doc-freq > DF_CUT share")
    p.add_argument("--valid-frac", type=float, default=0.20)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()

# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = cli()

    movies_df = build_movies(args)
    ratings   = map_ratings(args, set(movies_df["id"].values))
    split_users(args, ratings)


if __name__ == "__main__":
    main()
