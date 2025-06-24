#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────
# build_features.py  –  create LightGBM point-wise rows
#
#   • Works for either the training or validation split.
#   • Ensures ★ ≥ --pos-thresh are the only positives.
#   • Samples negatives deterministically via a seeded RNG.
#
# Example (training set)
# ──────────────────────
# python src/build_features.py \
#        --set train \
#        --ratings     data/tmp/train_ratings.csv \
#        --candidates  data/processed/candidates_train.parquet \
#        --movies      data/processed/movies_processed.csv \
#        --out-train   data/features/train_lgb.parquet \
#        --hard-neg    20 \
#        --easy-neg    10
# ────────────────────────────────────────────────────────────────
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Set

import numpy as np
import pandas as pd
from tqdm import tqdm


# ───────────────────────── helpers ──────────────────────────
def sample_easy_neg(seen: Set[int],
                    pop_items: np.ndarray,
                    n: int,
                    rng: np.random.Generator) -> List[int]:
    """
    Sample `n` unseen items with probability ∝ popularity.
    Mutates *seen* in-place to avoid duplicates.
    """
    if n == 0 or pop_items.size == 0:
        return []
    out: List[int] = []
    while len(out) < n:
        item = int(rng.choice(pop_items, 1)[0])
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def make_row(uid: int,
             item: int,
             label: int,
             meta: pd.Series | None,
             ts: int | None) -> Dict | None:
    """Return a single feature row or None if metadata is missing."""
    if meta is None:
        return None
    # recency = 9_999  # default far-past
    # if label and ts is not None:
    #     recency = (pd.Timestamp.now(tz="UTC") -
    #                pd.to_datetime(ts, unit="s", utc=True)).days
    return dict(
        userId       = uid,
        movieId      = item,
        label        = label,
        runtime_z    = meta["runtime_z"],
        lang_idx     = meta["lang_idx"],
        popularity   = meta["popularity"],
        vote_avg     = meta["vote_average"],
        vote_cnt     = meta["vote_count"],
        # recency_days = recency,
    )


# ───────────────────────── main ──────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--set", choices=["train", "valid"], required=True,
                    help="Which split to generate for")
    ap.add_argument("--ratings",    required=True,
                    help="CSV with userId,movieId,rating,timestamp")
    ap.add_argument("--candidates", required=True,
                    help="Parquet with userId,candidates list[int]")
    ap.add_argument("--movies",     default="data/processed/movies_processed.csv",
                    help="movies_processed.csv")
    ap.add_argument("--out-train",  default="none",
                    help="Parquet path for training rows")
    ap.add_argument("--out-valid",  default="none",
                    help="Parquet path for validation rows")
    ap.add_argument("--hard-neg",   type=int, default=20,
                    help="# of hard negatives per user")
    ap.add_argument("--easy-neg",   type=int, default=10,
                    help="# of easy (popularity-based) negatives per user")
    ap.add_argument("--pos-thresh", type=int, default=4,
                    help="Star rating ≥ THRESH counts as positive")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for deterministic sampling")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── load assets ───────────────────────────────────────────
    print("loading data …")
    ratings  = pd.read_csv(args.ratings)
    movies   = pd.read_csv(args.movies).set_index("id")
    cand_map = (pd.read_parquet(args.candidates)
                .set_index("userId")["candidates"].to_dict())
    pop_items = ratings["movieId"].value_counts().index.values

    rows: List[Dict] = []
    kept_users = 0

    # ── iterate users serially ───────────────────────────────
    for uid, grp in tqdm(ratings.groupby("userId"), unit="user", desc="users"):
        # ── positives (rating ≥ pos_thresh) ──────────────────
        pos_items = grp[grp["rating"] >= args.pos_thresh] \
                    .sort_values("timestamp")

        if pos_items.empty:
            continue  # skip users with no positives

        # ── items seen at all (used for masking) ─────────────
        seen_items = set(grp["movieId"].values)

        # ── build positive rows ──────────────────────────────
        pos_rows = []
        for it, ts in zip(pos_items["movieId"].values,
                          pos_items["timestamp"].values):
            if it in movies.index:
                meta = movies.loc[it]
                r = make_row(uid, it, 1, meta, ts)
                if r:
                    pos_rows.append(r)

        if not pos_rows:
            continue  # dropped due to metadata holes

        # ── negatives ────────────────────────────────────────
        hard_neg = [i for i in cand_map.get(uid, [])
                    if i not in seen_items and i in movies.index][:args.hard_neg]

        easy_neg_ids = sample_easy_neg(seen_items.copy(),
                                       pop_items,
                                       args.easy_neg,
                                       rng)
        easy_neg = [i for i in easy_neg_ids if i in movies.index]

        if not (hard_neg or easy_neg):
            # last-resort: one random unseen
            fallback = sample_easy_neg(seen_items.copy(),
                                       pop_items,
                                       1,
                                       rng)
            easy_neg = [i for i in fallback if i in movies.index]

        neg_rows = []
        for it in hard_neg + easy_neg:
            meta = movies.loc[it]
            neg_rows.append(make_row(uid, it, 0, meta, None))

        if not neg_rows:
            continue  # still no negatives

        rows.extend(pos_rows)
        rows.extend(neg_rows)
        kept_users += 1

    df = pd.DataFrame(rows)
    print(f"users kept : {kept_users:,}")
    print(f"rows       : {len(df):,}")

    # ── write parquet(s) ─────────────────────────────────────
    def _save(path: str, label: str):
        if str(path).lower() in {"none", "null", "nul", "/dev/null", "-"}:
            print(f"{label} set not written (null path)")
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"saved → {path}")

    if args.set == "train":
        _save(args.out_train, "train")
    else:
        _save(args.out_valid, "valid")


if __name__ == "__main__":
    main()
