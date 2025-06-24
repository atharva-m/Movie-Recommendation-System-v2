#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────
# train_lgbm.py ― LightGBM LambdaRank re-ranker with strict checks
# ────────────────────────────────────────────────────────────────
import argparse, time, os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score, average_precision_score
from tqdm import tqdm
from sklearn.utils import check_array

# ═════════════════════════ helper ══════════════════════════════
def load_split(path: str):
    """Return (df, group_sizes  [list[int]])"""
    df = pd.read_parquet(path)
    # ensure every user has at least 1 pos & 1 neg – otherwise drop
    stats = df.groupby("userId")["label"].agg(["sum", "count"])
    ok    = stats[(stats["sum"] > 0) & (stats["sum"] < stats["count"])].index
    df    = df[df.userId.isin(ok)].reset_index(drop=True)

    grp_sizes = (
        df.groupby("userId", sort=False)["movieId"]
          .size()
          .astype("int32")
          .to_list()
    )
    assert sum(grp_sizes) == len(df), "group size mismatch"
    return df, grp_sizes


def lgb_dataset(df: pd.DataFrame,
                group_sizes,
                cat_feat_name: str):
    """Build a LightGBM Dataset; returns (dataset, feature_names, cat_index)."""
    # keep order!
    X_df = (
        df.drop(columns=["userId", "movieId", "label"])
          .reset_index(drop=True)
    )
    feat_names = X_df.columns.tolist()
    # index of categorical column
    cat_idx = [feat_names.index(cat_feat_name)]

    X = X_df.values.astype(np.float32)
    y = df["label"].astype(int).values

    dset = lgb.Dataset(
        X,
        label=y,
        group=group_sizes,
        free_raw_data=False,
        categorical_feature=cat_idx,
    )
    return dset, feat_names, cat_idx


def _dcg(rels: np.ndarray) -> float:
    """DCG with gains = rel (binary) and  log2(rank+1) discount."""
    return float((rels / np.log2(np.arange(2, rels.size + 2))).sum())


def eval_metrics(df_pred, k: int = 10) -> tuple[float, float]:
    """
    MAP@k and NDCG@k averaged over *all* users.

    • IDCG is computed from the *full* relevance vector (LightGBM style).
    • Users with no positives at all contribute 0 to both metrics
      (same as LightGBM’s built-in behaviour).
    """
    map_scores, ndcg_scores = [], []

    for _, g in tqdm(df_pred.groupby("userId"), desc="eval", unit="usr", leave=False):
        y_true = g["label"].to_numpy(dtype=np.int8, copy=False)
        y_pred = g["pred"].to_numpy(copy=False)

        if y_true.sum() == 0:              # safety guard, should be rare
            map_scores.append(0.0)
            ndcg_scores.append(0.0)
            continue

        order = y_pred.argsort()[::-1]
        topk  = order[:k]

        # ---------- MAP@k -------------------------------------
        hits = y_true[topk]
        if hits.sum() == 0:
            map_scores.append(0.0)
        else:
            cumsum = np.cumsum(hits)
            precision_at_i = cumsum / (np.arange(k) + 1)
            ap = (precision_at_i * hits).sum() / min(y_true.sum(), k)
            map_scores.append(float(ap))

        # ---------- NDCG@k ------------------------------------
        dcg  = _dcg(y_true[topk])

        # ideal DCG uses the *full* relevance vector, but we need only k best
        ideal_topk = np.sort(y_true)[::-1][:k]
        idcg = _dcg(ideal_topk)

        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(map_scores)), float(np.mean(ndcg_scores))


# ═════════════════════════ main ════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train LightGBM LambdaRank with robust checks")
    ap.add_argument("--train",   default="data/features/train_lgb.parquet")
    ap.add_argument("--valid",   default="data/features/valid_lgb.parquet")
    ap.add_argument("--model",   default="models/lgb_ranker.txt")
    ap.add_argument("--trees",   type=int,   default=500)
    ap.add_argument("--lr",      type=float, default=0.05)
    ap.add_argument("--ndcg_k",  type=int,   default=10)
    args = ap.parse_args()

    t0 = time.time()
    print("loading data …")
    train_df, train_grp = load_split(args.train)
    valid_df, valid_grp = load_split(args.valid)

    lgb_train, feat_names, cat_idx = lgb_dataset(train_df, train_grp, "lang_idx")
    lgb_valid, _, _                = lgb_dataset(valid_df, valid_grp, "lang_idx")

    print(f"train rows  : {len(train_df):,}   groups : {len(train_grp):,}")
    print(f"valid rows  : {len(valid_df):,}   groups : {len(valid_grp):,}")

    # ── LightGBM params ─────────────────────────────────────────
    params = dict(
        task               = "train",
        objective          = "lambdarank",
        metric             = "ndcg",
        ndcg_eval_at       = [args.ndcg_k],
        learning_rate      = args.lr,
        num_leaves         = 127,
        max_depth          = -1,
        feature_fraction   = 0.85,
        bagging_fraction   = 0.85,
        bagging_freq       = 1,
        min_data_in_leaf   = 1,         # ← avoid empty-leaf crash
        lambda_l2          = 0.0,
        verbose            = -1,
        deterministic      = True,
        seed               = 42,
        device_type        = "cpu",
    )

    print("training LightGBM …")
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=args.trees,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train", "valid"],
        callbacks=[lgb.log_evaluation(50),
                   lgb.early_stopping(stopping_rounds=50)],
    )

    # ── offline metrics ────────────────────────────────────────
    print("evaluating final model …")
    valid_df["pred"] = gbm.predict(
        valid_df[feat_names].to_numpy(np.float32),
        num_iteration=gbm.best_iteration,
    )
    mapk, ndcgk = eval_metrics(valid_df, k=args.ndcg_k)
    print(f"\nMAP@{args.ndcg_k}:  {mapk:.4f}")
    print(f"NDCG@{args.ndcg_k}: {ndcgk:.4f}")

    # ── save model ─────────────────────────────────────────────
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    gbm.save_model(args.model)
    print(f"saved → {args.model}")
    print(f"done  → {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
