# ────────────────────────────────────────────────────────────────
# eval_lgbm.py ― evaluate an existing LightGBM LambdaRank model
# ────────────────────────────────────────────────────────────────
# example:
#   python src/07_eval_lgbm.py \
#          --features data/features/valid_lgb.parquet \
#          --model    models/lgb_ranker.txt \
#          --k 10
# ──────────────────────────────────────────────────────────────── 
import argparse, time
import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────── 
# helpers
# ──────────────────────────────────────────────────────────────── 
def load_split(path: str):
    """Return (df, feature_names) with users that have ≥1 pos & neg"""
    df = pd.read_parquet(path)
    stats = df.groupby("userId")["label"].agg(["sum", "count"])
    ok    = stats[(stats["sum"] > 0) & (stats["sum"] < stats["count"])].index
    df    = df[df.userId.isin(ok)].reset_index(drop=True)

    feat_names = (
        df.drop(columns=["userId", "movieId", "label"])
          .columns
          .tolist()
    )
    return df, feat_names


def _dcg(rels: np.ndarray) -> float:
    return float((rels / np.log2(np.arange(2, rels.size + 2))).sum())


def eval_metrics(df_pred: pd.DataFrame, k: int = 10) -> tuple[float, float]:
    """MAP@k and NDCG@k averaged over users (binary relevance)"""
    map_scores, ndcg_scores = [], []

    for _, g in tqdm(df_pred.groupby("userId"), desc="eval", unit="usr", leave=False):
        y_true = g["label"].to_numpy(dtype=np.int8, copy=False)
        y_pred = g["pred"].to_numpy(copy=False)

        if y_true.sum() == 0:                 # shouldn't happen, guard anyway
            map_scores.append(0.0)
            ndcg_scores.append(0.0)
            continue

        order = y_pred.argsort()[::-1]
        topk  = order[:k]

        # ---------- MAP@k ----------
        hits = y_true[topk]
        if hits.sum() == 0:
            map_scores.append(0.0)
        else:
            cumsum = np.cumsum(hits)
            precision_at_i = cumsum / (np.arange(k) + 1)
            ap = (precision_at_i * hits).sum() / min(y_true.sum(), k)
            map_scores.append(float(ap))

        # ---------- NDCG@k ---------
        dcg  = _dcg(y_true[topk])
        ideal_topk = np.sort(y_true)[::-1][:k]
        idcg = _dcg(ideal_topk)
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(map_scores)), float(np.mean(ndcg_scores))


# ──────────────────────────────────────────────────────────────── 
# main
# ──────────────────────────────────────────────────────────────── 
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate a trained LightGBM LambdaRanker")
    ap.add_argument("--features", default="data/features/valid_lgb.parquet",
                    help="Parquet file produced by 05_build_features.py")
    ap.add_argument("--model",    default="models/lgb_ranker.txt",
                    help="LightGBM model file saved by train_lgbm.py")
    ap.add_argument("--k",        type=int, default=10,
                    help="Cut-off for MAP@k / NDCG@k")
    args = ap.parse_args()

    t0 = time.time()
    print("loading feature table …")
    df, feat_names = load_split(args.features)
    print(f"rows: {len(df):,}   users: {df.userId.nunique():,}")

    print(f"loading model → {args.model}")
    booster = lgb.Booster(model_file=args.model)

    print("predicting …")
    df["pred"] = booster.predict(
        df[feat_names].to_numpy(np.float32),
        num_iteration=booster.best_iteration or booster.num_trees()
    )

    print("computing metrics …")
    mapk, ndcgk = eval_metrics(df, k=args.k)
    print(f"\nMAP@{args.k}:  {mapk:.4f}")
    print(f"NDCG@{args.k}: {ndcgk:.4f}")
    print(f"done → {(time.time()-t0):.1f} s")

# ──────────────────────────────────────────────────────────────── 
if __name__ == "__main__":
    main()
