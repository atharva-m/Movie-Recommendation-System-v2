#!/usr/bin/env python3
# ────────────────────────────────────────────────────────────────
# faiss_recall.py — MF candidate recall  +  optional content merge
# ────────────────────────────────────────────────────────────────
import argparse, json
from pathlib import Path
from typing  import List, Dict, Tuple

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm


# ─────────────────────────── utilities ──────────────────────────
def load_als(als_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (user_factors, item_factors) as float32."""
    data = np.load(als_path)
    return data["user_factors"].astype("float32"), data["item_factors"].astype("float32")


def build_ip_index(vecs: np.ndarray) -> faiss.IndexFlatIP:
    """L2-normalise & build a CPU IndexFlatIP (cosine on unit vectors)."""
    faiss.normalize_L2(vecs)
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    return idx


def mask_seen(cand: np.ndarray, seen: set) -> np.ndarray:
    """Filter out already-interacted items, preserve order."""
    keep = [x for x in cand if x not in seen]
    return np.asarray(keep, dtype=np.int64)


def recall_mf(u_row: int,
              U: np.ndarray,
              idx_mf: faiss.Index,
              R: int,
              seen: set) -> np.ndarray:
    """Top-R MF rows (row-indices!), never in *seen*."""
    q = U[u_row:u_row + 1].copy()
    faiss.normalize_L2(q)
    _, rows = idx_mf.search(q, R + len(seen) + 50)          # over-fetch
    return mask_seen(rows[0], seen)[:R]                     # row indices


def merge_lists(mf: np.ndarray,
                ct: np.ndarray,
                how: str,
                w_mf: float,
                w_ct: float,
                R: int) -> np.ndarray:
    """Union / interleave / weighted blend of two ranked lists."""
    if how == "union":
        out = list(mf) + [x for x in ct if x not in mf]
        return np.asarray(out[:R], dtype=np.int64)

    if how == "interleave":
        out: List[int] = []
        for a, b in zip(mf, ct):
            if a not in out:
                out.append(a)
            if len(out) >= R:
                break
            if b not in out:
                out.append(b)
            if len(out) >= R:
                break
        return np.asarray(out[:R], dtype=np.int64)

    if how == "weighted":
        rank_mf = {id_: len(mf) - i for i, id_ in enumerate(mf)}
        rank_ct = {id_: len(ct) - i for i, id_ in enumerate(ct)}
        scores: Dict[int, float] = {}
        for i, s in rank_mf.items():
            scores[i] = scores.get(i, 0) + w_mf * s
        for i, s in rank_ct.items():
            scores[i] = scores.get(i, 0) + w_ct * s
        best = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:R]
        return np.asarray([i for i, _ in best], dtype=np.int64)

    raise ValueError("merge strategy not recognised")


# ─────────────────────────── main routine ───────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description="ALS-based recall → candidate parquet")
    p.add_argument("--als",    default="models/als_model.npz")
    p.add_argument("--maps",   default="models/mf_mappings.json")
    p.add_argument("--seen",   required=True, help="ratings csv for seen-mask")
    p.add_argument("--user",   type=int)                       # optional single user
    p.add_argument("--topR",   type=int, default=300)

    # optional content blending
    p.add_argument("--content_ix")
    p.add_argument("--content_top", type=int, default=300)
    p.add_argument("--merge", choices=["union", "interleave", "weighted"],
                   default="union")
    p.add_argument("--weight_mf", type=float, default=0.5)
    p.add_argument("--weight_ct", type=float, default=0.5)

    p.add_argument("--output", default="data/processed/candidates.parquet")
    args = p.parse_args()

    # ── 1)  load matrices & mappings ───────────────────────────
    U, V = load_als(args.als)
    maps = json.loads(Path(args.maps).read_text())
    user2row = {int(u): int(r) for u, r in maps["user2row"].items()}
    item2row = {int(i): int(c) for i, c in maps["item2col"].items()}
    row2item = {r: i for i, r in item2row.items()}

    idx_mf = build_ip_index(V.copy())

    idx_ct = faiss.read_index(args.content_ix) if args.content_ix else None

    # ── 2)  seen-mask ──────────────────────────────────────────
    seen_map: Dict[int, set] = {}
    if args.seen:
        seen_df = pd.read_csv(args.seen, usecols=["userId", "movieId"])
        seen_map = seen_df.groupby("userId")["movieId"].apply(set).to_dict()

    # target users
    target = [args.user] if args.user is not None else list(seen_map.keys())

    # ── 3)  iterate users ──────────────────────────────────────
    out: List[Dict] = []
    for uid in tqdm(target, desc="recall"):
        if uid not in user2row:
            continue

        urow = user2row[uid]
        seen = seen_map.get(uid, set())

        # ----- MF recall (row indices) -------------------------
        mf_rows = recall_mf(urow, U, idx_mf, args.topR, seen)
        mf_items = np.vectorize(row2item.get)(mf_rows)

        # ----- optional content neighbours ---------------------
        if idx_ct is not None and mf_rows.size:
            seed_item_id = mf_items[0]                 # tmdbId
            seed_row     = item2row.get(seed_item_id)  # row in content index
            if seed_row is not None:
                seed_vec = idx_ct.reconstruct(seed_row).astype("float32")
                faiss.normalize_L2(seed_vec.reshape(1, -1))
                _, ct_rows = idx_ct.search(seed_vec.reshape(1, -1), args.content_top)
                ct_items = np.vectorize(row2item.get)(ct_rows[0])
            else:
                ct_items = np.empty(0, dtype=np.int64)
            cand = merge_lists(mf_items, ct_items, args.merge,
                               args.weight_mf, args.weight_ct, args.topR)
        else:
            cand = mf_items  # pure MF

        out.append({"userId": uid, "candidates": cand.tolist()})

    # ── 4)  write parquet ─────────────────────────────────────
    out_df = pd.DataFrame(out)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.output, index=False)
    print(f"→ saved {len(out_df):,} users  to  {args.output}")


# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
