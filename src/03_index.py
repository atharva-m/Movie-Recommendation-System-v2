# ────────────────────────────────────────────────────────────────
# index.py — build & save a FAISS index for the hybrid vectors
# ________________________________________________________________
# example:
#   python src/index.py \
#          --features  models/X_hybrid.npy \
#          --index     models/knn_hybrid.faiss \
#          --metric    cosine
# ────────────────────────────────────────────────────────────────
import argparse
from pathlib import Path
import numpy as np
import faiss

# ────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────
def build_index(x: np.ndarray, metric: str = "cosine"):
    """
    Build a *CPU* FAISS index
    """
    dim = x.shape[1]
    if metric.lower() in {"cosine", "ip"}:
        index = faiss.IndexFlatIP(dim)
    elif metric.lower() in {"l2", "euclidean"}:
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError("metric must be one of {'cosine','ip','l2'}")

    index.add(x)
    return index


# ────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Build a FAISS index for hybrid movie vectors")
    ap.add_argument("--features", default="data/processed/X_hybrid.npy", help=".npy produced by vectorize.py")
    ap.add_argument("--index",    default="models/knn_hybrid.faiss", help="where to save the index")
    ap.add_argument("--metric",   default="cosine", choices=["cosine", "ip", "l2"],
                    help="distance metric (cosine ↔ IP on normalised vectors)")

    args = ap.parse_args()

    # ── load vectors ────────────────────────────────────────────
    X = np.load(args.features).astype("float32")
    print(f"Loaded feature matrix  shape={X.shape}")

    # ── safety: L2-normalise if we’ll be using cosine/IP  ───────
    if args.metric in {"cosine", "ip"}:
        # `vectorize.py` *usually* L2-normalises, but if someone
        # ran it with `--no_l2`, we normalise here to guarantee
        # that cosine == dot-product on the unit sphere.
        print("Ensuring vectors are L2-normalised …")
        faiss.normalize_L2(X)

    # ── build CPU index first ──────────────────────────────────
    print(f"Building FAISS index  (metric = {args.metric}) …")
    cpu_index = build_index(X, metric=args.metric)
    print("CPU index built")


    # ── persist CPU index so any machine can load ──────────────
    Path(args.index).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(cpu_index, str(args.index))
    print(f"Saved FAISS index → {args.index}")

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
