# ────────────────────────────────────────────────────────────────
# vectorize.py ― build hybrid item-vectors  (tag-SVD ⊕ lang ⊕ runtime [⊕ year])
# ────────────────────────────────────────────────────────────────
# example:
#   python src/vectorize.py \
#          --processed  data/processed/movies_processed.csv \
#          --lang_json  models/lang_order.json \
#          --tfidf      models/tfidf_vectorizer.pkl \
#          --svd        models/svd_transformer.pkl \
#          --out        data/processed/X_hybrid.npy \
#          --svd_dim    400                 \
#          --add-year   10                  # optional decade feature
# ────────────────────────────────────────────────────────────────
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ─────────────────────────── helpers ────────────────────────────
def fit_tfidf(docs, min_df=3, max_features=20_000, ngram_range=(1, 2), stop_words="english"):
    vec = TfidfVectorizer(min_df=min_df, max_features=max_features, ngram_range=ngram_range, stop_words=stop_words, lowercase=True)
    X = vec.fit_transform(docs)
    return X, vec


def apply_svd(X, n_components=200, random_state=42):
    svd = TruncatedSVD(n_components=n_components, algorithm="randomized", random_state=random_state)
    X_red = svd.fit_transform(X).astype("float32")
    return X_red, svd


# ─────────────────────────── main ───────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed/movies_processed.csv",
                    help="movies_processed.csv with clean_tags/lang_idx/runtime_z")
    ap.add_argument("--lang_json", default="models/lang_order.json",
                    help="lang_order.json created in preprocessing")
    ap.add_argument("--tfidf", default="models/tfidf_vectorizer.pkl")
    ap.add_argument("--svd",   default="models/svd_transformer.pkl")
    ap.add_argument("--out",   default="data/processed/X_hybrid.npy")
    ap.add_argument("--id2row", default="models/id2row.json",
                    help="store list of TMDB ids in the exact row order")
    # hyper-params
    ap.add_argument("--min_df",       type=int, default=3)
    ap.add_argument("--max_features", type=int, default=20_000)
    ap.add_argument("--svd_dim",      type=int, default=400,
                    help="0 → skip SVD / keep full TF-IDF")
    ap.add_argument("--no_l2",        action="store_true",
                    help="skip final L2 normalisation")
    # optional year bucket feature
    ap.add_argument("--add-year", type=int, default=10,
                    metavar="SIZE",
                    help="bucket size in years (e.g. 10 → decade one-hot). "
                         "0 disables the feature.")
    args = ap.parse_args()

    # ── create output folders ──────────────────────────────────
    for p in (args.tfidf, args.svd, args.out, args.id2row):
        Path(p).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    # ── load metadata ───────────────────────────────────────────
    print("reading processed CSV …")
    df = pd.read_csv(args.processed)

    # guard: drop accidental duplicate IDs (keep first)
    before = len(df)
    df = df.drop_duplicates("id", keep="first")
    if len(df) != before:
        print(f"  • dropped {before-len(df)} duplicate movie id(s)")

    # save id order → later FAISS / recall can rely on a single source of truth
    Path(args.id2row).write_text(json.dumps(df["id"].tolist()))
    print(f"  saved id↔row order → {args.id2row}")

    # ── TF-IDF on clean_tags ───────────────────────────────────
    print("fitting TF-IDF …")
    X_tfidf, tfidf = fit_tfidf(df["clean_tags"].fillna(""),
                               min_df=args.min_df,
                               max_features=args.max_features)
    joblib.dump(tfidf, args.tfidf)
    print(f"  TF-IDF vectoriser saved → {args.tfidf}")

    # optional SVD compression
    if args.svd_dim > 0:
        print(f"applying Truncated SVD  (k = {args.svd_dim}) …")
        X_tags, svd = apply_svd(X_tfidf, n_components=args.svd_dim)
        joblib.dump(svd, args.svd)
        print(f"  SVD model saved       → {args.svd}")
    else:
        X_tags = X_tfidf.astype("float32")

    # ── language one-hot ───────────────────────────────────────
    lang_order = json.loads(Path(args.lang_json).read_text())
    n_lang = len(lang_order)
    lang_idx = df["lang_idx"].fillna(-1).astype(int).values
    X_lang = np.zeros((len(df), n_lang), dtype="float32")
    valid = (0 <= lang_idx) & (lang_idx < n_lang)
    X_lang[np.arange(len(df))[valid], lang_idx[valid]] = 1.0

    # ── runtime feature ────────────────────────────────────────
    X_rt = df["runtime_z"].fillna(0.0).to_numpy("float32").reshape(-1, 1)

    # ── optional year bucket feature ───────────────────────────
    extra_feats = []
    if args.add_year > 0 and "year" in df.columns:
        size = args.add_year
        years = df["year"].fillna(0).astype(int)
        bucket = ((years // size).clip(lower=0)).astype(int)
        n_bucket = bucket.max() + 1
        X_year = np.zeros((len(df), n_bucket), dtype="float32")
        X_year[np.arange(len(df)), bucket] = 1.0
        extra_feats.append(X_year)
        print(f"  added year one-hot ({n_bucket} buckets, size={size})")

    # ── concatenate all parts ──────────────────────────────────
    parts = [X_tags, X_lang, X_rt] + extra_feats
    X_hybrid = np.hstack(parts)

    # ── (optional) final L2 normalisation ──────────────────────
    if not args.no_l2:
        print("L2-normalising hybrid vectors …")
        X_hybrid = normalize(X_hybrid, copy=False)

    # save matrix
    np.save(args.out, X_hybrid)
    print(f"feature matrix shape={X_hybrid.shape}  →  {args.out}")

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
