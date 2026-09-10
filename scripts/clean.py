from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.logging_utils import get_logger
from src.normalize import mask, scan_pii, strip_signature, unescape
from src.seed import seed_everything

logger = get_logger(__name__)

EN_WORDS = frozenset(
    "the be to of and a in that have i it for not on with he as you do at this but his by from they we say her she or an will my one all would there their what so up out if about who get which go me when make can like time just him know take people into year your good some could them see other than then now look only come its over think also back after use two how our work first well way even new want because any these give day most us is are was were has had have please thanks thank sorry help need want dm message account order refund".split()
)


def heuristic_en_prob(text: str) -> float:
    if not text.strip():
        return 0.0
    toks = [t.strip(".,!?;:'\"()[]").lower() for t in text.split()]
    toks = [t for t in toks if t]
    if not toks:
        return 0.0
    ascii_ratio = sum(1 for ch in text if ord(ch) < 128) / max(1, len(text))
    hits = sum(1 for t in toks if t in EN_WORDS)
    word_score = min(1.0, hits / max(1, min(len(toks), 12)) * 2.0)
    return round(0.35 * ascii_ratio + 0.65 * word_score, 4)


def lid_scores(texts: list[str], keep_gt: float = 0.65):
    try:
        import fasttext

        model = fasttext.load_model(str(ROOT / "cache" / "lid.176.ftz"))
        keep: list[bool] = []
        probs: list[float] = []
        for t in texts:
            line = t.replace("\n", " ").strip()
            if not line:
                keep.append(False)
                probs.append(0.0)
                continue
            labels, scores = model.predict(line, k=1)
            p = float(scores[0])
            probs.append(round(p, 4))
            keep.append(str(labels[0]) == "__label__en" and p > keep_gt)
        return keep, probs, "fasttext-lid.176.ftz"
    except Exception as e:
        logger.info("fasttext unavailable (%s), heuristic fallback", e)
        probs = [heuristic_en_prob(t) for t in texts]
        keep = [p > keep_gt for p in probs]
        return keep, probs, "heuristic-fallback"


def sig_worker(args):
    i, text, num_perm, seed = args
    from datasketch import MinHash

    m = MinHash(num_perm=num_perm, seed=seed, scheme="affine32")
    low = text.lower()
    if len(low) < 5:
        m.update(low.encode("utf-8"))
    else:
        seen: set[bytes] = set()
        for k in range(len(low) - 4):
            b = low[k : k + 5].encode("utf-8")
            if b not in seen:
                seen.add(b)
                m.update(b)
    return i, list(m.hashvalues)


def main() -> int:
    seed_everything(42)
    import pandas as pd

    df = pd.read_parquet(ROOT / "cache" / "threads.parquet")
    n_turns_in = len(df)
    logger.info("turns_in=%d", n_turns_in)
    df["text_norm"] = df["text"].astype(str).map(lambda s: strip_signature(mask(unescape(s))))
    n_sig = int((df["text"].astype(str) != df["text_norm"]).sum())
    g = df.sort_values(["created_at", "thread_id"]).groupby("thread_id", sort=False)
    trows: list[dict] = []
    for tid, grp in g:
        grp = grp.sort_values("pos")
        trows.append(
            {
                "thread_id": tid,
                "thread_text": " | ".join(grp["text_norm"].astype(str).tolist()),
            }
        )
    th = pd.DataFrame(trows)
    n_threads_in = len(th)
    logger.info("threads_in=%d", n_threads_in)
    keep_lid, probs, method = lid_scores(th["thread_text"].tolist(), 0.65)
    th["lid_prob"] = probs
    band = th[(th["lid_prob"] >= 0.50) & (th["lid_prob"] <= 0.75)]
    apath = ROOT / "data-sample" / "lid_audit_200.json"
    apath.write_text(
        json.dumps(
            {"method": method, "n": int(min(200, len(band))), "items": band.head(200).to_dict(orient="records")},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    th = th[pd.Series(keep_lid, index=th.index)].copy()
    n_lid_drop = n_threads_in - len(th)
    logger.info("lid method=%s kept=%d dropped=%d band=%d", method, len(th), n_lid_drop, len(band))
    th["norm_key"] = th["thread_text"].map(lambda s: hashlib.sha256(" ".join(s.strip().lower().split()).encode("utf-8")).hexdigest())
    before = len(th)
    th = th.drop_duplicates(subset=["norm_key"]).reset_index(drop=True)
    n_exact = before - len(th)
    logger.info("exact removed=%d remain=%d", n_exact, len(th))
    import multiprocessing as mp

    from datasketch import MinHash, MinHashLSH

    texts = th["thread_text"].tolist()
    num_perm = 128
    with mp.Pool(processes=8) as pool:
        sigs = pool.map(sig_worker, [(i, t, num_perm, 42) for i, t in enumerate(texts)], chunksize=500)
    sigs.sort(key=lambda x: x[0])
    lsh = MinHashLSH(threshold=0.85, num_perm=num_perm)
    keep: list[int] = []
    for i, hv in sigs:
        m = MinHash(num_perm=num_perm, hashvalues=hv, scheme="affine32")
        if lsh.query(m):
            continue
        lsh.insert(str(i), m)
        keep.append(i)
    n_near = len(texts) - len(keep)
    logger.info("lsh kept=%d flagged=%d b=%d r=%d", len(keep), n_near, lsh.b, lsh.r)
    from sklearn.feature_extraction.text import TfidfVectorizer

    kept_texts = [texts[i] for i in keep]
    flagged_texts = [texts[i] for i in range(len(texts)) if i not in set(keep)]
    verified_dup = 0
    if flagged_texts:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True, min_df=2)
        K = vec.fit_transform(kept_texts)
        B = 2000
        for s in range(0, len(flagged_texts), B):
            F = vec.transform(flagged_texts[s : s + B])
            S = (F * K.T).toarray().max(axis=1)
            import numpy as np

            verified_dup += int((S > 0.85).sum())
    logger.info("near=%d verified_tfidf_gt085=%d", n_near, verified_dup)
    th = th.iloc[keep].copy()
    keep_ids = set(th["thread_id"].astype(str))
    clean = df[df["thread_id"].astype(str).isin(keep_ids)].copy()
    clean = clean.drop(columns=["text_norm"])
    df["text_norm"] = df["text"].astype(str).map(lambda s: strip_signature(mask(unescape(s))))
    norm_map = dict(zip(df["tweet_id"].astype(str), df["text_norm"].astype(str)))
    clean["text_norm"] = clean["tweet_id"].astype(str).map(norm_map)
    pii = scan_pii(clean["text_norm"].astype(str).tolist())
    logger.info("pii_norm=%s", pii)
    out = ROOT / "cache" / "clean_threads.parquet"
    clean.to_parquet(out, engine="pyarrow", compression="snappy", index=False)
    metrics = {
        "turns_in": int(n_turns_in),
        "threads_in": int(n_threads_in),
        "lid_method": method,
        "lid_threshold": 0.65,
        "lid_dropped": int(n_lid_drop),
        "lid_band_050_075": int(len(band)),
        "lid_audit_n": int(min(200, len(band))),
        "sig_changed_turns": int(n_sig),
        "exact_removed": int(n_exact),
        "near_removed": int(n_near),
        "near_verified_tfidf_gt085": int(verified_dup),
        "lsh_b": int(lsh.b),
        "lsh_r": int(lsh.r),
        "threads_out": int(len(th)),
        "turns_out": int(len(clean)),
        "pct_removed_threads": round(1 - len(th) / max(1, n_threads_in), 4),
        "pii_norm": pii,
    }
    mpath = ROOT / "manifests" / "clean.json"
    mpath.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if pii["email_hits"] or pii["phone_hits"]:
        raise SystemExit("PII residual in normalized text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
