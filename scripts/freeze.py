from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.logging_utils import get_logger, sha256_file
from src.manifest import cache_size_mb, write_json, write_manifest
from src.seed import seed_everything
from src.split import assert_gap

logger = get_logger(__name__)

CUT = "2017-10-15T00:00:00Z"
GAP_H = 24


def sig(args):
    i, t = args
    from datasketch import MinHash

    m = MinHash(num_perm=128, seed=42, scheme="affine32")
    low = t.lower()
    seen: set[bytes] = set()
    if len(low) < 5:
        m.update(low.encode())
    else:
        for k in range(len(low) - 4):
            b = low[k : k + 5].encode()
            if b not in seen:
                seen.add(b)
                m.update(b)
    return i, list(m.hashvalues)


def main() -> int:
    seed_everything(42)
    import pandas as pd

    df = pd.read_parquet(ROOT / "cache" / "clean_threads.parquet")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    g = df.groupby("thread_id", sort=False)
    starts = g["created_at"].min()
    ends = g["created_at"].max()
    cut_ts = pd.Timestamp(CUT, tz="UTC")
    gap = pd.Timedelta(hours=float(GAP_H))
    train_ids: list[str] = []
    golden_ids: list[str] = []
    quar_ids: list[str] = []
    for tid in starts.index:
        s, e = starts[tid], ends[tid]
        if s < cut_ts:
            if e >= cut_ts or e + gap >= cut_ts:
                quar_ids.append(tid)
            else:
                train_ids.append(tid)
        else:
            golden_ids.append(tid)
    train = df[df["thread_id"].isin(set(train_ids))].copy()
    golden = df[df["thread_id"].isin(set(golden_ids))].copy()
    quar = df[df["thread_id"].isin(set(quar_ids))].copy()
    max_train = pd.to_datetime(train["created_at"], utc=True).max()
    min_golden = pd.to_datetime(golden["created_at"], utc=True).min()
    delta_h = assert_gap(max_train, min_golden, GAP_H)
    logger.info("train_threads=%d golden_threads=%d quar_threads=%d delta_h=%.2f", len(train_ids), len(golden_ids), len(quar_ids), delta_h)
    train_text = train.sort_values(["thread_id", "pos"]).groupby("thread_id")["text_norm"].apply(lambda s: " | ".join(s.astype(str))).to_dict()
    golden_text = golden.sort_values(["thread_id", "pos"]).groupby("thread_id")["text_norm"].apply(lambda s: " | ".join(s.astype(str))).to_dict()
    train_keys = {hashlib.sha256(" ".join(v.strip().lower().split()).encode()).hexdigest(): k for k, v in train_text.items()}
    golden_keys = {hashlib.sha256(" ".join(v.strip().lower().split()).encode()).hexdigest(): k for k, v in golden_text.items()}
    exact_overlap = len(set(train_keys) & set(golden_keys))
    logger.info("exact_overlap=%d", exact_overlap)
    import multiprocessing as mp

    from datasketch import MinHash, MinHashLSH

    tids = list(train_text.keys())
    tvals = [train_text[k] for k in tids]
    with mp.Pool(processes=8) as pool:
        sigs = pool.map(sig, list(enumerate(tvals)), chunksize=500)
    lsh = MinHashLSH(threshold=0.85, num_perm=128)
    train_hv: dict[str, list] = {}
    for i, hv in sigs:
        train_hv[tids[i]] = hv
        lsh.insert(tids[i], MinHash(num_perm=128, hashvalues=hv, scheme="affine32"))
    gids = list(golden_text.keys())
    gvals = [golden_text[k] for k in gids]
    with mp.Pool(processes=8) as pool:
        gsigs = pool.map(sig, list(enumerate(gvals)), chunksize=500)
    flagged = 0
    jac_max = 0.0
    max_pair: dict = {}
    for gi, hv in gsigs:
        m = MinHash(num_perm=128, hashvalues=hv, scheme="affine32")
        hits = lsh.query(m)
        for h in hits:
            tm = MinHash(num_perm=128, hashvalues=train_hv[h], scheme="affine32")
            j = round(float(m.jaccard(tm)), 4)
            if j > jac_max:
                jac_max = round(j, 4)
                max_pair = {"train_id": str(h), "golden_id": str(gids[gi]), "jaccard": jac_max}
        if hits:
            flagged += 1
    logger.info("near_flagged=%d jac_max=%s", flagged, jac_max)
    train.to_parquet(ROOT / "cache" / "train.parquet", engine="pyarrow", compression="snappy", index=False)
    golden.to_parquet(ROOT / "cache" / "golden.parquet", engine="pyarrow", compression="snappy", index=False)
    for inter in ("amazon_raw.parquet", "threads.parquet"):
        p = ROOT / "cache" / inter
        if p.exists():
            p.unlink()
    rng = __import__("random").Random(42)
    sample_ids = rng.sample(sorted(golden_ids), min(200, len(golden_ids)))
    eval_df = golden[golden["thread_id"].isin(set(sample_ids))].copy()
    eval_df.to_parquet(ROOT / "data-sample" / "eval_200.parquet", engine="pyarrow", compression="snappy", index=False)
    pool_sha = sha256_file(ROOT / "cache" / "clean_threads.parquet")
    tsplit = {
        "version": "1.0.0",
        "cut": CUT,
        "gap_hours": GAP_H,
        "tz": "UTC",
        "train": {"rule": "thread_start < cut", "n_threads": len(train_ids), "n_turns": int(len(train)), "max_created_at": pd.Timestamp(max_train).isoformat()},
        "golden_pool": {"rule": "thread_start >= cut", "n_threads": len(golden_ids), "n_turns": int(len(golden)), "min_created_at": pd.Timestamp(min_golden).isoformat()},
        "quarantine": {"n_threads": len(quar_ids), "n_turns": int(len(quar)), "reason": "straddle cut-24h embargo"},
        "gap_assert": {"pass": True, "delta_hours": round(delta_h, 2)},
        "inputs": {"pool": "cache/clean_threads.parquet", "pool_sha256": pool_sha, "brand": "AmazonHelp"},
        "seed": 42,
    }
    write_json(tsplit, ROOT / "manifests" / "temporal_split.json")
    audit = {
        "version": "1.0.0",
        "cut": CUT,
        "gap_hours": GAP_H,
        "n_train": len(train_ids),
        "n_golden_pool": len(golden_ids),
        "n_golden_sample": int(len(sample_ids)),
        "exact_overlap": int(exact_overlap),
        "exact_method": "sha256(lower(normalize(text))) + thread granularity",
        "near_dup": {
            "method": "datasketch MinHash num_perm=128 char-5gram seed=42 scheme=affine32 + LSH threshold=0.85",
            "threshold_freeze": 0.85,
            "n_flagged": int(flagged),
            "near_dup_max": float(jac_max),
            "max_pair": max_pair,
            "disposition": "pass" if exact_overlap == 0 else "fail",
        },
        "id_overlap": {"thread_id_overlap": 0},
        "seed": 42,
        "pass": bool(exact_overlap == 0 and (max_train + gap < min_golden)),
    }
    write_json(audit, ROOT / "manifests" / "leakage_audit.json")
    cache_files = sorted((ROOT / "cache").glob("*"))
    write_manifest([p for p in cache_files if p.is_file()])
    size_mb = cache_size_mb()
    logger.info("cache_size_mb=%.1f", size_mb)
    print(json.dumps({"delta_h": round(delta_h, 2), "exact_overlap": exact_overlap, "flagged": flagged, "jac_max": jac_max, "cache_mb": round(size_mb, 1)}, indent=2))
    if exact_overlap != 0:
        raise SystemExit("exact cross-split overlap detected")
    if size_mb > 150:
        raise SystemExit(f"cache guard violated: {size_mb:.1f}MB > 150MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
