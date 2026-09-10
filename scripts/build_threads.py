from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io import normalize_inbound, parse_created_at, read_twcs
from src.logging_utils import get_logger
from src.seed import seed_everything
from src.threads import reconstruct_threads, threads_to_frame

logger = get_logger(__name__)

BRAND = "AmazonHelp"


def main() -> int:
    seed_everything(42)
    import pandas as pd

    raw_path = ROOT / "data-raw" / "twcs.csv"
    df = read_twcs(raw_path)
    df["inbound"] = normalize_inbound(df["inbound"])
    df["created_at"] = parse_created_at(df["created_at"])
    df = df[df["created_at"].notna()].copy()
    logger.info("loaded rows=%d", len(df))
    seed_ids = set(df[(df["author_id"] == BRAND) & (~df["inbound"])]["tweet_id"].astype(str))
    logger.info("seeds=%d", len(seed_ids))
    keep: set[str] = set(seed_ids)
    prev = -1
    for it in range(10):
        new_keep: set[str] = set(keep)
        sub = df[df["tweet_id"].astype(str).isin(keep)]
        for r in sub.itertuples():
            p = str(r.in_response_to_tweet_id).strip()
            if p:
                new_keep.add(p)
            c = str(r.response_tweet_id).strip()
            if c:
                for x in c.split(","):
                    x = x.strip()
                    if x:
                        new_keep.add(x)
        mask_p = df["in_response_to_tweet_id"].astype(str).isin(keep)
        new_keep.update(df[mask_p]["tweet_id"].astype(str).tolist())
        if len(new_keep) == prev:
            break
        prev = len(new_keep)
        keep = new_keep
        logger.info("closure iter=%d size=%d", it, len(keep))
    sub = df[df["tweet_id"].astype(str).isin(keep)].copy()
    logger.info("closure rows=%d", len(sub))
    threads, stats = reconstruct_threads(sub, gap_hours=24.0, max_span_hours=48.0, max_nominal_turns=8)
    amz_threads = [t for t in threads if t["company"] == BRAND]
    stats["amazon_threads"] = len(amz_threads)
    stats["total_threads"] = len(threads)
    logger.info("threads total=%d amazon=%d avg=%.3f", len(threads), len(amz_threads), stats.get("avg_turns", 0))
    frame = threads_to_frame(amz_threads, sub)
    out = ROOT / "cache" / "threads.parquet"
    frame.to_parquet(out, engine="pyarrow", compression="snappy", index=False)
    logger.info("wrote %s rows=%d", out, len(frame))
    mpath = ROOT / "manifests" / "threads.json"
    mpath.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    samp = amz_threads[:20]
    spath = ROOT / "data-sample" / "threads_sample20.json"
    spath.write_text(json.dumps(samp, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, sort_keys=True))
    inv = int((pd.to_datetime(frame["created_at"], utc=True).sort_values().diff().dropna() < pd.Timedelta(0)).sum()) if len(frame) else 0
    assert inv == 0, "time inversion detected"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
