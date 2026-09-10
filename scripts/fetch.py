from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io import TWCS_COLUMNS, check_sample_schema, normalize_inbound, parse_created_at, read_twcs, write_parquet
from src.logging_utils import get_logger, sha256_file
from src.seed import seed_everything

logger = get_logger(__name__)

HANDLE = "thoughtvector/customer-support-on-twitter/versions/10"
BRAND = "AmazonHelp"


def main() -> int:
    seed_everything(42)
    import kagglehub

    raw_path = ROOT / "data-raw" / "twcs.csv"
    sha = sha256_file(raw_path)
    size = raw_path.stat().st_size
    df = read_twcs(raw_path)
    assert list(df.columns) == TWCS_COLUMNS
    df["inbound"] = normalize_inbound(df["inbound"])
    df["created_at_parsed"] = parse_created_at(df["created_at"])
    n_nat = int(df["created_at_parsed"].isna().sum())
    if n_nat:
        df = df[df["created_at_parsed"].notna()].copy()
    df["created_at"] = df["created_at_parsed"]
    df = df.drop(columns=["created_at_parsed"])
    n_true = int(df["inbound"].sum())
    n_false = int((~df["inbound"]).sum())
    n_resp_null = int((df["response_tweet_id"].astype(str).str.strip() == "").sum())
    n_reply_null = int((df["in_response_to_tweet_id"].astype(str).str.strip() == "").sum())
    amz_company = df[(df["author_id"] == BRAND) & (~df["inbound"])]
    seed_ids = set(amz_company["tweet_id"].astype(str))
    seed_parents: set[str] = set()
    seed_children: set[str] = set()
    for r in amz_company.itertuples():
        p = str(r.in_response_to_tweet_id).strip()
        if p:
            seed_parents.add(p)
        c = str(r.response_tweet_id).strip()
        if c:
            for x in c.split(","):
                x = x.strip()
                if x:
                    seed_children.add(x)
    mask_parent_in_seed = df["in_response_to_tweet_id"].astype(str).isin(seed_ids)
    mask_author_brand = df["author_id"] == BRAND
    child_of_seed = set(df[mask_parent_in_seed]["tweet_id"].astype(str))
    parent_of_seed_rows = df[df["tweet_id"].astype(str).isin(seed_parents)]
    child_rows = df[df["tweet_id"].astype(str).isin(seed_children)]
    amz_pool = df[
        mask_author_brand
        | mask_parent_in_seed
        | df["tweet_id"].astype(str).isin(seed_parents)
        | df["tweet_id"].astype(str).isin(seed_children)
    ].copy()
    logger.info("amazon_company_rows=%d pool_rows=%d", len(amz_company), len(amz_pool))
    sample = df.head(100)
    sample_path = ROOT / "data-sample" / "sample.csv"
    sample.to_csv(sample_path, index=False, encoding="utf-8")
    check_sample_schema(sample_path)
    cache_path = ROOT / "cache" / "amazon_raw.parquet"
    write_parquet(amz_pool, cache_path)
    fetch = {
        "handle": HANDLE,
        "kagglehub_version": kagglehub.__version__,
        "raw_path": "data-raw/twcs.csv",
        "bytes": size,
        "sha256": sha,
        "rows": int(len(df)),
        "cols": list(df.columns),
        "inbound_true": n_true,
        "inbound_false": n_false,
        "response_empty": n_resp_null,
        "in_reply_empty": n_reply_null,
        "nat_dates": n_nat,
        "amazon_company_rows": int(len(amz_company)),
        "amazon_pool_rows": int(len(amz_pool)),
        "brand": BRAND,
    }
    mpath = ROOT / "manifests" / "fetch.json"
    mpath.write_text(json.dumps(fetch, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(fetch, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
