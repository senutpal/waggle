from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import ROOT
from src.logging_utils import get_logger, sha256_file

logger = get_logger(__name__)

TWCS_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

TWCS_DTYPES = {
    "tweet_id": str,
    "author_id": str,
    "response_tweet_id": str,
    "in_response_to_tweet_id": str,
}

CREATED_AT_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def read_twcs(path: str | Path, nrows: int | None = None):
    import pandas as pd

    df = pd.read_csv(
        path,
        dtype=TWCS_DTYPES,
        keep_default_na=False,
        encoding="utf-8",
        engine="c",
        nrows=nrows,
    )
    return df


def normalize_inbound(series):
    if series.dtype == bool:
        return series
    mapped = series.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        bad = series[mapped.isna()].unique()[:10]
        raise ValueError(f"unmapped inbound values: {list(bad)}")
    return mapped.astype(bool)


def parse_created_at(series):
    import pandas as pd

    return pd.to_datetime(series, format=CREATED_AT_FORMAT, utc=True, errors="coerce")


def write_parquet(df, path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, engine="pyarrow", compression="snappy", index=False)
    sha = sha256_file(p)
    logger.info("wrote %s rows=%d sha=%s", p, len(df), sha)
    return sha


def log_rows_sha(name: str, path: str | Path) -> dict[str, Any]:
    p = Path(path)
    sha = sha256_file(p)
    size = p.stat().st_size
    logger.info("%s path=%s bytes=%d sha256=%s", name, p, size, sha)
    return {"path": str(p), "bytes": size, "sha256": sha}


def check_sample_schema(path: str | Path) -> None:
    import pandas as pd

    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8", nrows=5)
    missing = [c for c in TWCS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"sample.csv missing columns: {missing}")
    if ".0" in "".join(df.get("tweet_id", [])):
        raise ValueError("float join bug detected in sample.csv")


def repo_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (ROOT / p)
