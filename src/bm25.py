from __future__ import annotations

"""BM25 sparse retrieval over codes (S3.1).

Thin wrapper around ``rank-bm25.BM25Okapi`` with tweet-tuned defaults
``k1=0.8, b=0.4`` (NOT long-doc 1.2/0.75 — TF~1 in 20-40 tok tweets,
b=0.75 over-penalizes length). Tokenization lives in ``src.tokenize``;
this module owns index build / search / persistence helpers only.

Lazy ``rank-bm25`` import keeps ``import src`` fast and offline-safe.
"""

from pathlib import Path

from src.config import ROOT
from src.logging_utils import get_logger, sha256_file

logger = get_logger(__name__)


def default_params() -> dict[str, float]:
    try:
        from src.config import load_config

        r = load_config().get("retrieval", {}).get("bm25", {})
        return {"k1": float(r.get("k1", 0.8)), "b": float(r.get("b", 0.4))}
    except Exception:
        return {"k1": 0.8, "b": 0.4}


def build_bm25(tokenized_corpus: list[list[str]], k1: float | None = None, b: float | None = None):
    from rank_bm25 import BM25Okapi

    p = default_params()
    return BM25Okapi(tokenized_corpus, k1=p["k1"] if k1 is None else k1, b=p["b"] if b is None else b)


def search_bm25(
    query_tokens: list[str], index, doc_ids: list[str], top_k: int = 50
) -> list[tuple[str, float]]:
    import time

    import numpy as np

    t0 = time.perf_counter()
    scores = index.get_scores(query_tokens)
    arr = np.asarray(scores, dtype=float)
    k = min(top_k, len(doc_ids))
    if k <= 0:
        return []
    part = np.argpartition(arr, -k)[-k:]
    order = sorted(part.tolist(), key=lambda i: (-float(arr[i]), str(doc_ids[i])))
    ms = (time.perf_counter() - t0) * 1000.0
    logger.info("bm25_search n=%d k=%d ms=%.1f", len(doc_ids), k, ms)
    return [(str(doc_ids[i]), float(arr[i])) for i in order]


def save_bm25(index, path: Path | None = None) -> Path:
    import pickle

    dest = Path(path) if path else (ROOT / "cache" / "bm25.pkl")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(index, f, protocol=4)
    import os

    os.replace(tmp, dest)
    logger.info("wrote %s sha=%s", dest, sha256_file(dest))
    return dest


def load_bm25(path: Path | None = None):
    import pickle

    src = Path(path) if path else (ROOT / "cache" / "bm25.pkl")
    with open(src, "rb") as f:
        return pickle.load(f)
