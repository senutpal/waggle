from __future__ import annotations

"""Dense index with metadata (S3.2).

``all-MiniLM-L6-v2`` default (``bge-small-en-v1.5`` deferred ablation,
query-prefix on QUERIES only when enabled). L2-normalize both sides then
``faiss.IndexFlatIP`` float32 (exact, CPU-only). Sidecar ``meta.json`` is a
``list[{id,text,intent,lang,date,evidence_ids,entities}]`` aligned with
FAISS rows ``0..ntotal-1`` — JSON, never pickle (``meta.pkl`` read-shim
only). All heavy imports lazy so ``import src`` stays fast and offline-safe.
"""

import json
import os
from pathlib import Path

from src import _lazy
from src.config import ROOT
from src.logging_utils import get_logger, sha256_file

logger = get_logger(__name__)

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

MODEL_REGISTRY = {
    "minilm": {"hf_id": "sentence-transformers/all-MiniLM-L6-v2", "prefix": None, "dim": 384},
    "bge-small": {"hf_id": "BAAI/bge-small-en-v1.5", "prefix": QUERY_PREFIX, "dim": 384},
}

META_FIELDS = {"id", "text", "intent", "lang", "date", "evidence_ids", "entities"}


def resolve_model(name: str = "minilm") -> dict:
    try:
        from src.config import load_config

        models = load_config().get("retrieval", {}).get("models", {})
        if name == "minilm" and "default" in models:
            return {
                "hf_id": models["default"].get("hf_id", MODEL_REGISTRY["minilm"]["hf_id"]),
                "prefix": None,
                "dim": int(models["default"].get("dim", 384)),
            }
    except Exception:
        pass
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name!r}")
    return dict(MODEL_REGISTRY[name])


def encode_corpus(texts: list[str], model: str = "minilm", batch_size: int = 64, encode_fn=None):
    """Encode documents (never query-prefixed). Returns float32 ndarray."""
    import numpy as np

    if encode_fn is not None:
        return np.asarray(encode_fn(texts), dtype=np.float32)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _lazy.require_extra("sentence_transformers", "pip install -r requirements-full.txt or run with --with-embeddings")
    from sentence_transformers import SentenceTransformer

    cfg = resolve_model(model)
    enc = SentenceTransformer(cfg["hf_id"])
    if enc.get_sentence_embedding_dimension() != cfg["dim"]:
        raise RuntimeError("unexpected embedding dim")
    return np.asarray(
        enc.encode(list(texts), batch_size=batch_size, normalize_embeddings=False, show_progress_bar=False),
        dtype=np.float32,
    )


def encode_query(text: str, model: str = "minilm", encode_fn=None):
    """Encode a query (BGE prefix applied here only, never on docs)."""
    import numpy as np

    cfg = resolve_model(model)
    t = (cfg["prefix"] or "") + (text or "") if cfg.get("prefix") else (text or "")
    if encode_fn is not None:
        return np.asarray(encode_fn([t]), dtype=np.float32)
    return encode_corpus([t], model=model, encode_fn=None)


def l2_normalize(mat):
    import numpy as np

    a = np.ascontiguousarray(mat, dtype=np.float32)
    faiss = _lazy.import_or_none("faiss")
    if faiss is not None:
        faiss.normalize_L2(a)
        return a
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (a / n).astype(np.float32)


def build_index(vecs, dim: int = 384):
    _lazy.require_extra("faiss", "pip install -r requirements-repro.txt (faiss-cpu)")
    import faiss

    import numpy as np

    xb = l2_normalize(np.ascontiguousarray(vecs, dtype=np.float32))
    index = faiss.IndexFlatIP(int(dim))
    index.add(xb)
    return index


def dense_search(query_vec, index, doc_ids: list[str], top_k: int = 50) -> list[tuple[str, float]]:
    import time

    import numpy as np

    t0 = time.perf_counter()
    xq = l2_normalize(np.ascontiguousarray(query_vec, dtype=np.float32).reshape(1, -1))
    k = min(top_k, len(doc_ids))
    if k <= 0:
        return []
    D, I = index.search(xq, k)
    order = sorted(range(k), key=lambda j: (-float(D[0, j]), str(doc_ids[int(I[0, j])])))
    ms = (time.perf_counter() - t0) * 1000.0
    logger.info("dense_search n=%d k=%d ms=%.1f", len(doc_ids), k, ms)
    return [(str(doc_ids[int(I[0, j])]), float(D[0, j])) for j in order]


def validate_meta(meta: list, ntotal: int) -> None:
    if not isinstance(meta, list):
        raise ValueError("meta must be a list")
    if len(meta) != ntotal:
        raise ValueError(f"meta len {len(meta)} != ntotal {ntotal}")
    if not meta:
        raise ValueError("empty meta")
    try:
        from src.golden import TAXONOMY
    except Exception:
        TAXONOMY = None
    seen: set[str] = set()
    for rec in meta:
        if not isinstance(rec, dict) or set(rec) != META_FIELDS:
            raise ValueError(f"bad meta record keys: {sorted(rec) if isinstance(rec, dict) else type(rec)}")
        for f in ("id", "text", "intent", "lang", "date"):
            if not isinstance(rec[f], str) or not rec[f]:
                raise ValueError(f"bad meta field {f}")
        if TAXONOMY is not None and rec["intent"] not in TAXONOMY:
            raise ValueError(f"bad intent {rec['intent']!r}")
        if not isinstance(rec["evidence_ids"], list) or not rec["evidence_ids"]:
            raise ValueError("bad evidence_ids")
        if not isinstance(rec["entities"], dict):
            raise ValueError("bad entities")
        if rec["id"] in seen:
            raise ValueError(f"dup id {rec['id']}")
        seen.add(rec["id"])


def save_index_meta(index, meta: list, model: str = "minilm", corpus_sha: str = "") -> dict[str, str]:
    import faiss

    paths = _index_paths()
    paths["index"].parent.mkdir(parents=True, exist_ok=True)
    tmp_idx = paths["index"].with_suffix(".index.tmp")
    faiss.write_index(index, str(tmp_idx))
    os.replace(tmp_idx, paths["index"])
    validate_meta(meta, index.ntotal)
    payload = json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp_meta = paths["meta"].with_suffix(".json.tmp")
    tmp_meta.write_text(payload, encoding="utf-8")
    os.replace(tmp_meta, paths["meta"])
    info = {
        "index": paths["index"].name,
        "index_sha": sha256_file(paths["index"]),
        "index_bytes": str(paths["index"].stat().st_size),
        "meta": paths["meta"].name,
        "meta_sha": sha256_file(paths["meta"]),
        "meta_bytes": str(paths["meta"].stat().st_size),
        "model": resolve_model(model)["hf_id"],
        "dim": str(index.d),
        "ntotal": str(index.ntotal),
        "faiss_version": faiss.__version__,
        "corpus_sha": corpus_sha,
    }
    logger.info("wrote index ntotal=%d dim=%d sha=%s", index.ntotal, index.d, info["index_sha"])
    return info


def _index_paths() -> dict[str, Path]:
    try:
        from src.config import load_config

        p = load_config().get("paths", {})
        return {"index": ROOT / p.get("index", "cache/faiss.index"), "meta": ROOT / p.get("meta", "cache/meta.json")}
    except Exception:
        return {"index": ROOT / "cache" / "faiss.index", "meta": ROOT / "cache" / "meta.json"}


def load_index_meta(path: Path | None = None):
    import faiss

    paths = _index_paths()
    idx_path = Path(path) if path else paths["index"]
    index = faiss.read_index(str(idx_path))
    meta_path = paths["meta"]
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        legacy = meta_path.with_suffix(".pkl")
        if legacy.exists():
            import pickle

            logger.warning("meta.json missing, falling back to legacy meta.pkl shim")
            with open(legacy, "rb") as f:
                meta = pickle.load(f)
        else:
            raise FileNotFoundError(f"missing {meta_path}")
    validate_meta(meta, index.ntotal)
    return index, meta


def cache_key(model: str = "minilm", corpus_sha: str = "") -> str:
    import faiss

    from src.logging_utils import sha256_text

    cfg = resolve_model(model)
    return sha256_text("|".join([cfg["hf_id"], cfg.get("prefix") or "", corpus_sha, faiss.__version__, str(cfg["dim"])]))
