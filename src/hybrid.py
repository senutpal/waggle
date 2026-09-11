from __future__ import annotations

"""Hybrid ranking: RRF fusion + cross-encoder rerank + MMR (S3.3).

Pipeline: ``BM25-50 + dense-50 (parallel) -> RRF 1/(60+rank) union -> keep 20
-> cross-encoder/ms-marco-MiniLM-L-6-v2 10->3 ONLY (batch 16-32, trunc 512)
-> MMR λ=0.7 k=3 (skip when pairwise>0.92)``.

RRF is scale-free (BM25 unbounded vs cosine [-1,1]); ranks are 1-indexed,
missing lists contribute 0. All orderings tie-break ``(-score, doc_id)``.
Cross-encoder is full-only + lazy; any failure falls back to lexical RRF
top-3 (fail-open, flagged).
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

from src import _lazy
from src.logging_utils import get_logger

logger = get_logger(__name__)

_cross_encoder = None


def rrf_fuse(rank_lists: list[list[str]], k: int = 60, top_n: int = 20) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rank_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[str(doc_id)] = scores.get(str(doc_id), 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]


def get_cross_encoder(model_id: str | None = None):
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _lazy.require_extra(
        "sentence_transformers", "pip install -r requirements-full.txt or run with --with-embeddings"
    )
    from sentence_transformers import CrossEncoder

    try:
        from src.config import load_config

        model_id = model_id or load_config().get("retrieval", {}).get("rerank", {}).get(
            "model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    except Exception:
        model_id = model_id or "cross-encoder/ms-marco-MiniLM-L-6-v2"
    _cross_encoder = CrossEncoder(model_id, device="cpu", max_length=512)
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    # warm-up (excluded from p95)
    try:
        _cross_encoder.predict([("warmup query", "warmup document")], show_progress_bar=False)
    except Exception:
        pass
    return _cross_encoder


def rerank(
    query: str, candidates: list[str], texts: dict[str, str], top_k: int = 3, batch_size: int = 16
) -> tuple[list[tuple[str, float]], dict]:
    """Rerank top-10 candidates -> top-k. Returns ((id,score).., info)."""
    cands = list(candidates)[:10]
    if not cands:
        return [], {"backend": "none", "fallback": True, "reason": "empty", "ms": 0.0}
    t0 = time.perf_counter()
    try:
        ce = get_cross_encoder()
        pairs = [(query, texts.get(d, d)) for d in cands]
        import numpy as np

        logits = np.asarray(
            ce.predict(pairs, batch_size=batch_size, show_progress_bar=False), dtype=float
        ).ravel()
        scores = 1.0 / (1.0 + np.exp(-logits))  # sigmoid to [0,1]; ranking unchanged
        order = sorted(range(len(cands)), key=lambda j: (-float(scores[j]), str(cands[j])))
        ms = (time.perf_counter() - t0) * 1000.0
        logger.info("rerank n=%d ms=%.1f backend=cross-encoder", len(cands), ms)
        return [(str(cands[j]), float(scores[j])) for j in order[:top_k]], {
            "backend": "cross-encoder",
            "fallback": False,
            "ms": round(ms, 1),
        }
    except Exception as e:
        ms = (time.perf_counter() - t0) * 1000.0
        reason = f"{type(e).__name__}: {str(e)[:120]}"
        logger.warning("rerank_fallback reason=%s", reason)
        return [(d, 0.0) for d in cands[:top_k]], {"backend": "lexical-rrf", "fallback": True, "reason": reason, "ms": round(ms, 1)}


def mmr_select(
    relevance: dict[str, float],
    vectors: dict[str, list[float]],
    lambda_: float = 0.7,
    top_k: int = 3,
    skip_pairwise: float = 0.92,
) -> tuple[list[str], dict]:
    """Greedy MMR over reranked pool. Skipped when max pairwise>threshold."""
    import numpy as np

    ids = sorted(relevance, key=lambda d: (-relevance[d], d))
    if not ids:
        return [], {"skipped": False, "reason": "empty"}
    vecs = {d: np.asarray(vectors[d], dtype=float) for d in ids if d in vectors}
    if len(vecs) >= 2:
        mat = np.stack([vecs[d] for d in ids if d in vecs])
        n = (np.linalg.norm(mat, axis=1, keepdims=True))
        n[n == 0] = 1.0
        mat = mat / n
        sim = mat @ mat.T
        mx = float(sim[np.triu_indices(len(mat), k=1)].max(initial=0.0))
        if mx > skip_pairwise:
            return ids[:top_k], {"skipped": True, "reason": f"pairwise {mx:.3f}>{skip_pairwise}", "max_pairwise": round(mx, 4)}
    selected: list[str] = [ids[0]]
    while len(selected) < min(top_k, len(ids)):
        best, best_s = None, None
        for d in ids:
            if d in selected or d not in vecs:
                continue
            import numpy as _np

            def _cos(a, b):
                na, nb = float(_np.linalg.norm(a)) or 1.0, float(_np.linalg.norm(b)) or 1.0
                return float(_np.dot(a, b) / (na * nb))

            red = max((_cos(vecs[d], vecs[s]) for s in selected if s in vecs), default=0.0)
            s = lambda_ * relevance[d] - (1 - lambda_) * red
            if best_s is None or s > best_s or (s == best_s and d < best):
                best, best_s = d, s
        if best is None:
            for d in ids:
                if d not in selected:
                    selected.append(d)
                    break
            else:
                break
        else:
            selected.append(best)
    # append any relevance-ranked leftovers if vectors missing
    for d in ids:
        if len(selected) >= top_k:
            break
        if d not in selected:
            selected.append(d)
    return selected[:top_k], {"skipped": False, "lambda": lambda_}


def hybrid_search(
    query: str,
    query_tokens: list[str],
    query_vec,
    bm25_index,
    dense_index,
    doc_ids: list[str],
    texts: dict[str, str],
    vectors: dict[str, list[float]] | None = None,
    per_leg: int = 50,
    fused_n: int = 20,
    rrf_k: int = 60,
    mmr_lambda: float = 0.7,
    batch_size: int = 16,
) -> dict:
    """Full S3.3 pipeline. Returns stages with IDs+scores for audit logging."""
    from src.bm25 import search_bm25
    from src.embed_index import dense_search

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_b = ex.submit(search_bm25, query_tokens, bm25_index, doc_ids, per_leg)
        f_d = ex.submit(dense_search, query_vec, dense_index, doc_ids, per_leg)
        bm25_hits, dense_hits = f_b.result(), f_d.result()
    bm25_ids = [d for d, _ in bm25_hits]
    dense_ids = [d for d, _ in dense_hits]
    fused = rrf_fuse([bm25_ids, dense_ids], k=rrf_k, top_n=fused_n)
    fused_ids = [d for d, _ in fused]
    ranked, rinfo = rerank(query, fused_ids, texts, top_k=3, batch_size=batch_size)
    rel = {d: s for d, s in ranked}
    # extend relevance with fused order for MMR pool stability
    for d, s in fused:
        rel.setdefault(d, 0.0)
    if vectors:
        final_ids, minfo = mmr_select(rel, vectors, lambda_=mmr_lambda, top_k=3)
        # keep rerank order for ids missing from vectors
        final_ids = [d for d in final_ids if d in [x for x, _ in ranked] or True][:3]
        if minfo.get("skipped"):
            final_ids = [d for d, _ in ranked[:3]]
    else:
        final_ids, minfo = [d for d, _ in ranked[:3]], {"skipped": True, "reason": "no-vectors"}
    return {
        "bm25": bm25_hits,
        "dense": dense_hits,
        "rrf": fused,
        "rerank": ranked,
        "rerank_info": rinfo,
        "final": final_ids,
        "mmr_info": minfo,
    }
