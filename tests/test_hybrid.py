from __future__ import annotations


def test_rrf_union_and_order():
    from src.hybrid import rrf_fuse

    out = rrf_fuse([["a", "b"], ["b", "a"]], k=60, top_n=2)
    assert [d for d, _ in out] == ["a", "b"]  # tie -> sorted-doc-id
    out = rrf_fuse([["a", "x"], ["a", "y"]], k=60, top_n=3)
    assert out[0][0] == "a"  # consensus dual-middling beats single-best


def test_rrf_one_indexed_missing_zero():
    from src.hybrid import rrf_fuse

    out = dict(rrf_fuse([["a"], ["b"]], k=60, top_n=2))
    assert abs(out["a"] - 1 / 61) < 1e-12
    assert abs(out["b"] - 1 / 61) < 1e-12


def test_mmr_skip_on_near_dup():
    from src.hybrid import mmr_select

    rel = {"a": 0.9, "b": 0.89, "c": 0.5}
    vecs = {"a": [1.0, 0.0], "b": [1.0, 0.001], "c": [0.0, 1.0]}
    ids, info = mmr_select(rel, vecs, lambda_=0.7, top_k=3, skip_pairwise=0.92)
    assert info["skipped"] is True
    assert ids == ["a", "b", "c"]


def test_mmr_diversity_when_distinct():
    from src.hybrid import mmr_select

    rel = {"a": 0.9, "b": 0.89, "c": 0.88}
    vecs = {"a": [1.0, 0.0], "b": [0.99, 0.1], "c": [0.0, 1.0]}
    ids, info = mmr_select(rel, vecs, lambda_=0.7, top_k=2)
    assert ids[0] == "a"
    assert len(ids) == 2


def test_rerank_fallback_without_models(monkeypatch):
    import sys

    from src import hybrid

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    hybrid._cross_encoder = None
    monkeypatch.setattr(hybrid, "get_cross_encoder", lambda *a, **k: (_ for _ in ()).throw(ImportError("no st")))
    ranked, info = hybrid.rerank("q", ["a", "b", "c"], {"a": "x", "b": "y", "c": "z"}, top_k=2)
    assert info["fallback"] is True
    assert [d for d, _ in ranked] == ["a", "b"]
