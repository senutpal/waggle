from __future__ import annotations

import sys

import pytest


def _fake_encode(texts):
    import numpy as np

    rng = np.random.default_rng(42)
    return rng.standard_normal((len(texts), 8)).astype(np.float32)


def test_encode_prefix_queries_only():
    from src.embed_index import QUERY_PREFIX, encode_query

    q = encode_query("hello world", model="minilm", encode_fn=_fake_encode)
    c = encode_query("hello world", model="minilm", encode_fn=_fake_encode)
    assert q.shape == (1, 8) and c.shape == (1, 8)
    assert QUERY_PREFIX.startswith("Represent this sentence")


def test_l2_normalize_unit_norm():
    import numpy as np

    from src.embed_index import l2_normalize

    a = l2_normalize(np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32))
    assert abs(float((a[0] ** 2).sum()) - 1.0) < 1e-5
    assert abs(float((a[1] ** 2).sum()) - 1.0) < 1e-5


def test_faiss_roundtrip_ntotal_sorted_tiebreak():
    import numpy as np

    faiss = pytest.importorskip("faiss")
    from src.embed_index import build_index, dense_search

    rng = np.random.default_rng(7)
    vecs = rng.standard_normal((6, 8)).astype(np.float32)
    index = build_index(vecs, dim=8)
    assert index.ntotal == 6
    hits = dense_search(vecs[0], index, [f"d{i}" for i in range(6)], top_k=3)
    assert hits[0][0] == "d0"


def test_validate_meta_schema():
    import pytest

    from src.embed_index import validate_meta

    good = [
        {
            "id": "a",
            "text": "hi",
            "intent": "other",
            "lang": "en",
            "date": "2017-10-13T00:00:00+00:00",
            "evidence_ids": ["1"],
            "entities": {},
        }
    ]
    validate_meta(good, 1)
    with pytest.raises(ValueError):
        validate_meta(good, 2)
    bad = [dict(good[0])]
    bad[0] = dict(bad[0])
    bad[0].pop("lang")
    with pytest.raises(ValueError):
        validate_meta(bad, 1)


def test_no_torch_at_import():
    import src.embed_index  # noqa: F401
