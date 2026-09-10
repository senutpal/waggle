from __future__ import annotations


def test_tokenizer_keeps_codes_atomic():
    from src.tokenize import tokenize

    toks = tokenize("order ORD-12345 refund @bob see http://t.co/x 😡 #joke")
    assert "ord-12345" in toks
    assert "@USER" in toks
    assert "<URL>" in toks
    assert "😡" in toks
    assert "#joke" in toks and "joke" in toks


def test_light_stem_text_only():
    from src.tokenize import tokenize

    a = tokenize("refund")
    b = tokenize("refunds")
    c = tokenize("refunded")
    assert a == b == c
    # codes never stemmed
    assert tokenize("ORD-12345") == ["ord-12345"]


def test_bm25_exact_code_rank1():
    from src.bm25 import build_bm25, search_bm25
    from src.tokenize import tokenize

    docs = [
        tokenize("order ORD-12345 refund please"),
        tokenize("hello world tracking package"),
        tokenize("cancel my subscription now"),
    ]
    idx = build_bm25(docs, k1=0.8, b=0.4)
    hits = search_bm25(tokenize("ORD-12345"), idx, ["a", "b", "c"], top_k=3)
    assert hits[0][0] == "a"


def test_bm25_determinism_sorted_tiebreak():
    from src.bm25 import build_bm25, search_bm25

    docs = [["hello", "world"], ["hello", "world"], ["other", "words"]]
    idx = build_bm25(docs)
    h1 = search_bm25(["hello"], idx, ["b", "a", "c"], top_k=3)
    h2 = search_bm25(["hello"], idx, ["b", "a", "c"], top_k=3)
    assert [d for d, _ in h1] == [d for d, _ in h2]
