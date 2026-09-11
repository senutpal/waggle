from __future__ import annotations

EVID = [
    "DM @AmazonHelp with order 123-4567890-1234567. Help: https://amazon.com/gp/help/x1 or 1-800-555-0147. Refund $12.99 reviewed 2017-10-15."
]


def test_bad_citation_extra():
    from src.ground import faithful

    r = faithful("See https://amazon.com/gp/help/x1.", ["D1", "D9"], ["D1", "D2"], EVID, ret_score=0.6)
    assert r.ok is False and r.reason == "BAD_CITATION"


def test_citation_subset_pass():
    from src.ground import faithful

    r = faithful(
        "See https://amazon.com/gp/help/x1.",
        ["D1"], ["D1", "D2"], EVID, ret_score=0.6,
        nli_entails=[0.45], nli_contras=[0.1],
    )
    assert r.ok is True


def test_entity_ground_fail_phone():
    from src.ground import faithful

    r = faithful("Call 1-800-555-9999.", ["D1"], ["D1"], EVID, ret_score=0.7,
                 nli_entails=[0.9], nli_contras=[0.0])
    assert r.ok is False and r.reason.startswith("ENTITY_GROUND_FAIL")


def test_nli_fail_tierv():
    from src.ground import faithful

    r = faithful("Refund approved per policy.", ["D1"], ["D1"], EVID, ret_score=0.6,
                 nli_entails=[-0.35] if False else [0.05], nli_contras=[0.2])
    assert r.ok is False


def test_abstain_ret_low_and_contra_high():
    from src.ground import faithful

    r1 = faithful("See https://amazon.com/gp/help/x1.", ["D1"], ["D1"], EVID, ret_score=0.22,
                  nli_entails=[0.9], nli_contras=[0.1])
    assert r1.ok is False and r1.reason == "INSUFFICIENT_EVIDENCE"
    r2 = faithful("See https://amazon.com/gp/help/x1.", ["D1"], ["D1"], EVID, ret_score=0.7,
                  nli_entails=[0.9], nli_contras=[0.75])
    assert r2.ok is False and r2.reason == "INSUFFICIENT_EVIDENCE"


def test_any_fail_not_all_fail():
    from src.ground import faithful

    # invented phone alongside passing URL -> ANY-fail must reject
    r = faithful("See https://amazon.com/gp/help/x1 or call 1-800-111-2222.", ["D1"], ["D1"], EVID,
                 ret_score=0.7, nli_entails=[0.9], nli_contras=[0.0])
    assert r.ok is False


def test_tier_b_exempt_and_mixed():
    from src.ground import faithful

    ok = faithful("Sorry to hear this \u2014 please DM @Support with details.", [], ["D1"], EVID)
    assert ok.ok is True and ok.reason == "TIER_B_EXEMPT"
    mixed = faithful("Sorry \u2014 DM us. Your refund is $99 at https://amazon.com/gp/help/x9.", [], ["D1"], EVID)
    assert mixed.ok is False


def test_semantic_backstop_paraphrase():
    from src.ground import faithful

    r = faithful("You'll get money back tomorrow.", ["D1"], ["D1"], ["DM us for options."],
                 ret_score=0.6, nli_entails=[0.05], nli_contras=[0.1])
    assert r.ok is False


def test_nli_skip_fail_closed(monkeypatch):
    import sys

    from src import ground

    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setattr(ground, "get_nli", lambda *a, **k: (_ for _ in ()).throw(ImportError("no st")))
    r = ground.faithful("See https://amazon.com/gp/help/x1.", ["D1"], ["D1"], EVID, ret_score=0.6)
    assert r.ok is False and r.reason == "INSUFFICIENT_EVIDENCE"
