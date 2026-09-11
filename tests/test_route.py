from __future__ import annotations


def test_q0_inbound_only():
    from src.route import build_q0

    assert build_q0(["a", "b"]) == "a | b"
    assert build_q0([]) == ""


def test_followup_triggers():
    from src.route import is_followup_short

    flag, _ = is_followup_short("Yes 10 minutes ago! Don't want it", 2)
    assert flag is True
    flag, _ = is_followup_short("x" * 200, 1)
    assert flag is False


def test_q1_strips_apology_keeps_action():
    from src.route import build_q1

    q1 = build_q1("where is my refund", "Sorry to hear! Your refund was issued, see <URL>. DM us anytime.")
    assert "Sorry" not in q1
    assert "refund" in q1.lower()
    assert "[SEP]" in q1
    # empty action -> q1 == last inbound
    assert build_q1("hi", "") == "hi"


def test_soft_boost_intent_only_when_match():
    from src.route import soft_boost

    doc = {"intent": "shipping-tracking", "lang": "en", "date": "2017-10-13T00:00:00+00:00", "evidence_ids": ["1"]}
    f1, parts = soft_boost(0.6, doc, "shipping-tracking", "2017-10-14T00:00:00+00:00")
    f2, _ = soft_boost(0.6, doc, "other", "2017-10-14T00:00:00+00:00")
    assert parts["intent"] == 0.2
    assert f1 > f2
    # other is neutral
    f3, p3 = soft_boost(0.6, {"intent": "other", "lang": "en", "date": "", "evidence_ids": ["1"]}, "other", "")
    assert p3["intent"] == 0.0
    # missing date -> 0, never penalize
    assert p3["date"] == 0.0


def test_deprecated_block_and_backoff():
    from src.route import apply_routing

    docs = {
        "a": {"intent": "other", "lang": "en", "date": "", "evidence_ids": ["ea"]},
        "b": {"intent": "other", "lang": "en", "date": "", "evidence_ids": ["eb"]},
    }
    out = apply_routing([("a", 0.9), ("b", 0.8)], docs, "other", "", deprecated={"ea"})
    assert out["excluded"] == ["a"]
    assert out["backoff"] is True  # <3 kept -> backoff
    assert all(d not in ("a",) for d, _ in out["ranked"])
    out2 = apply_routing(
        [(f"d{i}", 0.9 - i * 0.01) for i in range(5)],
        {f"d{i}": {"intent": "other", "lang": "en", "date": "", "evidence_ids": [f"e{i}"]} for i in range(5)},
        "other", "", deprecated=set(),
    )
    assert out2["backoff"] is False


def test_load_deprecated_empty_default():
    from src.route import load_deprecated

    assert load_deprecated() == set()
