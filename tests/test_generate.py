from __future__ import annotations


def test_prompt_has_all_vars_and_rules():
    from src.generate import PromptContext, build_prompt

    ctx = PromptContext(
        tone="warm", lang="en", intent="shipping-tracking",
        inbound_clean="where is my package",
        thread_context="Customer: where is my package",
        evidence=[{"id": "E1", "text": "track in app"}, {"id": "E2", "text": "dm us"}, {"id": "E3", "text": ""}],
    )
    p = build_prompt(ctx)
    for var in ("warm", "en", "shipping-tracking", "where is my package", "track in app"):
        assert var in p
    assert "240" in p and "JSON ONLY" in p and "[D#]" in p


def test_body_240_and_no_leak():
    from src.generate import validate_output

    ok = validate_output({"body": "Hi! Please DM us with order details.", "evidence_ids": []}, set())
    assert ok.ok is True
    over = validate_output({"body": "x" * 241, "evidence_ids": []}, set())
    assert over.ok is False and over.reason == "OVER_240"
    leak = validate_output({"body": "see [D1] for help", "evidence_ids": ["D1"]}, {"D1"})
    assert leak.ok is False and leak.reason == "CITATION_LEAK"


def test_bad_citation_subset():
    from src.generate import validate_output

    r = validate_output({"body": "See link.", "evidence_ids": ["D9"]}, {"D1", "D2"})
    assert r.ok is False and r.reason == "BAD_CITATION"


def test_single_action_rule():
    from src.generate import check_single_action

    assert check_single_action("Sorry! Please DM order# + email so we can track it.")[0] is True
    ok, n, reason = check_single_action("Sorry! DM order#, check tracking link, and try restarting app?")
    assert ok is False and reason == "MULTI_ACTION"


def test_weighted_len_urls_emoji():
    from src.generate import weighted_len

    assert weighted_len("Hello") == 5
    assert weighted_len("Hi https://example.com/a/very/long/url/that/goes/on") == 26
    assert weighted_len("\U0001F602") == 2
    assert weighted_len("a" * 280) == 280
    assert weighted_len("a" * 281) == 281


def test_offline_fallback_template():
    from src.generate import MISSING_ANSWER, PromptContext, generate_draft

    ctx = PromptContext("warm", "en", "other", "hi", "(no prior turns)", [])
    out = generate_draft(ctx, client=None)
    # no GEMINI_API_KEY in test env -> offline fallback
    assert out["body"] == MISSING_ANSWER
    assert out["evidence_ids"] == []


def test_no_torch_at_import():
    import src.generate  # noqa: F401
