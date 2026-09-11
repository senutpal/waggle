from __future__ import annotations


def test_redact_order_card_first():
    from src.pii import redact

    assert redact("my card 4111 1111 1111 1111 expired") == "my card [REDACTED_CARD] expired"
    assert redact("card 4111-1111-1111-1111") == "card [REDACTED_CARD]"
    assert redact("mail j.doe+tag@example.com please") == "mail [REDACTED_EMAIL] please"
    assert redact("call 415-555-0147") == "call [REDACTED_PHONE]"
    assert redact("order 123-4567890-1234567 missing") == "order [REDACTED_ORDER] missing"
    assert "[REDACTED_HANDLE]" in redact("hi @bob help")
    combo = redact("card 4111 1111 1111 1111 + a@b.com + 415-555-0147 + 123-4567890-1234567 + @bob")
    for tok in ("[REDACTED_CARD]", "[REDACTED_EMAIL]", "[REDACTED_PHONE]", "[REDACTED_ORDER]", "[REDACTED_HANDLE]"):
        assert tok in combo


def test_post_sweep_catches_reintroduced():
    from src.pii import post_sweep

    swept, leaked = post_sweep("contact jane@example.com now")
    assert leaked is True
    assert "jane@example.com" not in swept and "[REDACTED_EMAIL]" in swept
    swept2, leaked2 = post_sweep("hello world")
    assert leaked2 is False


def test_banned_phrases_hedged():
    from src.pii import HEDGE_TEMPLATE, hedge

    for banned in (
        "You are guaranteed a refund for this ticket.",
        "we will refund you today",
        "refund approved, done",
        "You get a 100% refund, no questions asked.",
        "You'll get an instant refund to your card.",
    ):
        out, changed, _ = hedge(banned)
        assert changed is True
        assert "guaranteed refund" not in out.lower() or HEDGE_TEMPLATE in out


def test_paraphrase_semantic_hedge():
    from src.pii import HEDGE_TEMPLATE, hedge

    out, changed, reason = hedge("you'll get money back tomorrow", _semantic=0.88)
    assert changed is True and out == HEDGE_TEMPLATE and reason.startswith("SEMANTIC")
    out2, changed2, _ = hedge("Sorry \u2014 please DM @Support; refund eligibility varies by policy.", _semantic=0.4)
    assert changed2 is False


def test_fit_280_word_boundary_and_postfit():
    from src.pii import extract_links, fit, post_fit_validate

    long_no_link = "word " * 120
    r = fit(long_no_link.strip())
    assert r.escalate is False and len(r.text) <= 280
    assert fit(fit(long_no_link.strip()).text).text == fit(long_no_link.strip()).text
    draft = " ".join(["word"] * 100) + " see https://example.com/x"
    r2 = fit(draft)
    # link at tail is severed by truncation -> must escalate, never silent drop
    assert r2.escalate is True
    ok, _ = post_fit_validate("see https://example.com/x", "see https://example.com/x", [])
    assert ok is True
    ok2, _ = post_fit_validate("see https://example.com/x", "see shortened", [])
    assert ok2 is False


def test_tone_mirror_and_anger_gate():
    from src.pii import is_angry, mirror_tone

    assert mirror_tone("casual").startswith("Thanks")
    assert mirror_tone("formal").startswith("Thank you")
    assert mirror_tone("angry", "the delayed order").startswith("You're right to be upset")
    assert mirror_tone("angry", "x", anger_conf=0.3) == ""
    hit, conf = is_angry("WHERE IS MY PRODUCT??? STILL WAITING, USELESS!!!")
    assert hit is True and conf >= 0.7
