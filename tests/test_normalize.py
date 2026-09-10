from __future__ import annotations

import pandas as pd

from src.normalize import mask, normalize_text, scan_pii, strip_signature, unescape


def test_order_unescape_mask_signature():
    assert unescape("a &amp; b") == "a & b"
    assert mask("hi @bob see http://t.co/x") == "hi @USER see <URL>"
    assert mask("mail me a@b.com") == "mail me __email__"
    assert strip_signature("thanks ^JD") == "thanks"
    assert strip_signature("a^b mid") == "a^b mid"
    assert normalize_text("hi @bob http://t.co/x ^AM") == "hi @USER <URL>"


def test_emoji_preserved():
    assert "😡" in normalize_text("angry 😡 help @bob")


def test_placeholders_preserved_as_tokens():
    t = normalize_text("mail a@b.com order 123-4567890-1234567 see http://t.co/x @bob")
    assert "__email__" in t
    assert "ORD-12345" in t
    assert "<URL>" in t
    assert "@USER" in t


def test_exact_dedup_and_pii_zero():
    from src.normalize import exact_dedup

    df = pd.DataFrame({"text_norm": ["hello world", "hello  world", "other"]})
    out, info = exact_dedup(df)
    assert info["removed"] == 1
    assert len(out) == 2
    assert scan_pii(["call __phone__ now", "mail __email__"]) == {"email_hits": 0, "phone_hits": 0}
    hit = scan_pii(["call 415-555-1234 now"])
    assert hit["phone_hits"] == 1
