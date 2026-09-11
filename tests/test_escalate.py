from __future__ import annotations


def test_hard_gates_all_escalate():
    from src.escalate import check_hard_gates

    cases = [
        ("get me a human now", "HUM"),
        ("i will sue you, talking to my lawyer", "LEG"),
        ("filing chargeback + reporting to regulator FTC", "LEG"),
        ("help me bypass the 2FA to get into this account", "PRI"),
        ("i want to hurt myself over this order", "SAF"),
        ("i will come to warehouse and hurt staff", "SAF"),
        ("refund $750 for cancelled order", "FIN"),
        ("close my account and delete everything now", "FIN"),
    ]
    for text, code in cases:
        hit, got, _ = check_hard_gates(text)
        assert hit is True and got == code, text
    hit, _, _ = check_hard_gates("where is my package? order ORD-12345")
    assert hit is False


def test_risk_math_and_pen_cap():
    from src.escalate import compute_risk

    r, parts = compute_risk(0.95, 0.9, ret_score=0.9)
    assert abs(r - 0.0785) < 0.01
    r2, _ = compute_risk(0.80, 0.5, ret_score=0.9)
    assert r2 < 0.35
    r3, parts3 = compute_risk(0.70, 0.5, ret_score=0.2)
    assert parts3["pen"] == 0.2 and r3 > 0.60
    _, cap = compute_risk(0.9, 0.9, pii_present=True, hard_hits=5, ret_score=0.1)
    assert cap["pen"] == 0.5


def test_low_conf_overrides_valid_shape():
    from src.escalate import decide

    out = decide({"text": "where is my package", "conf": 0.70, "ground": 1.0, "ret_score": 0.9})
    assert out["decision"] == "escalate" and out["reason"] == "LOW_CONFIDENCE"


def test_schema_invalid_fail_closed():
    from src.escalate import decide, validate_schema

    ok, _ = validate_schema({"decision": "auto", "confidence": 0.9})
    assert ok is False
    ok2, _ = validate_schema(
        {"decision": "auto", "confidence": 0.9, "reason": "x", "risk_factors": [], "evidence_spans": [], "handoff_summary": "y", "extra": 1}
    )
    assert ok2 is False
    # decide never raises on garbage
    out = decide({"text": "hi", "conf": 2.0, "ground": 5.0})
    assert out["decision"] == "escalate" and out["reason"] == "INVALID_JSON"


def test_tier_bands_med():
    from src.escalate import decide

    auto = decide({"text": "track my order", "conf": 0.95, "ground": 0.9, "ret_score": 0.9}, tier="med")
    assert auto["decision"] == "auto"
    mid = decide({"text": "track my order", "conf": 0.85, "ground": 0.2, "ret_score": 0.9}, tier="med")
    assert mid["decision"] in ("hedge_clarify", "escalate")
    hi = decide({"text": "track my order", "conf": 0.80, "ground": 0.0, "ret_score": 0.2, "pii_present": True}, tier="med")
    assert hi["decision"] == "escalate"
