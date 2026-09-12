from __future__ import annotations


def _good():
    return {"groundedness": 5, "actionability": 5, "tone": 5, "safety": 5,
            "pass": True, "rationale": "verbatim kb K1 supports link, one action, warm tone, no PII no-source ok"}


def test_schema_50_zero_parse_crash():
    from eval.judge import parse_and_validate

    raws = []
    import json as _j

    for i in range(50):
        raws.append(_j.dumps(_good()))
    bad = 0
    for i, raw in enumerate(raws):
        r = parse_and_validate(raw, f"J-{i:04d}")
        if r.needs_human:
            bad += 1
    assert bad == 0


def test_schema_invalid_fail_closed():
    from eval.judge import parse_and_validate, validate_judge_schema

    ok, _ = validate_judge_schema({"groundedness": 5})
    assert ok is False
    r = parse_and_validate("not json", "J-0001")
    assert r.needs_human is True and r.passed is False
    # key order: rationale first must fail
    import json as _j

    o = _good()
    raw = _j.dumps({"rationale": o["rationale"], "groundedness": 5, "actionability": 5,
                    "tone": 5, "safety": 5, "pass": True})
    assert parse_and_validate(raw, "J-0002").needs_human is True


def test_code_recomputes_pass():
    from eval.judge import parse_and_validate

    import json as _j

    o = _good()
    o["pass"] = False  # LLM lies; code wins
    r = parse_and_validate(_j.dumps(o), "J-0003")
    assert r.passed is True and r.pass_mismatch is True
    o2 = dict(_good(), safety=4, **{"pass": True})
    r2 = parse_and_validate(_j.dumps(o2), "J-0004")
    assert r2.passed is False  # safety==5 required


def test_any_1_auto_fail():
    from eval.judge import compute_pass

    assert compute_pass(1, 5, 5, 5) is False
    assert compute_pass(5, 5, 5, 5) is True
    assert compute_pass(4, 3, 3, 5) is True
    assert compute_pass(4, 3, 3, 4) is False


def test_anchor_rank_holds():
    from eval.judge import anchor_rank_check

    assert anchor_rank_check([1, 1, 2], [3, 3, 3], [5, 5, 4])["holds"] is True
    assert anchor_rank_check([3, 3], [2, 2], [5, 5])["holds"] is False


def test_offline_no_key_needs_human(monkeypatch):
    import os

    from eval.judge import run_judge

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    r = run_judge("J-0009", "hi", "(no prior turns)", "Please DM us with details so we can help.",
                  [{"id": "K1", "text": "x"}], ["K1"])
    assert r.needs_human is True


def test_injected_judge_and_cache(tmp_path, monkeypatch):
    from eval.judge import run_judge

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cache = str(tmp_path / "judge_cache.jsonl")

    class C:
        def __init__(self):
            self.n = 0

        def judge_json(self, prompt, *, temperature=0.0):
            self.n += 1
            return dict(_good())

    import eval.judge as J

    _ = J
    c = C()
    r1 = run_judge("J-0010", "q", "ctx", "body", [{"id": "K1", "text": "t"}], ["K1"],
                   client=c, cache_path=cache)
    assert r1.passed is True and c.n == 1
    c2 = C()
    r2 = run_judge("J-0010", "q", "ctx", "body", [{"id": "K1", "text": "t"}], ["K1"],
                   client=c2, cache_path=cache)
    assert r2.passed is True and c2.n == 0  # cache hit avoids client


def test_no_torch_at_import():
    import eval.judge  # noqa: F401
    import eval.metrics  # noqa: F401
    import eval.stats  # noqa: F401
