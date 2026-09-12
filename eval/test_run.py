from __future__ import annotations


def test_run_eval_smoke_limit2(tmp_path):
    from eval.run_eval import main

    out = tmp_path / "metrics.json"
    rc = main(["--limit", "2", "--out", str(out)])
    assert rc == 0
    import json

    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["n"] == 2 and d["live_calls"] == 0
    assert d["primary"]["metric"] == "intent_macro_f1"
    assert d["appendix"]["headline"] is False
    assert d["total_seconds"] < 20
