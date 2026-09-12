from __future__ import annotations


def test_run_eval_smoke_limit2(tmp_path):
    from pathlib import Path as _P

    from eval.run_eval import main

    # main() always writes manifests/s5_1_metrics.json (repo path); snapshot + restore
    # so the smoke run never pollutes the committed n=200 artifact.
    man = _P(__file__).resolve().parents[1] / "manifests" / "s5_1_metrics.json"
    snap = man.read_bytes() if man.exists() else None
    try:
        out = tmp_path / "metrics.json"
        rc = main(["--limit", "2", "--out", str(out)])
        assert rc == 0
        import json

        d = json.loads(out.read_text(encoding="utf-8"))
        assert d["n"] == 2 and d["live_calls"] == 0
        assert d["primary"]["metric"] == "intent_macro_f1"
        assert d["appendix"]["headline"] is False
        assert d["total_seconds"] < 20
    finally:
        if snap is not None:
            man.write_bytes(snap)
