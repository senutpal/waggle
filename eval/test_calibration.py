from __future__ import annotations


def test_qwk_helpers_sanity():
    from eval.calibration import LIKERT_LABELS, qwk_1_5, safe_spearman

    assert LIKERT_LABELS == [1, 2, 3, 4, 5]
    assert qwk_1_5([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0
    # constant inputs -> defined, not nan
    assert qwk_1_5([5, 5, 5], [5, 5, 5]) == 1.0
    sp = safe_spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert abs(sp["rho"] - 1.0) < 1e-9
    const = safe_spearman([5, 5, 5], [5, 5, 4])
    assert const["rho"] != const["rho"]  # nan for constant


def test_qwk_ordinal_only():
    import json
    from pathlib import Path

    a = json.loads((Path(__file__).resolve().parents[1] / "results" / "agreement.json").read_text(encoding="utf-8"))
    # S2.2 intent path stays QWK-free
    assert "qw" not in json.dumps(a).lower().replace("follow", "")


def test_fast_qwk_matches_sklearn():
    from eval.calibration import _qwk_confusion, qwk_1_5

    a = [5, 5, 4, 4, 3, 5, 4, 2, 5, 4, 3, 3, 5, 1, 4]
    b = [5, 4, 4, 3, 3, 5, 4, 2, 4, 4, 3, 5, 5, 2, 4]
    assert abs(_qwk_confusion(a, b) - qwk_1_5(a, b)) < 1e-9


def test_dim_agreement_and_gate():
    from eval.calibration import dim_agreement, gate_ship

    j = [5, 5, 4, 4, 3, 5, 4, 5, 5, 4] * 5
    h1 = [5, 5, 4, 3, 3, 5, 4, 5, 4, 4] * 5
    h2 = [5, 4, 4, 4, 3, 5, 4, 5, 5, 4] * 5
    d = dim_agreement(j, h1, h2)
    assert d["n"] == 50 and "J_Hmean" in d and "H1_H2" in d
    g = gate_ship({"groundedness": d, "safety": d}, anchor_holds=True)
    assert "ship" in g and "fallback" in g


def test_triage_and_committed():
    from eval.calibration import select_committed, triage_rows

    rows = [
        {"query_id": "g0001", "stratum": "safety", "sys": "B2", "dim": "safety",
         "J": 5, "H1": 2, "H2": 2, "J_pass": True, "H_pass": False, "safety_mismatch": True},
        {"query_id": "g0002", "stratum": "easy", "sys": "B2", "dim": "tone",
         "J": 3, "H1": 3, "H2": 3, "J_pass": True, "H_pass": True},
    ]
    tri = triage_rows(rows)
    assert len(tri) == 1 and tri[0]["query_id"] == "g0001"
    assert len(select_committed(tri)) == 1
