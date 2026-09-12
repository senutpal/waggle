from __future__ import annotations


def test_swap_agreement_and_tie():
    from eval.pairwise import adjudicate

    assert adjudicate("A", "B")["final"] == "X"  # X wins both orders
    assert adjudicate("B", "A")["final"] == "Y"
    m = adjudicate("A", "A")  # primacy both times -> mismatch
    assert m["final"] == "Tie" and m["needs_human"] is True
    assert adjudicate("Tie", "Tie")["final"] == "Tie"


def test_bad_kb_and_winner_validation():
    from eval.pairwise import parse_pairwise

    assert parse_pairwise({"winner": "A", "kb_id": "K1", "length_influenced": False,
                           "rationale": "content decides, kb K1"})["ok"] is True
    assert parse_pairwise({"winner": "C"})["ok"] is False


def test_consistency_gate():
    from eval.pairwise import consistency_stats

    rows = [{"agree": True, "winner_ab": "A", "winner_ba": "B"}] * 88
    rows += [{"agree": False, "winner_ab": "A", "winner_ba": "A"}] * 12
    s = consistency_stats(rows)
    assert s["consistency"] == 0.88 and s["gate_0_85"] is True
    s2 = consistency_stats([{"agree": False, "winner_ab": "A", "winner_ba": "A"}] * 10)
    assert s2["gate_0_85"] is False


def test_verbose_wrong_gate():
    from eval.pairwise import verbose_probe_gate

    assert verbose_probe_gate([{"probe_id": "x", "verbose_pass": False}])["pass"] is True
    assert verbose_probe_gate([{"probe_id": "x", "verbose_pass": True}])["pass"] is False


def test_lc_properties():
    from eval.pairwise import length_controlled_win_rate

    r = length_controlled_win_rate([1, 0, 1, 0] * 10, [0] * 40)
    assert r["lc"] == r["raw"]
    r2 = length_controlled_win_rate([1, 1, 1, 0, 0, 0] * 5, list(range(30)))
    assert 0 <= r2["lc"] <= 1 and r2["n"] == 30
