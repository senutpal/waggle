from __future__ import annotations


def test_primary_only_win():
    from eval.metrics import PRE_REGISTERED, decide_win

    assert PRE_REGISTERED["primary"] == "intent_macro_f1"
    assert decide_win({"delta": 0.10, "mcnemar_p": 0.01, "overlap_single_cis": True})["win"] is False
    assert decide_win({"delta": 0.02, "mcnemar_p": 0.01, "overlap_single_cis": False})["win"] is False
    assert decide_win({"delta": 0.10, "mcnemar_p": 0.01, "overlap_single_cis": False})["win"] is True
    # secondary delta never claims win via primary gate
    assert "appendix_only" in PRE_REGISTERED


def test_false_auto_formula():
    from eval.metrics import esc_false_auto

    r = esc_false_auto([1, 1, 1, 1, 1, 0, 0], ["auto", "auto", "escalate", "escalate", "escalate", "auto", "auto"])
    assert r["fp"] == 2 and r["tn"] == 3 and r["point"] == 0.4


def test_ece_matches_calibrate():
    from eval.metrics import esc_ece_10bin
    from src.calibrate import ece_score

    import numpy as np

    y = [0, 0, 1, 1]
    p = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    assert abs(esc_ece_10bin(y, p)["point"] - round(ece_score(y, p)["ece"], 4)) < 1e-9


def test_tierv_denominator():
    from eval.metrics import grounded_boilerplate_rate, grounded_verifiable

    gv = grounded_verifiable([True, True, False], [True, False, True])
    assert gv["ok"] == 1 and gv["n_tierv"] == 2 and gv["point"] == 0.5
    gb = grounded_boilerplate_rate([True, True, False])
    assert gb["headline"] is False


def test_rouge_appendix_never_headline():
    from eval.metrics import bertscore_appendix, rouge_appendix

    r = rouge_appendix(["hello world"], ["hello"])
    assert r["headline"] is False
    b = bertscore_appendix(["a"], ["a"], _inject=0.9)
    assert b["headline"] is False and b["bertscore_f1"] == 0.9


def test_tail_recall_split():
    from eval.metrics import tail_recall

    y = ["a", "a", "b", "b", "c", "c"]
    p = ["a", "b", "b", "b", "c", "a"]
    r = tail_recall(y, p, ["a", "b", "c"], ["b", "c"])
    assert r["tail"] == ["b", "c"] and 0 <= r["tail_macro"] <= 1
