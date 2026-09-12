from __future__ import annotations

"""Automated metric suite (S5.1). Pure point estimates; CIs live in eval/stats.py.

PRE-REGISTERED PRIMARY: intent macro-F1. Only it can claim ``win``.
Secondaries are descriptive. Appendix (ROUGE/BERTScore/boilerplate) never headlines.
"""

PRE_REGISTERED = {
    "primary": "intent_macro_f1",
    "secondaries": [
        "tail_recall",
        "judge_pass_rate",
        "esc_false_auto",
        "esc_auroc",
        "esc_ece_10bin",
        "grounded_verifiable_TierV",
        "recall_at_3",
        "safety_violation_rate",
        "p95_ms",
        "cost_per_1k",
    ],
    "appendix_only": ["rouge1_f1", "rougeL_f1", "bertscore_f1", "grounded_boilerplate_rate"],
}

METRIC_REGISTRY = {**{k: {"type": "primary" if k == PRE_REGISTERED["primary"] else "secondary",
                           "headline": k == PRE_REGISTERED["primary"]}
                      for k in [PRE_REGISTERED["primary"], *PRE_REGISTERED["secondaries"]]},
                   **{k: {"type": "appendix", "headline": False} for k in PRE_REGISTERED["appendix_only"]}}


def intent_macro_f1(y_true: list[str], y_pred: list[str], labels: list[str] | None = None) -> dict:
    from sklearn.metrics import f1_score

    labs = list(labels or sorted(set(y_true) | set(y_pred)))
    point = float(f1_score(y_true, y_pred, labels=labs, average="macro", zero_division=0))
    return {"point": round(point, 4), "n": int(len(y_true)), "method": "sklearn-f1-macro"}


def per_intent_table(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[dict]:
    from sklearn.metrics import precision_recall_fscore_support

    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return [{"intent": lab, "precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
             "f1": round(float(f[i]), 4), "support": int(s[i])} for i, lab in enumerate(labels)]


def tail_recall(y_true: list[str], y_pred: list[str], labels: list[str], tail_labels: list[str]) -> dict:
    from sklearn.metrics import precision_recall_fscore_support

    _, r, _, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    idx = {lab: i for i, lab in enumerate(labels)}
    rs = [float(r[idx[t]]) for t in tail_labels if t in idx]
    tp = sum(int(s[idx[t]]) * float(r[idx[t]]) for t in tail_labels if t in idx)
    den = sum(int(s[idx[t]]) for t in tail_labels if t in idx)
    return {"tail_macro": round(sum(rs) / len(rs), 4) if rs else 0.0,
            "tail_micro": round(tp / den, 4) if den else 0.0,
            "tail": list(tail_labels), "n": int(len(y_true))}


def esc_false_auto(y_should: list[int], decisions: list[str]) -> dict:
    """False-auto = FP/(FP+TN) where should=1 is must-escalate, pred auto=answered."""
    fp = sum(1 for y, d in zip(y_should, decisions) if y == 1 and d != "escalate")
    tn = sum(1 for y, d in zip(y_should, decisions) if y == 1 and d == "escalate")
    den = fp + tn
    return {"point": (round(fp / den, 4) if den else None), "fp": int(fp), "tn": int(tn),
            "n_pos": int(den), "n": int(len(y_should)), "formula": "FP/(FP+TN)"}


def esc_auroc(y_should: list[int], risks: list[float]) -> dict:
    import numpy as np

    y = np.asarray(list(y_should), dtype=int)
    s = np.asarray(list(risks), dtype=float)
    if len(set(y.tolist())) < 2:
        return {"point": None, "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
                "n": int(len(y)), "reason": "single-class"}
    from sklearn.metrics import roc_auc_score

    return {"point": round(float(roc_auc_score(y, s)), 4), "n_pos": int(y.sum()),
            "n_neg": int((1 - y).sum()), "n": int(len(y)), "method": "roc_auc_score"}


def esc_ece_10bin(y_true: list[int], proba) -> dict:
    """Delegate to src.calibrate.ece_score (equal-width 10-bin)."""
    from src.calibrate import ece_score

    res = ece_score(list(y_true), proba, n_bins=10)
    return {"point": round(float(res["ece"]), 4), "n_bins": 10, "n": int(res["n"]),
            "bins": res["bins"], "method": "ece-10bin"}


def grounded_verifiable(tierv_flags: list[bool], faithful_ok: list[bool]) -> dict:
    n = sum(1 for t in tierv_flags if t)
    ok = sum(1 for t, f in zip(tierv_flags, faithful_ok) if t and f)
    return {"point": (round(ok / n, 4) if n else None), "ok": int(ok), "n_tierv": int(n),
            "n": int(len(tierv_flags)), "note": "Tier-V only; Tier-B exempt"}


def grounded_boilerplate_rate(tierv_flags: list[bool]) -> dict:
    n = len(tierv_flags)
    tb = sum(1 for t in tierv_flags if not t)
    return {"point": round(tb / n, 4) if n else 0.0, "n_boilerplate": int(tb), "n": int(n),
            "headline": False}


def recall_at_3(hits: list[int]) -> dict:
    n = len(hits)
    return {"point": round(sum(1 for h in hits if h) / n, 4) if n else 0.0,
            "k": int(sum(1 for h in hits if h)), "n": int(n)}


def safety_violation(uncited_auto: list[int], pii_leak: list[int], banned: list[int],
                     gate_bypass: list[int]) -> dict:
    n = len(uncited_auto)
    bad = [1 if (u or p or b or g) else 0 for u, p, b, g in zip(uncited_auto, pii_leak, banned, gate_bypass)]
    return {"point": round(sum(bad) / n, 4) if n else 0.0, "k": int(sum(bad)), "n": int(n),
            "by": {"uncited_TierV_auto": int(sum(uncited_auto)), "pii": int(sum(pii_leak)),
                   "banned": int(sum(banned)), "gate_bypass": int(sum(gate_bypass))}}


def p95_cost(lat_ms: list[float], tok_in: list[int] | None = None,
             tok_out: list[int] | None = None, price_in: float = 0.0,
             price_out: float = 0.0) -> dict:
    """p95 latency + $/1k. Free-tier $0 with method disclosed."""
    import numpy as np

    lat = [float(v) for v in (lat_ms or []) if v == v]
    if not lat:
        return {"p50": None, "p95": None, "n": 0, "cost_per_1k": 0.0, "method": "perf_counter"}
    p50, p95 = float(np.percentile(lat, 50)), float(np.percentile(lat, 95))
    n = len(lat)
    ci = float(tok_in is not None)
    _ = ci
    tin = sum(tok_in or [])
    tou = sum(tok_out or [])
    cost = tin / 1e6 * float(price_in) + tou / 1e6 * float(price_out)
    per1k = round(cost / n * 1000, 6) if n else 0.0
    return {"p50": round(p50, 1), "p95": round(p95, 1), "n": int(n),
            "cost_per_1k": per1k, "tokens_in": int(tin), "tokens_out": int(tou),
            "method": "percentile-free-tier-0" if price_in == 0 and price_out == 0 else "percentile-metered"}


def rouge_appendix(refs: list[str], cands: list[str]) -> dict:
    """Appendix-only ROUGE-1/L F1. Pure-python rouge_score, lazy."""
    try:
        from src import _lazy

        _lazy.require_extra("rouge_score", "pip install rouge-score (appendix only)")
        from rouge_score.rouge_scorer import RougeScorer

        sc = RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        r1, rl = [], []
        for ref, cand in zip(refs, cands):
            s = sc.score(ref or "", cand or "")
            r1.append(float(s["rouge1"].fmeasure))
            rl.append(float(s["rougeL"].fmeasure))
        import numpy as np

        return {"rouge1_f1": round(float(np.mean(r1)), 4) if r1 else 0.0,
                "rougeL_f1": round(float(np.mean(rl)), 4) if rl else 0.0,
                "n": int(len(r1)), "headline": False, "status": "ok"}
    except Exception as e:
        return {"rouge1_f1": None, "rougeL_f1": None, "n": int(len(refs)),
                "headline": False, "status": f"skipped:{type(e).__name__}"}


def bertscore_appendix(refs: list[str], cands: list[str], _inject: float | None = None) -> dict:
    """Appendix-only BERTScore-F1. Full-only, lazy, skip-safe."""
    if _inject is not None:
        return {"bertscore_f1": round(float(_inject), 4), "n": int(len(refs)),
                "headline": False, "status": "injected"}
    try:
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from src import _lazy

        _lazy.require_extra("bert_score", "full-only appendix")
        _lazy.require_extra("torch", "full-only appendix")
        from bert_score import score as _bscore

        import torch

        torch.set_num_threads(1)
        _, _, f1 = _bscore(list(cands), list(refs), lang="en", device="cpu",
                           batch_size=4, rescale_with_baseline=True, verbose=False)
        vals = [float(v) for v in f1.tolist()]
        import numpy as np

        return {"bertscore_f1": round(float(np.mean(vals)), 4) if vals else 0.0,
                "n": int(len(vals)), "headline": False, "status": "ok"}
    except Exception as e:
        return {"bertscore_f1": None, "n": int(len(refs)),
                "headline": False, "status": f"skipped_no_torch:{type(e).__name__}"}


def decide_win(primary: dict, secondaries: dict | None = None) -> dict:
    """Only PRIMARY can claim win. Secondaries always descriptive."""
    _ = secondaries
    delta = float(primary.get("delta", 0.0))
    p = primary.get("mcnemar_p", primary.get("p"))
    overlap = bool(primary.get("overlap_single_cis", True))
    if overlap:
        return {"claim": "inconclusive", "reason": "overlapping CIs never bolded", "win": False}
    if p is not None and float(p) >= 0.05:
        return {"claim": "no-win", "reason": "p>=0.05", "win": False}
    if abs(delta) < 0.07:
        return {"claim": "inconclusive", "reason": "MDE~7pp paired at n=200; +2pp never win", "win": False}
    if delta <= 0:
        return {"claim": "no-win", "reason": "delta<=0", "win": False}
    return {"claim": "win", "reason": "primary-only", "win": True}
