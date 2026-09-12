from __future__ import annotations

"""Pairwise ablations with debias (S5.3). Pairwise ONLY; never for Table 1.

Dual-order A/B swap (mismatch->Tie/human), length instruction +
length-controlled win-rate, referenced KB (kb_id|no-source), anchors + prompt hash.
Gate consistency>=85% else halt. Pure functions + thin adjudication.
"""

from src.logging_utils import sha256_text

PAIRWISE_PROMPT_VERSION = "judge-pairwise-v1.0.0-gptoss20b-temp0-20260911"


def build_pairwise_prompt(qid: str, inbound: str, body_a: str, body_b: str,
                          sources: list[dict[str, str]], order: str = "AB") -> str:
    first, second = (body_a, body_b) if order == "AB" else (body_b, body_a)
    src_lines = "\n".join(f"[{s.get('id', f'K{i+1}')}] {s.get('text', '')}" for i, s in enumerate(sources or []))
    return (
        "You are JUDGE-PAIRWISE-v1.0.0, strict. Compare two replies to the SAME customer turn "
        "against PROVIDED SOURCES ONLY. Do not prefer longer/fluent/detailed. Prefer grounded + exactly one action. "
        "Verbosity without an additional verified fact is a defect.\n"
        f"QID: {qid} ORDER: {order}\nCUSTOMER: {inbound}\nSOURCES:\n{src_lines}\n"
        f"RESPONSE A:\n{first}\nRESPONSE B:\n{second}\n"
        "RULES: Tier-V claims need verbatim kb_id citation; Tier-B DM boilerplate may use no-source. "
        "Quote kb_id or no-source. Report len consideration separately; do not use length as quality.\n"
        'Output STRICT JSON ONLY IN ORDER: {"winner":"A|B|Tie","kb_id":str,"length_influenced":bool,"rationale":str}\n'
        "Scores concept: winner is content winner with lengths ignored. JSON ONLY."
    )


def parse_pairwise(obj: object) -> dict:
    if not isinstance(obj, dict):
        return {"ok": False, "error": "not-object", "winner": "Tie"}
    w = str(obj.get("winner", "Tie"))
    if w not in ("A", "B", "Tie"):
        return {"ok": False, "error": "bad-winner", "winner": "Tie"}
    kb = str(obj.get("kb_id", "no-source"))
    li = obj.get("length_influenced", False)
    rat = obj.get("rationale", "")
    if not isinstance(rat, str) or len(rat) < 5:
        return {"ok": False, "error": "bad-rationale", "winner": "Tie"}
    return {"ok": True, "winner": w, "kb_id": kb, "length_influenced": bool(li), "rationale": rat[:600]}


def _to_system(winner_pos: str, order: str, sys_a: str = "X", sys_b: str = "Y") -> str:
    """Map positional winner to system winner given order AB|BA where A=X,B=Y in AB."""
    if winner_pos == "Tie":
        return "Tie"
    if order == "AB":
        return sys_a if winner_pos == "A" else sys_b
    return sys_b if winner_pos == "A" else sys_a


def adjudicate(winner_ab: str, winner_ba: str, sys_a: str = "X", sys_b: str = "Y") -> dict:
    """Dual-order adjudication. Mismatch -> Tie/human. Pure."""
    ca = _to_system(winner_ab, "AB", sys_a, sys_b)
    cb = _to_system(winner_ba, "BA", sys_a, sys_b)
    agree = bool(ca == cb)
    final = ca if agree else "Tie"
    return {"system_ab": ca, "system_ba": cb, "agree": agree, "final": final,
            "needs_human": not agree}


def win_rate(finals: list[str], sys: str = "X") -> dict:
    """Win-rate with ties as half-wins. Wilson via eval.stats at call site."""
    n = len(finals)
    w = sum(1 for f in finals if f == sys)
    t = sum(1 for f in finals if f == "Tie")
    point = (w + 0.5 * t) / n if n else 0.0
    return {"point": round(point, 4), "wins": int(w), "ties": int(t),
            "losses": int(n - w - t), "n": int(n)}


def length_controlled_win_rate(wins: list[int], delta_len: list[float]) -> dict:
    """Minimal LC (no gamma_x): logit = a + phi*tanh(d/std). LC = logistic(a).

    wins: 1=m wins, 0=b wins (drop ties or encode 0.5 upstream and filter here).
    delta_len: len(m)-len(b) chars. Returns raw + LC + phi.
    """
    import math

    import numpy as np

    y = np.asarray([float(v) for v in wins], dtype=float)
    d = np.asarray([float(v) for v in delta_len], dtype=float)
    mask = (y == 0.0) | (y == 1.0)
    y, d = y[mask], d[mask]
    n = int(len(y))
    raw = float(y.mean()) if n else 0.5
    if n < 10 or float(np.std(d)) == 0.0:
        return {"raw": round(raw, 4), "lc": round(raw, 4), "phi": 0.0, "n": n,
                "note": "length-effect unidentifiable"}
    t = np.tanh(d / float(np.std(d)))
    try:
        from sklearn.linear_model import LogisticRegression

        clf = LogisticRegression(C=1.0, solver="lbfgs")
        clf.fit(t.reshape(-1, 1), (y > 0.5).astype(int))
        a = float(clf.intercept_[0])
        phi = float(clf.coef_[0][0])
        lc = 1 / (1 + math.exp(-a))
        return {"raw": round(raw, 4), "lc": round(lc, 4), "phi": round(phi, 4), "n": n}
    except Exception:
        return {"raw": round(raw, 4), "lc": round(raw, 4), "phi": 0.0, "n": n,
                "note": "lr-fallback"}


def consistency_stats(rows: list[dict]) -> dict:
    """rows: [{agree:bool, winner_ab, winner_ba}]. Returns consistency + bias_first."""
    n = len(rows)
    nc = sum(1 for r in rows if r.get("agree"))
    cons = nc / n if n else 0.0
    first_picks = sum(1 for r in rows for k in ("winner_ab", "winner_ba") if r.get(k) == "A")
    calls = 2 * n
    bias = first_picks / calls if calls else 0.5
    return {"n": int(n), "n_consistent": int(nc), "consistency": round(cons, 4),
            "n_first": int(first_picks), "n_calls": int(calls),
            "bias_first": round(bias, 4), "gate_0_85": bool(cons >= 0.85)}


def prompt_hash(prompt: str) -> str:
    return sha256_text(prompt)[:16]


VERBOSE_PROBES: list[dict] = [
    {"id": "filler-expansion", "expect": "concise-or-tie"},
    {"id": "hallucinated-link", "expect": "concise"},
    {"id": "wrong-order-id", "expect": "concise"},
    {"id": "truncation-control", "expect": "long-complete"},
    {"id": "tone-trap", "expect": "concise-correct"},
    {"id": "length-matched", "expect": "correct"},
]


def verbose_probe_gate(results: list[dict]) -> dict:
    """results: [{probe_id, verbose_pass:bool}]. Verbose-wrong must fail."""
    fails = [r for r in results if r.get("verbose_pass")]
    return {"n": int(len(results)), "verbose_pass_n": int(len(fails)),
            "pass": bool(len(fails) == 0),
            "gate": "verbose-wrong still fails"}
