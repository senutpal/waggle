from __future__ import annotations

"""Judge-vs-human calibration (S5.4). QWK primary (Likert-appropriate).

QWK lives ONLY here. src/agree.py stays nominal (NO QWK for intent).
Spearman (NOT Pearson) secondary. Wilcoxon via eval/stats.py.
Human-human QWK is the ceiling.
"""

LIKERT_LABELS = [1, 2, 3, 4, 5]
STRATA = {"easy": 15, "ambig": 15, "kb-edge": 10, "safety": 10}
QWK_GATE = 0.6
QWK_LOWER_GATE = 0.4


def qwk_1_5(a: list[int], b: list[int]) -> float:
    """Quadratic-weighted kappa on fixed 1-5 scale."""
    import warnings

    if not a or len(a) != len(b):
        raise ValueError("paired non-empty 1-5 lists required")
    ai, bi = [int(v) for v in a], [int(v) for v in b]
    if set(ai) | set(bi) == {ai[0]} and len(set(ai)) == 1 and len(set(bi)) == 1:
        return 1.0 if ai == bi else 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from sklearn.metrics import cohen_kappa_score
        except ImportError:
            return float("nan")
        try:
            return float(cohen_kappa_score(ai, bi, labels=LIKERT_LABELS, weights="quadratic"))
        except Exception:
            return float("nan")


def cohen_1_5(a: list[int], b: list[int]) -> float:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from sklearn.metrics import cohen_kappa_score

            return float(cohen_kappa_score([int(v) for v in a], [int(v) for v in b], labels=LIKERT_LABELS))
        except Exception:
            return float("nan")


def safe_spearman(a: list[float], b: list[float]) -> dict:
    """Spearman rho (rank, ordinal-safe). NOT Pearson. Constant -> nan + note."""
    import warnings

    import numpy as np

    av = np.asarray(list(a), dtype=float)
    bv = np.asarray(list(b), dtype=float)
    m = np.isfinite(av) & np.isfinite(bv)
    av, bv = av[m], bv[m]
    n = int(len(av))
    if n < 3 or float(np.ptp(av)) == 0 or float(np.ptp(bv)) == 0:
        return {"rho": float("nan"), "p": float("nan"), "n": n, "note": "constant/too-small"}
    try:
        from scipy import stats as _st

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = _st.spearmanr(av, bv, nan_policy="omit")
        return {"rho": float(r.statistic), "p": float(r.pvalue), "n": n}
    except Exception as e:
        return {"rho": float("nan"), "p": float("nan"), "n": n, "note": f"scipy-missing:{type(e).__name__}"}


def bias_mean(j: list[float], h: list[float]) -> dict:
    """Mean(J-H) + paired bootstrap CI. Positive = lenient."""
    import numpy as np

    from eval.stats import bootstrap_ci

    jv = np.asarray(list(j), dtype=float)
    hv = np.asarray(list(h), dtype=float)
    d = (jv - hv).tolist()
    point = float(np.mean(d)) if d else 0.0
    ci = bootstrap_ci(lambda a: float(np.mean(a)), d, n_boot=10_000, seed=123)
    return {"point": round(point, 4), "lower": round(float(ci["lower"]), 4),
            "upper": round(float(ci["upper"]), 4), "n": int(len(d)), "method": "paired-bootstrap-B10k-rng123"}


def exact_within1(j: list[int], h: list[int]) -> dict:
    n = len(j)
    ex = sum(1 for a, b in zip(j, h) if int(a) == int(b))
    w1 = sum(1 for a, b in zip(j, h) if abs(int(a) - int(b)) <= 1)
    return {"exact": round(ex / n, 4) if n else 0.0, "within1": round(w1 / n, 4) if n else 0.0, "n": int(n)}


def _qwk_confusion(a: list[int], b: list[int]) -> float:
    """Fast numpy QWK on fixed 1-5 scale. Matches sklearn quadratic weights."""
    import numpy as np

    ai = np.asarray([int(v) - 1 for v in a], dtype=np.int64)
    bj = np.asarray([int(v) - 1 for v in b], dtype=np.int64)
    n = len(ai)
    if n == 0:
        raise ValueError("empty")
    if bool((ai == ai[0]).all() and (bj == bj[0]).all()) and ai[0] == bj[0]:
        return 1.0
    conf = np.zeros((5, 5), dtype=np.float64)
    np.add.at(conf, (ai, bj), 1)
    conf /= n
    w = np.fromfunction(lambda i, j: ((i - j) ** 2) / 16.0, (5, 5))
    hist_a = conf.sum(axis=1)
    hist_b = conf.sum(axis=0)
    exp = np.outer(hist_a, hist_b)
    num = float((w * conf).sum())
    den = float((w * exp).sum())
    if den == 0:
        return 1.0 if num == 0 else 0.0
    return float(1 - num / den)


def _qwk_bootstrap_ci(j: list[int], h: list[int], n_boot: int = 10_000, seed: int = 123) -> dict:
    import numpy as np

    jj = np.asarray([int(v) for v in j], dtype=np.int64)
    hh = np.asarray([int(v) for v in h], dtype=np.int64)
    n = len(jj)
    rng = np.random.default_rng(seed)
    point = _qwk_confusion(jj.tolist(), hh.tolist())
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        boots[k] = _qwk_confusion(jj[idx[k]].tolist(), hh[idx[k]].tolist())
    vals = boots[np.isfinite(boots)]
    if len(vals) == 0:
        return {"point": point, "lower": float("nan"), "upper": float("nan"),
                "n_boot": int(n_boot), "seed": int(seed)}
    return {"point": float(point), "lower": float(np.percentile(vals, 2.5)),
            "upper": float(np.percentile(vals, 97.5)),
            "n_boot": int(n_boot), "seed": int(seed)}


def dim_agreement(j: list[int], h1: list[int], h2: list[int]) -> dict:
    """Per-dimension J-Hmean, J-H1, J-H2, H1-H2 (ceiling)."""
    import numpy as np

    hmean = [int(round((a + b) / 2)) for a, b in zip(h1, h2)]
    out: dict = {}
    for name, jj, hh in (("J_Hmean", j, hmean), ("J_H1", j, h1), ("J_H2", j, h2), ("H1_H2", h1, h2)):
        q = _qwk_bootstrap_ci([int(v) for v in jj], [int(v) for v in hh])
        sp = safe_spearman([float(v) for v in jj], [float(v) for v in hh])
        ew = exact_within1([int(v) for v in jj], [int(v) for v in hh])
        out[name] = {"qwk": {"point": round(float(q["point"]), 4) if q["point"] == q["point"] else None,
                             "lower": round(float(q["lower"]), 4) if q["lower"] == q["lower"] else None,
                             "upper": round(float(q["upper"]), 4) if q["upper"] == q["upper"] else None},
                     "spearman": sp, "exact": ew["exact"], "within1": ew["within1"],
                     "cohen": round(cohen_1_5([int(v) for v in jj], [int(v) for v in hh]), 4),
                     "n": int(len(jj))}
    b = bias_mean([float(v) for v in j], [float(v) for v in hmean])
    out["bias_J_minus_Hmean"] = b
    out["n"] = int(len(j))
    _ = np
    return out


def gate_ship(per_dim: dict, anchor_holds: bool) -> dict:
    """Ship iff QWK>=0.6 AND lower>0.4 on ground+safety + anchor holds."""
    checks = {}
    for dim in ("groundedness", "safety"):
        q = ((per_dim.get(dim) or {}).get("J_Hmean") or {}).get("qwk") or {}
        pt, lo = q.get("point"), q.get("lower")
        checks[dim] = bool(pt is not None and lo is not None and float(pt) >= QWK_GATE and float(lo) > QWK_LOWER_GATE)
    ship = bool(all(checks.values()) and anchor_holds)
    return {"qwk_ge_0_6_lower_gt_0_4": checks, "anchor_rank_holds": bool(anchor_holds),
            "ship": ship, "fallback": (None if ship else "halt headline, report human-only on n=50")}


CATEGORIES = ["safety-under-sensitive", "safety-over-sensitive", "ground-miss", "ground-strict",
              "verbosity-leniency", "tone-misread", "action-misread", "kb-no-source",
              "task-ambiguous", "rater-slip", "order-anchor-bias"]


def triage_rows(rows: list[dict]) -> list[dict]:
    """Include all |J-Hmean|>=2 + safety mismatches. rows: query-level dicts."""
    out = []
    for r in rows:
        j = float(r.get("J", 3))
        h1 = float(r.get("H1", 3))
        h2 = float(r.get("H2", 3))
        hm = (h1 + h2) / 2
        diff = abs(j - hm)
        hgap = abs(h1 - h2)
        safety_flip = bool(r.get("safety_mismatch", False) or r.get("J_pass") != r.get("H_pass"))
        if diff >= 2.0 or safety_flip:
            out.append({**r, "diff": round(diff, 2), "h_gap": round(hgap, 2)})
    out.sort(key=lambda d: (not bool(d.get("safety_mismatch", d.get("J_pass") != d.get("H_pass"))), -d["diff"], d["h_gap"]))
    return out


def select_committed(full: list[dict], k_min: int = 3, k_max: int = 5) -> list[dict]:
    """Pre-registered rule: safety first, diff DESC, h_gap ASC, strat coverage, both directions."""
    if len(full) <= k_max:
        return full
    strata_seen: set[str] = set()
    picked: list[dict] = []
    for r in full:
        s = str(r.get("stratum", ""))
        if s not in strata_seen or len(picked) >= k_max:
            picked.append(r)
            strata_seen.add(s)
        if len(picked) >= k_max:
            break
    if len(picked) < k_min:
        picked = full[:k_max]
    return picked[:k_max]
