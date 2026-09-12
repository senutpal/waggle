from __future__ import annotations

"""Small-n significance gates (S5.5). Single source of truth for stats.

Conventions (frozen, byte-compatible with scripts/eval_*.py):
- Wilson z=1.96 for single rates, rounded 4dp, n==0 -> [0,0].
- Paired bootstrap percentile B=10k rng=123 for F1/deltas; BCa opt-in.
- McNemar exact if b+c<25 else chi2 (erfc, no scipy), deterministic.
- Wilcoxon signed-rank (pratt, two-sided, asymptotic) for Likert 1-5.
- Report ``point [CI] p n method``; MDE 7pp paired / 14pp unpaired @n=200.
- Overlapping single CIs -> never bold / inconclusive.

No torch. numpy/sklearn/scipy lazily inside functions only.
"""

import math
import statistics

B_DEFAULT = 10_000
RNG_DEFAULT = 123
SEED_GLOBAL = 42
ALPHA = 0.05
Z95 = 1.96
CI95 = 0.95
MDE_PAIRED_PP = 7.0
MDE_UNPAIRED_PP = 14.0

_NORM = statistics.NormalDist()

PRIMARY: dict | None = None


def wilson(x: int, n: int, z: float = Z95) -> list[float]:
    """Wilson score interval. Verbatim-compatible with scripts/eval_*.py."""
    if n == 0:
        return [0.0, 0.0]
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / den
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def mcnemar(a, b) -> dict:
    """Paired binary test. a,b are paired 0/1 lists. Exact if b+c<25."""
    disc_b = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    disc_c = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n_d = disc_b + disc_c
    if n_d == 0:
        return {"b": 0, "c": 0, "p": 1.0, "method": "exact"}
    if n_d < 25:
        k = min(disc_b, disc_c)
        p = 2 * sum(math.comb(n_d, i) for i in range(k + 1)) / 2**n_d
        return {"b": disc_b, "c": disc_c, "p": round(min(1.0, p), 4), "method": "exact"}
    chi2 = (disc_b - disc_c) ** 2 / n_d
    p = float(math.erfc((chi2 / 2) ** 0.5))
    return {"b": disc_b, "c": disc_c, "stat": round(chi2, 3), "p": round(p, 4), "method": "chi2"}


def paired_delta_ci(a, b, seed: int = RNG_DEFAULT, n_boot: int = B_DEFAULT,
                    method: str = "percentile") -> tuple[float, list[float]]:
    """Paired delta CI. method=percentile is legacy-exact; bca opt-in."""
    import numpy as np

    rng = np.random.default_rng(seed)
    av = np.asarray(list(a), dtype=float)
    bv = np.asarray(list(b), dtype=float)
    diffs = bv - av
    point = float(diffs.mean()) if len(diffs) else 0.0
    if len(diffs) == 0:
        return point, [0.0, 0.0]
    boots = np.array([float(diffs[rng.integers(0, len(diffs), len(diffs))].mean()) for _ in range(n_boot)])
    if method == "percentile":
        ci = [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]
        return point, ci
    jack = np.array([(diffs[np.arange(len(diffs)) != k]).mean() for k in range(len(diffs))])
    ci = bca_interval(point, boots, jack)
    return round(point, 4), ci


def bca_interval(theta_hat: float, boot, jack, alpha: float = ALPHA) -> list[float]:
    """Efron BCa. Falls back to percentile on degenerate inputs."""
    import numpy as np

    boots = np.asarray(list(boot), dtype=float)
    if len(boots) == 0 or bool((boots == boots[0]).all()):
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        return [round(lo, 4), round(hi, 4)]
    b = len(boots)
    prop = float((boots < float(theta_hat)).mean())
    prop = min(max(prop, 1 / (b + 1)), b / (b + 1))
    z0 = _NORM.inv_cdf(prop)
    j = np.asarray(list(jack), dtype=float)
    jm = float(j.mean())
    num = float(((jm - j) ** 3).sum())
    den = float(((jm - j) ** 2).sum() ** 1.5)
    acc = num / (6 * den) if den > 0 else 0.0
    out: list[float] = []
    for q in (alpha / 2, 1 - alpha / 2):
        zq = _NORM.inv_cdf(q)
        denom = 1 - acc * (z0 + zq)
        if abs(denom) < 1e-12:
            return [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]
        adj = _NORM.cdf(z0 + (z0 + zq) / denom) * 100
        adj = min(max(adj, 100 / (b + 1)), 100 * b / (b + 1))
        out.append(round(float(np.percentile(boots, adj)), 4))
    return out


def bootstrap_ci(stat_fn, *samples, n_boot: int = B_DEFAULT, seed: int = RNG_DEFAULT,
                 ci: float = CI95, method: str = "percentile") -> dict:
    """Generic bootstrap. Resamples all samples jointly (paired-safe)."""
    import numpy as np

    n = len(samples[0])
    if any(len(s) != n for s in samples) or n == 0:
        raise ValueError("samples must share non-empty length")
    arrs = [np.asarray(list(s), dtype=object) for s in samples]
    rng = np.random.default_rng(seed)
    point = float(stat_fn(*[list(a) for a in arrs]))
    boots: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            boots.append(float(stat_fn(*[list(a[idx]) for a in arrs])))
        except Exception:
            boots.append(float("nan"))
    vals = np.array([v for v in boots if v == v])
    if len(vals) == 0:
        raise ValueError("bootstrap produced no finite values")
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    if method == "percentile":
        return {"point": point, "lower": float(np.percentile(vals, lo_q)),
                "upper": float(np.percentile(vals, hi_q)),
                "n_boot": int(n_boot), "seed": int(seed), "ci": float(ci),
                "method": "percentile", "n": int(n)}
    jack: list[float] = []
    for k in range(n):
        try:
            jack.append(float(stat_fn(*[[v for j, v in enumerate(list(a)) if j != k] for a in arrs])))
        except Exception:
            continue
    import numpy as _np

    ci95 = bca_interval(point, vals, _np.asarray(jack) if jack else vals[: max(1, len(vals) // 10)])
    return {"point": point, "lower": ci95[0], "upper": ci95[1],
            "n_boot": int(n_boot), "seed": int(seed), "ci": float(ci),
            "method": "bca", "n": int(n)}


def bootstrap_delta(a, b, stat_fn=None, n_boot: int = B_DEFAULT,
                    seed: int = RNG_DEFAULT, method: str = "percentile") -> dict:
    """Delta CI via joint resampling. Default stat is mean."""
    import numpy as np

    av = np.asarray(list(a), dtype=float)
    bv = np.asarray(list(b), dtype=float)
    if stat_fn is None:
        return {"point": float((bv - av).mean()), **_delta_ci((bv - av), n_boot, seed, method),
                "n": int(len(av))}
    diff = float(stat_fn(list(bv)) - stat_fn(list(av)))
    rng = np.random.default_rng(seed)
    n = len(av)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(float(stat_fn(list(bv[idx])) - stat_fn(list(av[idx]))))
    vals = np.asarray(boots)
    ci95 = [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]
    if method == "bca":
        jack = np.array([float(stat_fn(list(np.delete(bv, k))) - stat_fn(list(np.delete(av, k)))) for k in range(n)])
        ci95 = bca_interval(diff, vals, jack)
    return {"point": round(diff, 4), "lower": ci95[0], "upper": ci95[1],
            "n_boot": int(n_boot), "seed": int(seed), "method": method, "n": int(n)}


def _delta_ci(diffs, n_boot: int, seed: int, method: str) -> dict:
    import numpy as np

    d = np.asarray(list(diffs), dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.array([float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(n_boot)])
    if method == "percentile":
        return {"lower": round(float(np.percentile(boots, 2.5)), 4),
                "upper": round(float(np.percentile(boots, 97.5)), 4),
                "n_boot": int(n_boot), "seed": int(seed), "method": "percentile"}
    jack = np.array([(d[np.arange(len(d)) != k]).mean() for k in range(len(d))])
    ci95 = bca_interval(float(d.mean()), boots, jack)
    return {"lower": ci95[0], "upper": ci95[1], "n_boot": int(n_boot), "seed": int(seed), "method": "bca"}


def wilcoxon_signed_rank(x, y=None, zero_method: str = "pratt",
                         alternative: str = "two-sided", method: str = "asymptotic") -> dict:
    """Paired Likert Wilcoxon. Prefers scipy; pure-python fallback."""
    import numpy as np

    d = (np.asarray(list(x), dtype=float) - np.asarray(list(y), dtype=float)) if y is not None else np.asarray(list(x), dtype=float)
    d = d[np.isfinite(d)]
    d = np.round(d, 6)
    n = int(d.size)
    n_zero = int((d == 0).sum())
    n_eff = int((d != 0).sum())
    if n == 0 or n_zero == n:
        return {"stat": 0.0, "p": 1.0, "n": n, "n_zero": n_zero, "n_eff": 0,
                "zero_method": zero_method, "method": "none(all-zero)"}
    try:
        from scipy import stats as _st

        try:
            res = _st.wilcoxon(d, zero_method=zero_method, correction=False,
                               alternative=alternative, method=method)
        except Exception:
            res = _st.wilcoxon(d, zero_method=zero_method, correction=False,
                               alternative=alternative, method="approx")
        z = float(getattr(res, "zstatistic", float("nan")))
        import numpy as _np

        r = float(abs(z) / _np.sqrt(n_eff)) if _np.isfinite(z) and n_eff else 0.0
        return {"stat": float(res.statistic), "p": float(res.pvalue), "n": n,
                "n_zero": n_zero, "n_eff": n_eff if zero_method != "pratt" else n,
                "n_eff_nonzero": n_eff, "z": z, "r": r,
                "zero_method": zero_method, "method": f"wilcoxon-{method}"}
    except Exception:
        nz = d[d != 0]
        ars = _avg_ranks([abs(v) for v in nz])
        wpos = sum(r for v, r in zip(nz, ars) if v > 0)
        wneg = sum(r for v, r in zip(nz, ars) if v < 0)
        w = min(wpos, wneg)
        m = len(nz)
        mean = m * (m + 1) / 4
        var = m * (m + 1) * (2 * m + 1) / 24 - _tie_corr([abs(v) for v in nz]) / 48
        if var <= 0:
            return {"stat": float(w), "p": 1.0, "n": n, "n_zero": n_zero, "n_eff": m,
                    "zero_method": f"{zero_method}-fallback", "method": "wilcoxon-normal-fallback"}
        z = (w - mean) / math.sqrt(var)
        p = float(math.erfc(abs(z) / math.sqrt(2)) * (2 if alternative == "two-sided" else 1))
        return {"stat": float(w), "p": min(1.0, p), "n": n, "n_zero": n_zero, "n_eff": m,
                "zero_method": f"{zero_method}-fallback", "method": "wilcoxon-normal-fallback"}


def _avg_ranks(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _tie_corr(absvals: list[float]) -> float:
    from collections import Counter

    return sum(t**3 - t for t in Counter(absvals).values() if t > 1)


def holm(pvals: list[float], alpha: float = ALPHA) -> list[dict]:
    """Holm step-down FWER. Returns [{p_raw, p_adj, reject}]."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out: list[dict | None] = [None] * m
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * float(pvals[i]))
        running = max(running, adj)
        out[i] = {"p_raw": round(float(pvals[i]), 4), "p_adj": round(running, 4),
                  "reject": bool(running < alpha)}
    return [d for d in out if d is not None]  # type: ignore[misc]


def bh(pvals: list[float], q: float = 0.10) -> list[dict]:
    """Benjamini-Hochberg FDR for exploratory slices."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    out: list[dict | None] = [None] * m
    best = 1.0
    for pos in range(m - 1, -1, -1):
        i = order[pos]
        adj = min(1.0, (m / (pos + 1)) * float(pvals[i]))
        best = min(best, adj)
        out[i] = {"p_raw": round(float(pvals[i]), 4), "p_adj": round(best, 4),
                  "reject": bool(best < q)}
    return [d for d in out if d is not None]  # type: ignore[misc]


def format_point_ci(point: float, lo: float, hi: float, as_percent: bool = True) -> str:
    """Render 72.5% [66.0,78.2] for rates; raw for deltas."""
    if as_percent:
        return f"{point * 100:.1f}% [{lo * 100:.1f},{hi * 100:.1f}]"
    return f"{point:.3f} [{lo:.3f},{hi:.3f}]"


def summarize(point: float, ci: list[float], p: float | None, n: int, method: str) -> dict:
    """Canonical headline payload: point [CI] p n method."""
    return {"point": round(float(point), 4), "ci": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
            "p": (None if p is None else round(float(p), 4)), "n": int(n), "method": str(method),
            "render": format_point_ci(float(point), float(ci[0]), float(ci[1])) + (f" p={float(p):.3f}" if p is not None else "") + f" n={int(n)} {method}"}


def mde(n: int, paired: bool = True, p: float = 0.5) -> float:
    """Normal-approx MDE. ~0.07 paired / ~0.14 unpaired at n=200 worst-case."""
    import math as _m

    if paired:
        pd = 0.12
        return round(2.80 * _m.sqrt(pd / max(1, n)), 4)
    se = _m.sqrt(2 * p * (1 - p) / max(1, n))
    return round(2.80 * se, 4)


def verdict(delta: float, delta_ci: list[float], mcnemar_p: float | None,
            overlap_single_cis: bool, mde_pp: float = MDE_PAIRED_PP) -> str:
    """Win only if CIs disjoint AND delta excludes 0 AND p<0.05 AND |delta|>=MDE."""
    if overlap_single_cis:
        return "inconclusive (overlapping CIs)"
    lo, hi = float(delta_ci[0]), float(delta_ci[1])
    if lo <= 0 <= hi:
        return "no-win (delta CI includes 0)"
    if mcnemar_p is not None and float(mcnemar_p) >= 0.05:
        return "no-win (p>=0.05)"
    if abs(float(delta)) < float(mde_pp) / 100:
        return "inconclusive (<MDE)"
    return "win"


def preregister_primary(metric: str, arm: str = "b3_routed", n: int = 200) -> dict:
    """Freeze ONE primary. Second call warns (returns locked)."""
    global PRIMARY
    if PRIMARY is not None:
        return {"locked": True, **PRIMARY}
    PRIMARY = {"metric": str(metric), "arm": str(arm), "n": int(n)}
    return {"locked": False, **PRIMARY}


def check_win_claim(metric: str, v: str) -> bool:
    """True only if metric is the pre-registered primary and verdict is win."""
    if PRIMARY is None:
        return False
    return str(metric) == str(PRIMARY.get("metric")) and v == "win"
