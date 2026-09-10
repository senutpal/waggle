from __future__ import annotations

"""Inter-annotator agreement for nominal intent labels.

Cohen kappa + per-class kappa + Krippendorff's alpha (nominal) with
percentile bootstrap 95% CIs. NO quadratic-weighted kappa here: QWK is
ordinal and reserved for the S5.4 Likert judge calibration.
"""


def cohen_kappa(y1: list[str], y2: list[str], labels: list[str] | None = None) -> float:
    from sklearn.metrics import cohen_kappa_score

    if len(y1) != len(y2) or not y1:
        raise ValueError("raters must have equal non-empty length")
    return float(cohen_kappa_score(y1, y2, labels=labels))


def per_class_kappa(y1: list[str], y2: list[str], labels: list[str]) -> dict[str, float]:
    """One-vs-rest Cohen kappa per class (binary collapse per label)."""
    out: dict[str, float] = {}
    for lab in labels:
        b1 = ["1" if v == lab else "0" for v in y1]
        b2 = ["1" if v == lab else "0" for v in y2]
        # degenerate (nobody used the label): define as 1.0 if perfect
        # agreement on absence, else 0.0
        if all(v == "0" for v in b1) and all(v == "0" for v in b2):
            out[lab] = 1.0
            continue
        out[lab] = float(__import__("sklearn.metrics", fromlist=["cohen_kappa_score"]).cohen_kappa_score(b1, b2))
    return out


def krippendorff_alpha_nominal(y1: list[str], y2: list[str]) -> float:
    """Krippendorff's alpha for 2 raters, nominal metric, no missing data.

    Implemented from first principles (no extra dependency). For this
    complete 2-rater nominal case it equals Cohen's kappa up to the
    chance-correction convention; we report both for transparency.
    """
    import numpy as np

    if len(y1) != len(y2) or not y1:
        raise ValueError("raters must have equal non-empty length")
    n = len(y1)
    labels = sorted(set(y1) | set(y2))
    # observed disagreement: fraction of pairs disagreeing
    do = sum(1 for a, b in zip(y1, y2) if a != b) / n
    # expected disagreement from pooled marginals
    counts: dict[str, int] = {lab: 0 for lab in labels}
    for v in list(y1) + list(y2):
        counts[v] += 1
    total = 2 * n
    de = 1.0 - sum((c / total) ** 2 for c in counts.values())
    if de == 0:
        return 1.0 if do == 0 else 0.0
    return float(1.0 - do / de)


def bootstrap_ci(
    fn, y1: list[str], y2: list[str], n_boot: int = 10_000, seed: int = 42, ci: float = 0.95
) -> dict[str, float]:
    """Percentile bootstrap CI for an agreement function fn(y1, y2)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(y1)
    a = np.array(y1, dtype=object)
    b = np.array(y2, dtype=object)
    stats: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            stats.append(float(fn(list(a[idx]), list(b[idx]))))
        except Exception:
            stats.append(float("nan"))
    arr = np.array([s for s in stats if s == s])
    if len(arr) == 0:
        raise ValueError("bootstrap produced no finite values")
    lo_q = (1 - ci) / 2 * 100
    hi_q = (1 + ci) / 2 * 100
    return {
        "point": float(fn(list(y1), list(y2))),
        "lower": float(np.percentile(arr, lo_q)),
        "upper": float(np.percentile(arr, hi_q)),
        "n_boot": int(n_boot),
        "ci": float(ci),
    }
