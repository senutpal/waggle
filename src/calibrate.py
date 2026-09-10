from __future__ import annotations

"""Calibration + OOD utilities for the intent heads.

- Temperature scaling on validation NLL (Guo et al. 2017), grid
  ``T in 0.1..3.0`` with 3-fold CV. Accuracy-preserving by construction.
- ECE (10 equal-width bins) + NLL delta + reliability plot data.
- Energy ``E = -T * logsumexp(logits / T)`` for OOS detection ONLY
  (Liu et al. 2020). No per-intent Mahalanobis: singular at ~20/class
  in 384-d, so it is deliberately absent.
"""

TEMP_GRID: list[float] = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]


def softmax_temp(logits, temp: float):
    import numpy as np

    z = np.asarray(logits, dtype=np.float64) / float(temp)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def nll_score(y_true: list[int], proba) -> float:
    import numpy as np

    p = np.asarray(proba, dtype=np.float64)
    eps = 1e-12
    p = p / p.sum(axis=1, keepdims=True).clip(min=eps)
    idx = np.arange(len(y_true))
    return float(-np.log(p[idx, np.asarray(y_true)].clip(min=eps)).mean())


def ece_score(y_true: list[int], proba, n_bins: int = 10) -> dict:
    """Equal-width ECE over confidence of the predicted class."""
    import numpy as np

    p = np.asarray(proba, dtype=np.float64)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    y = np.asarray(y_true)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins: list[dict] = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        n_b = int(mask.sum())
        if n_b == 0:
            bins.append({"bin": b, "lo": float(lo), "hi": float(hi), "n": 0, "acc": 0.0, "conf": 0.0})
            continue
        acc = float((pred[mask] == y[mask]).mean())
        cf = float(conf[mask].mean())
        ece += (n_b / len(y)) * abs(acc - cf)
        bins.append({"bin": b, "lo": float(lo), "hi": float(hi), "n": n_b, "acc": acc, "conf": cf})
    return {"ece": float(ece), "n_bins": int(n_bins), "n": int(len(y)), "bins": bins}


def ece_bootstrap_ci(y_true: list[int], proba, n_boot: int = 2000, seed: int = 42) -> dict[str, float]:
    import numpy as np

    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    p = np.asarray(proba)
    n = len(y)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(ece_score(list(y[idx]), p[idx])["ece"])
    arr = np.asarray(vals)
    return {
        "point": float(ece_score(y_true, proba)["ece"]),
        "lower": float(np.percentile(arr, 2.5)),
        "upper": float(np.percentile(arr, 97.5)),
        "n_boot": int(n_boot),
    }


def tune_temperature_cv(logits, y_true: list[int], temps: list[float] | None = None, n_splits: int = 3, seed: int = 42) -> dict:
    """Pick T minimizing mean val-fold NLL (stratified when possible)."""
    import numpy as np

    ts = list(temps or TEMP_GRID)
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true)
    try:
        from sklearn.model_selection import StratifiedKFold

        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(z, y))
    except Exception:
        from sklearn.model_selection import KFold

        splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(z))
    mean_nll: dict[float, float] = {}
    for t in ts:
        vals = []
        for _, va in splits:
            vals.append(nll_score(list(y[va]), softmax_temp(z[va], t)))
        mean_nll[t] = float(np.mean(vals))
    best = min(ts, key=lambda t: mean_nll[t])
    return {"best_temp": float(best), "mean_nll_by_temp": {str(k): v for k, v in mean_nll.items()}, "n_splits": n_splits}


def energy_score(logits, temp: float = 1.0):
    """Energy ``E = -T * logsumexp(z / T)``. Lower = more in-distribution."""
    import numpy as np

    z = np.asarray(logits, dtype=np.float64) / float(temp)
    m = z.max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp(z - m).sum(axis=1, keepdims=True))).ravel()
    return (-float(temp) * lse).tolist()


def oos_metrics(id_scores: list[float], oos_scores: list[float]) -> dict[str, float]:
    """FPR95 / AUROC with ID as positive class (higher = more ID).

    Energy is lower-for-ID, so callers must negate it before passing in.
    """
    import numpy as np

    try:
        from sklearn.metrics import auc, roc_curve
    except ImportError:
        return {"fpr95": float("nan"), "auroc": float("nan"), "n_id": len(id_scores), "n_oos": len(oos_scores)}
    y = np.array([1] * len(id_scores) + [0] * len(oos_scores))
    s = np.array(list(id_scores) + list(oos_scores), dtype=float)
    fpr, tpr, _ = roc_curve(y, s)
    auroc = float(auc(fpr, tpr))
    # FPR when TPR >= 0.95
    idx = np.where(tpr >= 0.95)[0]
    fpr95 = float(fpr[idx[0]]) if len(idx) else 1.0
    return {"fpr95": fpr95, "auroc": auroc, "n_id": int(len(id_scores)), "n_oos": int(len(oos_scores))}


def reliability_plot(bins: list[dict], path, title: str = "Reliability diagram (10-bin)") -> str:
    """Matplotlib reliability diagram. Lazy import; raises if mpl missing."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [(b["lo"] + b["hi"]) / 2 for b in bins]
    acc = [b["acc"] for b in bins]
    conf = [b["conf"] for b in bins]
    ns = [b["n"] for b in bins]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.plot(conf, acc, marker="o", linewidth=1.5, label="model")
    # rug of bin mass via marker size is noisy; annotate top-3 bins instead
    for x, a, c, n in sorted(zip(xs, acc, conf, ns), key=lambda t: -t[3])[:3]:
        ax.annotate(f"n={n}", (c, a), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    p = str(path)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def confusion_plot(cm, labels: list[str], path, title: str = "Confusion matrix") -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import numpy as np

    arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(arr, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    p = str(path)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p
