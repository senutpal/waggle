from __future__ import annotations

"""S2.4: calibrated embedding classifier (T2) + temperature scaling.

- T1 (TF-IDF+LR, from S2.3) and T2 (frozen all-MiniLM-L6-v2 + LR) heads are
  each temperature-scaled (grid T 0.1-3.0, 3-fold CV on train-OOF NLL).
- Reported on golden (overall / random-170 / hard-30): ECE-10bin with
  bootstrap CI, NLL before/after, reliability.png for the SELECTED head.
- Energy E=-T·logsumexp(z/T) is OOS-detection ONLY (hard-30 as the OOD
  proxy, disclosed); no per-intent Mahalanobis anywhere (singular at
  ~20/class in 384-d).
- Decision rule: adopt T2 iff overall macro-F1 >= T1 + 2pp, else stay on
  T1 (recorded in metrics). Offline/no-dep environments skip T2 with a
  logged reason and still calibrate T1 (repro default path).
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibrate import (
    TEMP_GRID,
    confusion_plot,
    ece_bootstrap_ci,
    ece_score,
    energy_score,
    nll_score,
    oos_metrics,
    reliability_plot,
    softmax_temp,
)
from src.golden import TAXONOMY, pseudo_intent
from src.intent_baseline import accuracy_macro_f1, build_t1_pipeline
from src.logging_utils import get_logger
from src.manifest import write_json
from src.seed import seed_everything

logger = get_logger(__name__)

LABEL2IDX = {lab: i for i, lab in enumerate(TAXONOMY)}


def inbound_thread_texts(df) -> dict[str, str]:
    g = df.sort_values(["thread_id", "pos"])
    out: dict[str, str] = {}
    for tid, grp in g.groupby("thread_id", sort=False):
        inbound = grp[grp["inbound"]]["text_norm"].astype(str).tolist() or grp["text_norm"].astype(str).tolist()
        out[str(tid)] = " | ".join(inbound)
    return out


def oof_logits_t1(X: list[str], y_idx: list[int]):
    """3-fold OOF decision-function logits for T1 (for temp tuning)."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof = np.zeros((len(X), len(TAXONOMY)))
    fold_nll: dict[float, list[float]] = {t: [] for t in TEMP_GRID}
    for tri, vai in skf.split(X, y_idx):
        p = build_t1_pipeline()
        p.fit([X[i] for i in tri], [y_idx[i] for i in tri])
        z = p.decision_function([X[i] for i in vai])
        oof[vai] = z
        for t in TEMP_GRID:
            fold_nll[t].append(nll_score([y_idx[i] for i in vai], softmax_temp(z, t)))
    mean_nll = {t: float(np.mean(v)) for t, v in fold_nll.items()}
    best = min(TEMP_GRID, key=lambda t: mean_nll[t])
    return oof, {"best_temp": float(best), "mean_nll_by_temp": {str(k): v for k, v in mean_nll.items()}}


def fit_lr_head(Z_tr, y_tr):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=4.0, class_weight="balanced", solver="lbfgs", max_iter=500, random_state=42)
    clf.fit(Z_tr, y_tr)
    return clf


def main() -> int:
    seed_everything(42)
    import joblib
    import numpy as np
    import pandas as pd

    t_start = time.perf_counter()
    train = pd.read_parquet(ROOT / "cache" / "train.parquet")
    golden_rows = [json.loads(line) for line in (ROOT / "data-sample" / "golden.jsonl").read_text(encoding="utf-8").splitlines()]
    metrics = json.loads((ROOT / "results" / "intent_metrics.json").read_text(encoding="utf-8"))

    train_text = inbound_thread_texts(train)
    tr_ids = sorted(train_text)
    X_tr = [train_text[t] for t in tr_ids]
    y_tr = [LABEL2IDX[pseudo_intent(t)[0]] for t in X_tr]
    X_te = [r["text"] for r in golden_rows]
    y_te = [LABEL2IDX[r["intent"]] for r in golden_rows]
    is_hard = [bool(r["hard_flag"]) for r in golden_rows]

    # ---- T1 calibration ----
    oof, tune = oof_logits_t1(X_tr, y_tr)
    t1_pipe = joblib.load(ROOT / "cache" / "intent_t1.joblib")
    z_raw = np.asarray(t1_pipe.decision_function(X_te))
    # column alignment: the saved pipe was trained on STRING labels, so its
    # decision_function columns follow pipe.classes_ (alphabetical), NOT
    # TAXONOMY order. Reorder to TAXONOMY before any argmax/calibration.
    order = [list(t1_pipe.classes_).index(lab) for lab in TAXONOMY]
    z_te_t1 = z_raw[:, order]
    T1 = tune["best_temp"]
    logger.info("T1 best_temp=%.2f (expect 1.5-2.5 on noisy tweets)", T1)

    # ---- T2 attempt (frozen MiniLM default; bge-small ablation deferred) ----
    t2_info: dict = {"attempted": True, "model": "sentence-transformers/all-MiniLM-L6-v2"}
    t2_pred = None
    z_te_t2 = None
    try:
        from src.calibrate import encode_t2

        Z_tr = encode_t2(X_tr)
        Z_te = encode_t2(X_te)
        t2_info["encode_seconds"] = round(time.perf_counter() - t_start, 1)
        # temp-tune T2 head on 3-fold OOF embedding logits
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        oof2 = np.zeros((len(X_tr), len(TAXONOMY)))
        fold_nll2: dict[float, list[float]] = {t: [] for t in TEMP_GRID}
        for tri, vai in skf.split(Z_tr, y_tr):
            clf = fit_lr_head(Z_tr[tri], [y_tr[i] for i in tri])
            z = clf.decision_function(Z_tr[vai])
            oof2[vai] = z
            for t in TEMP_GRID:
                fold_nll2[t].append(nll_score([y_tr[i] for i in vai], softmax_temp(z, t)))
        mean2 = {t: float(np.mean(v)) for t, v in fold_nll2.items()}
        T2 = float(min(TEMP_GRID, key=lambda t: mean2[t]))
        head2 = fit_lr_head(Z_tr, y_tr)
        z2_raw = np.asarray(head2.decision_function(Z_te))
        order2 = [list(head2.classes_).index(i) for i in range(len(TAXONOMY))]
        z_te_t2 = z2_raw[:, order2]
        p_te_t2 = softmax_temp(z_te_t2, T2)
        t2_pred = list(np.argmax(p_te_t2, axis=1))
        t2_info.update({"ok": True, "best_temp": T2, "dim": 384,
                        "bge_ablation": "deferred (not downloaded; default-only per $0 budget)"})
        joblib.dump({"head": head2, "temp": T2, "model": t2_info["model"]}, ROOT / "cache" / "intent_t2.joblib")
        np.asarray(Z_te, dtype=np.float16).dump(ROOT / "cache" / "embeddings_200_fp16.npy")
        logger.info("T2 ok best_temp=%.2f", T2)
    except Exception as e:  # offline / no full deps -> stay on T1 (spec-allowed)
        t2_info.update({"ok": False, "reason": f"{type(e).__name__}: {str(e)[:200]}",
                        "bge_ablation": "deferred"})
        logger.info("T2 skipped (%s); staying on T1", t2_info["reason"])

    # ---- head-to-head on golden (overall; splits reported too) ----
    p_te_t1 = softmax_temp(z_te_t1, T1)
    pred_t1 = list(np.argmax(p_te_t1, axis=1))
    t1_overall = accuracy_macro_f1([TAXONOMY[i] for i in y_te], [TAXONOMY[i] for i in pred_t1])
    decision = {"rule": "adopt T2 iff overall macro-F1 >= T1 + 0.02 else stay on T1"}
    if t2_pred is not None:
        t2_overall = accuracy_macro_f1([TAXONOMY[i] for i in y_te], [TAXONOMY[i] for i in t2_pred])
        t2_rnd = accuracy_macro_f1([TAXONOMY[y] for y, h in zip(y_te, is_hard) if not h],
                                   [TAXONOMY[p] for p, h in zip(t2_pred, is_hard) if not h])
        t2_hrd = accuracy_macro_f1([TAXONOMY[y] for y, h in zip(y_te, is_hard) if h],
                                   [TAXONOMY[p] for p, h in zip(t2_pred, is_hard) if h])
        delta = t2_overall["macro_f1"] - t1_overall["macro_f1"]
        decision.update({"t1_macro_f1": round(t1_overall["macro_f1"], 4),
                         "t2_macro_f1": round(t2_overall["macro_f1"], 4),
                         "t2_random_only_170": t2_rnd, "t2_hard_only_30": t2_hrd,
                         "delta": round(delta, 4),
                         "selected": "t2" if delta >= 0.02 else "t1"})
    else:
        decision.update({"t1_macro_f1": round(t1_overall["macro_f1"], 4), "selected": "t1"})
    sel = decision["selected"]
    logger.info("head decision: %s", decision)

    # ---- calibration report for the SELECTED head ----
    z_sel = z_te_t2 if (sel == "t2" and z_te_t2 is not None) else z_te_t1
    T_sel = (t2_info.get("best_temp", T1) if sel == "t2" else T1)
    p_before = softmax_temp(z_sel, 1.0)
    p_after = softmax_temp(z_sel, T_sel)
    ece_b = ece_score(y_te, p_before)
    ece_a = ece_score(y_te, p_after)
    ece_ci = ece_bootstrap_ci(y_te, p_after, n_boot=2000, seed=42)
    calib = {
        "head": sel,
        "best_temp": float(T_sel),
        "t1_temp_tune": tune,
        "nll_before": round(nll_score(y_te, p_before), 4),
        "nll_after": round(nll_score(y_te, p_after), 4),
        "ece_before": round(ece_b["ece"], 4),
        "ece_after": round(ece_a["ece"], 4),
        "ece_after_ci95": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ece_ci.items()},
        "ece_gate_lt_0_05": bool(ece_a["ece"] < 0.05),
        "bins_after": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()} for b in ece_a["bins"]],
    }
    reliability_plot(ece_a["bins"], ROOT / "results" / "reliability.png",
                     title=f"Reliability ({sel}, T={T_sel}, ECE={ece_a['ece']:.3f})")
    if sel == "t2":
        from sklearn.metrics import confusion_matrix

        confusion_plot(confusion_matrix(y_te, t2_pred).tolist(), TAXONOMY,
                       ROOT / "results" / "confusion_t2.png", title="T2 confusion (golden 200)")

    # ---- Energy OOS (hard-30 as disclosed OOD proxy; T1 head, always available) ----
    e_all = energy_score(z_te_t1, temp=T1)
    neg = [-v for v in e_all]
    id_s = [s for s, h in zip(neg, is_hard) if not h]
    oos_s = [s for s, h in zip(neg, is_hard) if h]
    oos = {**oos_metrics(id_s, oos_s),
           "proxy": "hard-30 (sarcasm/multi/low-margin) as OOD vs random-170 as ID; disclosed proxy, not true OOS",
           "score": "neg-energy from T1 temp-scaled logits"}
    logger.info("energy OOS fpr95=%.3f auroc=%.3f", oos["fpr95"], oos["auroc"])

    metrics["t2"] = {**t2_info, "decision": decision}
    metrics["calibration"] = calib
    metrics["energy_oos"] = oos
    metrics["total_seconds_s2_4"] = round(time.perf_counter() - t_start, 1)
    write_json(metrics, ROOT / "results" / "intent_metrics.json")
    write_json({"head": sel, "temp": float(T_sel), "t1": tune["best_temp"], "seed": 42},
               ROOT / "cache" / "intent_calib.json")
    print(json.dumps({"selected": sel, "T": T_sel, "nll_delta": round(calib["nll_after"] - calib["nll_before"], 4),
                       "ece_after": calib["ece_after"], "ece_ci": [ece_ci["lower"], ece_ci["upper"]],
                       "fpr95": oos["fpr95"], "auroc": oos["auroc"],
                       "secs": metrics["total_seconds_s2_4"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
