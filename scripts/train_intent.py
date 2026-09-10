from __future__ import annotations

"""S2.3: baseline intent classifiers T0 (regex/majority) + T1 (TF-IDF + LogReg).

Training signal is the disclosed pseudo-labeler ``heur-v1.2.0`` applied to
the frozen pools (reproducible from pool alone; humans may swap in
adjudicated labels and re-run). Thread granularity throughout: one row per
thread, train = train.parquet threads, test = golden.jsonl (reported as
overall / random-only-170 / hard-only-30, never blended).

T1: ``FeatureUnion(word 1-2, char_wb 3-5)`` capped at 20k total,
``min_df=5``, ``LogisticRegression(C=4, class_weight=balanced)``.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibrate import confusion_plot
from src.golden import STRATIFIER_NAME, STRATIFIER_VERSION, TAXONOMY, pseudo_intent
from src.intent_baseline import accuracy_macro_f1, build_t1_pipeline, majority_label, t0_predict
from src.logging_utils import get_logger
from src.manifest import write_json
from src.seed import seed_everything

logger = get_logger(__name__)


def inbound_thread_texts(df) -> dict[str, str]:
    g = df.sort_values(["thread_id", "pos"])
    out: dict[str, str] = {}
    for tid, grp in g.groupby("thread_id", sort=False):
        inbound = grp[grp["inbound"]]["text_norm"].astype(str).tolist() or grp["text_norm"].astype(str).tolist()
        out[str(tid)] = " | ".join(inbound)
    return out


def main() -> int:
    seed_everything(42)
    import joblib
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold

    train = pd.read_parquet(ROOT / "cache" / "train.parquet")
    golden_rows = [json.loads(line) for line in (ROOT / "data-sample" / "golden.jsonl").read_text(encoding="utf-8").splitlines()]

    train_text = inbound_thread_texts(train)
    tr_ids = sorted(train_text)
    X_tr = [train_text[t] for t in tr_ids]
    y_tr = [pseudo_intent(t)[0] for t in X_tr]
    logger.info("train_threads=%d", len(X_tr))

    te_text = [r["text"] for r in golden_rows]
    y_te = [r["intent"] for r in golden_rows]
    is_hard = [bool(r["hard_flag"]) for r in golden_rows]

    # ---- T0: regex + majority ----
    t0_pred = [t0_predict(t) for t in te_text]
    maj = majority_label(y_tr)
    t0_maj = [maj] * len(y_te)

    def split_report(name: str, pred: list[str]) -> dict:
        overall = accuracy_macro_f1(y_te, pred)
        rnd = accuracy_macro_f1([y for y, h in zip(y_te, is_hard) if not h],
                                [p for p, h in zip(pred, is_hard) if not h])
        hrd = accuracy_macro_f1([y for y, h in zip(y_te, is_hard) if h],
                                [p for p, h in zip(pred, is_hard) if h])
        return {"overall": overall, "random_only_170": rnd, "hard_only_30": hrd, "n": len(y_te)}

    # ---- T1: TF-IDF + LogReg ----
    t_fit = time.perf_counter()
    pipe = build_t1_pipeline()
    pipe.fit(X_tr, y_tr)
    train_s = round(time.perf_counter() - t_fit, 2)

    t_inf = time.perf_counter()
    t1_pred = list(pipe.predict(te_text))
    infer_ms = round((time.perf_counter() - t_inf) / max(1, len(te_text)) * 1000, 3)

    # 3-fold stratified CV on train (thread-grouped by construction)
    import numpy as np

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_f1: list[float] = []
    cv_acc: list[float] = []
    y_arr = np.array(y_tr)
    for tri, vai in skf.split(X_tr, y_tr):
        p = build_t1_pipeline()
        p.fit([X_tr[i] for i in tri], list(y_arr[tri]))
        pr = list(p.predict([X_tr[i] for i in vai]))
        m = accuracy_macro_f1(list(y_arr[vai]), pr)
        cv_f1.append(m["macro_f1"])
        cv_acc.append(m["accuracy"])
    logger.info("cv_macro_f1=%.3f±%.3f", float(np.mean(cv_f1)), float(np.std(cv_f1)))

    joblib.dump(pipe, ROOT / "cache" / "intent_t1.joblib")
    cm = confusion_matrix(y_te, t1_pred, labels=TAXONOMY).tolist()
    confusion_plot(cm, TAXONOMY, ROOT / "results" / "confusion_t1.png", title="T1 confusion (golden 200)")

    metrics = {
        "version": "1.0.0",
        "labels": {"source": f"{STRATIFIER_NAME}:{STRATIFIER_VERSION}", "taxonomy": TAXONOMY},
        "split": "thread-granularity train(12633, pseudo) -> golden(200); no turn-level split",
        # Honest reading guide: train AND test labels come from the same
        # pseudo-labeler, so T0~=1.0 is a determinism check (not a claim)
        # and T1 measures learnability of the proxy, not human agreement.
        # Human-grounded evidence is results/agreement.json (kappa 0.809);
        # external method anchor is report/banking_calibration.md (S2.5).
        "caveat": "pseudo-labels both sides; T0~1.0 is circular by construction",
        "t0_regex": split_report("t0", t0_pred),
        "t0_majority": {**split_report("maj", t0_maj), "majority_label": maj},
        "t1": {
            **split_report("t1", t1_pred),
            "cv_macro_f1_mean": round(float(__import__("numpy").mean(cv_f1)), 4),
            "cv_macro_f1_std": round(float(__import__("numpy").std(cv_f1)), 4),
            "cv_accuracy_mean": round(float(__import__("numpy").mean(cv_acc)), 4),
            "train_seconds": train_s,
            "infer_ms_per_doc": infer_ms,
            "params": {"max_features": 20000, "min_df": 5, "C": 4.0, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 500},
        },
        "confusion_t1": cm,
        "seed": 42,
    }
    write_json(metrics, ROOT / "results" / "intent_metrics.json")
    t1 = metrics["t1"]["overall"]
    logger.info("T1 overall acc=%.3f macroF1=%.3f (train %.1fs, infer %.3fms/doc)",
                t1["accuracy"], t1["macro_f1"], train_s, infer_ms)
    print(json.dumps({"t0_macro_f1": metrics["t0_regex"]["overall"]["macro_f1"],
                       "t1_macro_f1": t1["macro_f1"], "t1_acc": t1["accuracy"],
                       "train_s": train_s, "infer_ms": infer_ms}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
