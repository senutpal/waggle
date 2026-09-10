from __future__ import annotations

import json
import time
from pathlib import Path

from src.golden import TAXONOMY, pseudo_intent
from src.intent_baseline import accuracy_macro_f1, build_t1_pipeline, group_train_test_split, majority_label, t0_predict

ROOT = Path(__file__).resolve().parents[1]


def test_t0_priority_and_majority():
    assert t0_predict("locked out of my account, can't login") == "account-security"
    assert t0_predict("hello sir need help") == "other"
    assert majority_label([]) == "complaint-feedback"
    assert majority_label(["a", "b", "a"]) == "a"


def test_group_split_no_thread_overlap():
    tr, te = group_train_test_split(["t1", "t1", "t2", "t3", "t3", "t3"], test_size=0.34, seed=42)
    gtr = {"t1", "t2", "t3"} & {["t1", "t1", "t2", "t3", "t3", "t3"][i] for i in tr}
    gte = {["t1", "t1", "t2", "t3", "t3", "t3"][i] for i in te}
    assert not (gtr & gte)


def test_t1_trains_fast_and_infers_sub_ms():
    X = ["my package has not arrived, tracking shows delivered",
         "charged twice on my card, billing error",
         "locked out of my account password reset"] * 20
    y = [pseudo_intent(t)[0] for t in X]
    t0 = time.perf_counter()
    p = build_t1_pipeline()
    p.fit(X, y)
    assert time.perf_counter() - t0 < 60
    t1 = time.perf_counter()
    pred = list(p.predict(X[:5]))
    assert (time.perf_counter() - t1) / 5 < 0.05
    assert all(v in TAXONOMY for v in pred)


def test_intent_metrics_committed():
    m = json.loads((ROOT / "results" / "intent_metrics.json").read_text(encoding="utf-8"))
    assert "t0_regex" in m and "t1" in m
    assert m["t1"]["overall"]["macro_f1"] > 0.5
    assert m["t1"]["infer_ms_per_doc"] < 5.0
    assert (ROOT / "results" / "confusion_t1.png").exists()
    assert (ROOT / "cache" / "intent_t1.joblib").exists()
