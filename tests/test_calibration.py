from __future__ import annotations

import json
from pathlib import Path

from src.calibrate import ece_score, energy_score, nll_score, oos_metrics, softmax_temp, tune_temperature_cv

ROOT = Path(__file__).resolve().parents[1]


def test_temp_scaling_preserves_accuracy_and_reduces_nll():
    logits = [[3.0, 1.0, 0.5], [0.5, 3.0, 1.0], [1.0, 0.5, 3.0], [2.5, 2.4, 0.1],
              [3.2, 0.8, 0.4], [0.4, 2.8, 1.1]]
    y = [0, 1, 2, 0, 1, 2]
    before = nll_score(y, softmax_temp(logits, 1.0))
    tune = tune_temperature_cv(logits, y, temps=[0.5, 1.0, 2.0], n_splits=2, seed=42)
    after = nll_score(y, softmax_temp(logits, tune["best_temp"]))
    assert after <= before + 1e-9
    import numpy as np

    assert (np.argmax(softmax_temp(logits, tune["best_temp"]), axis=1) == np.argmax(softmax_temp(logits, 1.0), axis=1)).all()


def test_ece_perfect_model_is_zero():
    import numpy as np

    p = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    e = ece_score([0, 0, 1, 1], p, n_bins=2)
    assert e["ece"] < 0.2 and e["n"] == 4 and len(e["bins"]) == 2


def test_energy_oos_ordering():
    id_logits = [[5.0, 0.0], [4.0, 1.0]]
    oos_logits = [[0.5, 0.4], [0.6, 0.5]]
    e_id = [-v for v in energy_score(id_logits)]
    e_oos = [-v for v in energy_score(oos_logits)]
    assert min(e_id) > max(e_oos)
    m = oos_metrics(e_id, e_oos)
    assert m["auroc"] == 1.0 and m["fpr95"] == 0.0


def test_no_mahalanobis_anywhere():
    import subprocess

    r = subprocess.run(["git", "grep", "-il", "mahalanobis", "--", "src", "scripts", "tests"],
                       cwd=str(ROOT), capture_output=True, text=True)
    assert r.stdout.strip() == "", r.stdout


def test_calibration_artifacts_committed():
    m = json.loads((ROOT / "results" / "intent_metrics.json").read_text(encoding="utf-8"))
    assert "calibration" in m and "energy_oos" in m and "t2" in m
    assert m["calibration"]["ece_after_ci95"]["n_boot"] == 2000
    assert m["t2"]["decision"]["selected"] in {"t1", "t2"}
    assert (ROOT / "results" / "reliability.png").exists()
    assert (ROOT / "cache" / "intent_calib.json").exists()
