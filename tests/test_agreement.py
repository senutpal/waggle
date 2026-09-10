from __future__ import annotations

import csv
import json
from pathlib import Path

from src.agree import bootstrap_ci, cohen_kappa, krippendorff_alpha_nominal, per_class_kappa
from src.golden import TAXONOMY

ROOT = Path(__file__).resolve().parents[1]


def test_agreement_gate_values():
    a = json.loads((ROOT / "results" / "agreement.json").read_text(encoding="utf-8"))
    assert a["n"] == 50 and a["disjoint_from_golden200"] is True
    assert a["pass"] is True
    assert a["cohen_kappa"]["point"] >= 0.6
    assert a["cohen_kappa"]["lower"] > 0.4
    assert set(a["per_class_kappa"]) == set(TAXONOMY)
    # NO quadratic-weighted kappa anywhere in the intent agreement path
    assert "qw" not in json.dumps(a).lower().replace("follow", "")


def test_pilot_disjoint_and_sealed():
    blind = list(csv.DictReader((ROOT / "data-sample" / "pilot_50_blind.csv").open(encoding="utf-8")))
    assert len(blind) == 50
    assert "label" not in blind[0]
    golden_ids = {json.loads(line)["golden_thread_id"] for line in (ROOT / "data-sample" / "golden.jsonl").read_text(encoding="utf-8").splitlines()}
    assert not ({r["thread_id"] for r in blind} & golden_ids)
    ra = {r["pilot_id"]: r["label"] for r in csv.DictReader((ROOT / "data-sample" / "pilot_50_rater_a.csv").open(encoding="utf-8"))}
    rb = {r["pilot_id"]: r["label"] for r in csv.DictReader((ROOT / "data-sample" / "pilot_50_rater_b.csv").open(encoding="utf-8"))}
    assert set(ra) == {r["pilot_id"] for r in blind} == set(rb)
    assert all(v in TAXONOMY for v in list(ra.values()) + list(rb.values()))


def test_kappa_helpers_sanity():
    assert cohen_kappa(["a", "a", "b"], ["a", "a", "b"]) == 1.0
    assert krippendorff_alpha_nominal(["a", "a", "b"], ["a", "a", "b"]) == 1.0
    ci = bootstrap_ci(cohen_kappa, ["a"] * 20 + ["b"] * 20, ["a"] * 20 + ["b"] * 20, n_boot=200, seed=42)
    assert ci["lower"] <= ci["point"] <= ci["upper"]
    per = per_class_kappa(["a", "b", "a"], ["a", "a", "a"], ["a", "b"])
    assert set(per) == {"a", "b"}
