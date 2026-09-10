from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from src.golden import TAXONOMY

ROOT = Path(__file__).resolve().parents[1]


def _golden_rows() -> list[dict]:
    return [json.loads(line) for line in (ROOT / "data-sample" / "golden.jsonl").read_text(encoding="utf-8").splitlines()]


def test_golden_size_and_quota():
    rows = _golden_rows()
    assert len(rows) == 200
    hard = [r for r in rows if r["hard_flag"]]
    rnd = [r for r in rows if not r["hard_flag"]]
    assert len(hard) == 30 and len(rnd) == 170
    from collections import Counter

    c = Counter(r["intent"] for r in rnd)
    assert set(c) == set(TAXONOMY)
    assert all(v >= 12 for v in c.values()), c
    assert all(v <= 30 for v in c.values()), c


def test_golden_schema_and_temporal_gate():
    rows = _golden_rows()
    need = {"sample_id", "golden_thread_id", "text", "created_at", "intent", "sentiment",
            "len_bucket", "resolved", "hard_flag", "stratum_id", "text_hash"}
    for r in rows:
        assert need <= set(r), set(r)
        assert r["intent"] in TAXONOMY
        assert r["sentiment"] in {"positive", "negative", "neutral"}
        assert r["len_bucket"] in {"short", "medium", "long"}
        assert pd.Timestamp(r["created_at"], tz="UTC") >= pd.Timestamp("2017-10-15T00:00:00Z", tz="UTC")


def test_golden_no_train_overlap():
    rows = _golden_rows()
    train = pd.read_parquet(ROOT / "cache" / "train.parquet")
    g = train.sort_values(["thread_id", "pos"])
    inb = g[g["inbound"]].groupby("thread_id")["text_norm"].apply(lambda s: " | ".join(s.astype(str)))
    train_keys = {" ".join(v.strip().lower().split()) for v in inb.tolist()}
    for r in rows:
        assert " ".join(r["text"].strip().lower().split()) not in train_keys


def test_golden_eval_parquet_matches():
    rows = _golden_rows()
    ids = {r["golden_thread_id"] for r in rows}
    ev = pd.read_parquet(ROOT / "data-sample" / "eval_200.parquet")
    assert set(ev["thread_id"].astype(str)) == ids
    assert (pd.to_datetime(ev["created_at"], utc=True) >= pd.Timestamp("2017-10-15T00:00:00Z", tz="UTC")).all()


def test_leakage_manifest_pass():
    m = json.loads((ROOT / "manifests" / "s2_1_golden.json").read_text(encoding="utf-8"))
    assert m["pass"] is True and m["exact_overlap"] == 0 and m["n"] == 200
    a = json.loads((ROOT / "manifests" / "leakage_audit.json").read_text(encoding="utf-8"))
    assert a["exact_overlap"] == 0 and a["pass"] is True
    flags = list(csv.DictReader((ROOT / "data-sample" / "golden_review_flags.csv").open(encoding="utf-8")))
    assert m["tfidf_flag"]["n_flagged"] == len(flags)
