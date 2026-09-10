from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.manifest import cache_size_mb, verify_manifest, write_manifest
from src.split import assert_gap

ROOT = Path(__file__).resolve().parents[1]


def test_assert_gap_pass_and_fail():
    d = assert_gap("2017-10-13T22:41:07Z", "2017-10-15T00:12:33Z", 24)
    assert d > 24
    with pytest.raises(AssertionError):
        assert_gap("2017-10-14T12:00:00Z", "2017-10-15T00:00:00Z", 24)


def test_manifest_roundtrip_tmp(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("hello", encoding="utf-8")
    man = tmp_path / "MANIFEST.sha256"
    write_manifest([a], out=man)
    assert verify_manifest(man) is True
    a.write_text("tampered", encoding="utf-8")
    assert verify_manifest(man) is False


def test_cache_guard_small_tmp(tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"x" * 1024)
    assert cache_size_mb(tmp_path) < 150


def test_frozen_manifests_exist_and_consistent():
    ts = ROOT / "manifests" / "temporal_split.json"
    la = ROOT / "manifests" / "leakage_audit.json"
    assert ts.exists() and la.exists()
    m = json.loads(ts.read_text(encoding="utf-8"))
    a = json.loads(la.read_text(encoding="utf-8"))
    assert m["gap_assert"]["pass"] is True
    assert m["gap_assert"]["delta_hours"] >= 24
    assert a["exact_overlap"] == 0
    assert a["pass"] is True
    ev = ROOT / "data-sample" / "eval_200.parquet"
    assert ev.exists()
    df = pd.read_parquet(ev, columns=["created_at"])
    assert (pd.to_datetime(df["created_at"], utc=True) >= pd.Timestamp("2017-10-15T00:00:00Z", tz="UTC")).all()
