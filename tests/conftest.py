from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _determinism(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "42")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
