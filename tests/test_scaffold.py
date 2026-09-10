from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "requirements-repro.txt",
    "requirements-full.txt",
    ".python-version",
    "config/base.yaml",
    "run.sh",
    "run.bat",
    "Makefile",
    "src/__init__.py",
    "src/config.py",
    "src/seed.py",
    "src/io.py",
    "src/threads.py",
    "src/normalize.py",
    "src/split.py",
    "src/manifest.py",
    "scripts/run.py",
    "scripts/smoke.py",
    "scripts/check_hashes.py",
]


def test_required_entries_exist():
    for rel in REQUIRED:
        assert (ROOT / rel).exists(), f"missing {rel}"
    for d in ("src", "eval", "cache", "data-sample", "results", "report", "manifests"):
        assert (ROOT / d).is_dir(), f"missing dir {d}"


def test_import_src_fast_offline():
    t0 = time.perf_counter()
    code = "import sys; sys.path.insert(0, r'%s'); import src; print(src.__version__)" % str(ROOT)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    dt = time.perf_counter() - t0
    assert r.returncode == 0, r.stderr
    assert dt < 2.0 + 3.0, f"import too slow: {dt}"
    assert "torch" not in sys.modules


def test_no_torch_in_repro_requirements():
    text = (ROOT / "requirements-repro.txt").read_text(encoding="utf-8").lower()
    assert "torch" not in text
    assert "sentence-transformers" not in text
    assert "transformers" not in text
