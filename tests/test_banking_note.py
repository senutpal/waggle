from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_banking77_absent_from_src():
    hits = [p for p in (ROOT / "src").rglob("*.py") if "banking77" in p.read_text(encoding="utf-8").lower()]
    assert hits == []


def test_banking_note_committed():
    md = (ROOT / "report" / "banking_calibration.md").read_text(encoding="utf-8")
    assert "no banking" in md.lower()
    assert "weights" in md.lower()
