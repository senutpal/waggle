from __future__ import annotations

import json
from pathlib import Path

from src.config import ROOT
from src.logging_utils import get_logger, sha256_file

logger = get_logger(__name__)


def write_manifest(paths: list[Path], out: Path | None = None) -> Path:
    dest = out or (ROOT / "cache" / "MANIFEST.sha256")
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for p in sorted(paths, key=lambda x: x.name):
        if p.name == "MANIFEST.sha256":
            continue
        if p.is_file():
            lines.append(f"{sha256_file(p)}  {p.name}")
    dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return dest


def verify_manifest(manifest: Path | None = None) -> bool:
    src = manifest or (ROOT / "cache" / "MANIFEST.sha256")
    if not src.exists():
        print(f"FAIL: missing {src}")
        return False
    ok = True
    for lineno, line in enumerate(src.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            print(f"FAIL line {lineno}: malformed")
            ok = False
            continue
        expected, rel = parts
        rel = rel.lstrip("* ").strip()
        target = (src.parent / rel).resolve()
        if not target.exists():
            print(f"FAIL: missing {rel}")
            ok = False
            continue
        got = sha256_file(target)
        if got.lower() != expected.lower():
            print(f"FAIL: {rel} expected {expected} got {got}")
            ok = False
        else:
            print(f"OK: {rel}")
    listed: set[str] = set()
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.strip():
            listed.add(line.strip().split(None, 1)[1].lstrip("* ").strip())
    for p in sorted((src.parent).glob("*")):
        if p.name == "MANIFEST.sha256":
            continue
        if p.is_file() and p.name not in listed:
            print(f"WARN: unlisted cache file {p.name}")
    print("PASS" if ok else "FAIL")
    return ok


def write_json(obj: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def cache_size_mb(path: Path | None = None) -> float:
    root = path or (ROOT / "cache")
    total = 0
    for p in root.glob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)
