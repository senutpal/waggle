from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.seed import seed_everything


def main() -> int:
    seed_everything(42)
    import subprocess

    cmds = [
        [sys.executable, "scripts/fetch.py"],
        [sys.executable, "scripts/build_threads.py"],
        [sys.executable, "scripts/clean.py"],
        [sys.executable, "scripts/freeze.py"],
    ]
    for cmd in cmds:
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            return r.returncode
    print("RUN PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
