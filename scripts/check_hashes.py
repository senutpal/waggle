from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.manifest import verify_manifest
from src.seed import seed_everything


def main() -> int:
    seed_everything(42)
    ok = verify_manifest(ROOT / "cache" / "MANIFEST.sha256")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
