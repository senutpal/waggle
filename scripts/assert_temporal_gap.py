from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.seed import seed_everything
from src.split import assert_gap


def main() -> int:
    seed_everything(42)
    mpath = ROOT / "manifests" / "temporal_split.json"
    if not mpath.exists():
        print(f"FAIL: missing {mpath}")
        return 1
    m = json.loads(mpath.read_text(encoding="utf-8"))
    delta = assert_gap(m["train"]["max_created_at"], m["golden_pool"]["min_created_at"], int(m.get("gap_hours", 24)))
    print(f"PASS gap {delta:.2f}h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
