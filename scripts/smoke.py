from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.seed import env_snapshot, seed_everything


def main() -> int:
    seed_everything(42)
    print("waggle smoke")
    for k, v in sorted(env_snapshot().items()):
        print(f"{k}={v}")
    import src

    print(f"src={src.__version__}")
    base = ROOT / "config" / "base.yaml"
    print(f"config_exists={base.exists()}")
    if not base.exists():
        return 1
    if "--offline" in sys.argv or len(sys.argv) == 1:
        print("offline ok, no key, no network")
    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
