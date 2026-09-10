from __future__ import annotations


def main() -> None:
    from src.config import ROOT
    from src.seed import env_snapshot

    print("waggle 0.1.0")
    print(f"root={ROOT}")
    for k, v in sorted(env_snapshot().items()):
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
