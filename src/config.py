from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    seed: int = 42
    raw_path: Path = Path("data-raw/twcs.csv")
    sample_path: Path = Path("data-sample/sample.csv")
    cache_dir: Path = Path("cache")
    manifests_dir: Path = Path("manifests")
    eval_parquet: Path = Path("data-sample/eval_200.parquet")
    train_before: str = "2017-10-15"
    gap_hours: int = 24


@dataclass(frozen=True)
class FullConfig:
    seed: int = 42
    raw: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)


def config_path(name: str = "base.yaml") -> Path:
    return ROOT / "config" / name


def load_config(name: str = "base.yaml") -> dict:
    import yaml

    path = config_path(name)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_typed_config(name: str = "base.yaml") -> Config:
    raw = load_config(name)
    paths = raw.get("paths", {})
    temporal = raw.get("temporal", {})
    return Config(
        seed=int(raw.get("seed", 42)),
        raw_path=Path(paths.get("raw", "data-raw/twcs.csv")),
        sample_path=Path(paths.get("sample", "data-sample/sample.csv")),
        cache_dir=Path(paths.get("cache", "cache")),
        manifests_dir=Path(paths.get("manifests", "manifests")),
        eval_parquet=Path(paths.get("eval_parquet", "data-sample/eval_200.parquet")),
        train_before=str(temporal.get("cut", "2017-10-15"))[:10],
        gap_hours=int(temporal.get("gap_hours", 24)),
    )


def resolve(rel: str | Path) -> Path:
    return (ROOT / rel).resolve()


def ensure_dirs(cfg: Config) -> None:
    for rel in ("cache", "data-sample", "results", "report", "manifests", "data-raw"):
        (ROOT / rel).mkdir(parents=True, exist_ok=True)
