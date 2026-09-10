from __future__ import annotations

import importlib
import importlib.util
from typing import Any


def find_spec(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def import_or_none(name: str) -> Any | None:
    if importlib.util.find_spec(name) is None:
        return None
    return importlib.import_module(name)


def require_extra(name: str, extra_hint: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(f"missing optional dependency {name!r}: {extra_hint}")
