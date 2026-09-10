from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]


def __getattr__(name: str):
    import importlib

    return importlib.import_module(f"src.{name}")
