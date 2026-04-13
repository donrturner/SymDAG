from __future__ import annotations

from importlib import import_module


class MissingDependencyError(ImportError):
    """Raised when an optional SymDAG backend dependency is unavailable."""


def _package_prefix() -> str:
    return __name__.rsplit(".", 1)[0]


def load_core():
    try:
        return import_module(f"{_package_prefix()}._core")
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "SymDAG requires the symbolic-regression backend to be installed. "
            "Install the package with `pip install -e .` or add `.[full]` for optional helpers."
        ) from exc
