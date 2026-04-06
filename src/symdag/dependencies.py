from __future__ import annotations

from importlib import import_module


class MissingDependencyError(ImportError):
    """Raised when an optional SymDAG backend dependency is unavailable."""


def _package_prefix() -> str:
    return __name__.rsplit(".", 1)[0]


def load_bayes_core():
    try:
        return import_module(f"{_package_prefix()}._bayes_core")
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "SymDAG-Bayes requires the Bayesian symbolic-regression stack to be installed. "
            "Install the package with the `bayes` or `full` extras."
        ) from exc


def load_greedy_core():
    try:
        return import_module(f"{_package_prefix()}._greedy_core")
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "SymDAG-Greedy requires `rils-rols` and the greedy backend dependencies. "
            "Install the package with the `greedy` or `full` extras."
        ) from exc
