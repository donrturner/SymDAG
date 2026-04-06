from __future__ import annotations

from typing import Optional

from .bayes import run_bayes
from .config import BayesConfig, DataGenerationConfig, GreedyConfig
from .greedy import run_greedy
from .results import SimulationResult, SymDAGResult
from .simulation import simulate_dataset


def run_symdag(data, method: str = "bayes", config=None, **overrides) -> SymDAGResult:
    method = method.lower().strip()
    if method == "bayes":
        if config is not None and not isinstance(config, BayesConfig):
            raise TypeError("Bayesian runs require a BayesConfig instance.")
        return run_bayes(data, config=config, **overrides)
    if method == "greedy":
        if config is not None and not isinstance(config, GreedyConfig):
            raise TypeError("Greedy runs require a GreedyConfig instance.")
        return run_greedy(data, config=config, **overrides)
    raise ValueError("method must be either 'bayes' or 'greedy'.")


def simulate_and_fit(
    method: str = "bayes",
    data_config: Optional[DataGenerationConfig] = None,
    config=None,
    **overrides,
) -> tuple[SimulationResult, SymDAGResult]:
    simulation = simulate_dataset(data_config or DataGenerationConfig())
    result = run_symdag(simulation.data, method=method, config=config, **overrides)
    return simulation, result
