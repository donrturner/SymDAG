from __future__ import annotations

from typing import Optional

from .config import DataGenerationConfig, SymDAGConfig
from .estimator import run_symdag as _run_symdag_impl
from .results import SimulationResult, SymDAGResult
from .simulation import simulate_dataset


def run_symdag(data, config: Optional[SymDAGConfig] = None, **overrides) -> SymDAGResult:
    if config is not None and not isinstance(config, SymDAGConfig):
        raise TypeError("SymDAG runs require a SymDAGConfig instance.")
    return _run_symdag_impl(data, config=config, **overrides)


def simulate_and_fit(
    data_config: Optional[DataGenerationConfig] = None,
    config: Optional[SymDAGConfig] = None,
    **overrides,
) -> tuple[SimulationResult, SymDAGResult]:
    simulation = simulate_dataset(data_config or DataGenerationConfig())
    result = run_symdag(simulation.data, config=config, **overrides)
    return simulation, result
