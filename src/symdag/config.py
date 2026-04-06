from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence


DEFAULT_BAYES_OPERATORS = ("+", "-", "*", "cos")


@dataclass
class DataGenerationConfig:
    n: int = 2000
    d: int = 20
    sigma: float = 0.1
    gen_method: str = "gp"
    seed: int = 42
    sparsity: float = 1.0
    edgeprob: Optional[float] = None

    def resolved_edgeprob(self) -> float:
        if self.edgeprob is not None:
            return float(self.edgeprob)
        if self.d <= 1:
            raise ValueError("d must be greater than 1 to infer edgeprob from sparsity.")
        return (2.0 * float(self.sparsity)) / float(self.d - 1)


@dataclass
class BayesConfig:
    n_iter: int = 3000
    burnin: int = 2000
    random_state: Optional[int] = 42
    checkpoint_dir: Optional[str] = None
    max_equation_evals: Optional[int] = None
    final_max_equation_evals: int = 4000
    score_method: str = "best"
    model_selection: str = "max_nml"
    samples_per_fit: int = 5
    n_chains: int = 1
    chain_n_jobs: Optional[int] = None
    edge_threshold: float = 0.3
    use_efdr_threshold: bool = True
    efdr_q: float = 0.05
    use_data_driven_threshold: bool = False
    target_n_edges: Optional[int] = None
    edgeprob: Optional[float] = None
    use_bn_warm_start: bool = True
    operators: Optional[Sequence[str]] = field(default_factory=lambda: list(DEFAULT_BAYES_OPERATORS))
    max_complexity: int = 25
    num_particles: int = 100
    num_mcmc_samples: int = 5


@dataclass
class GreedyConfig:
    sample_size: int = 1
    max_complexity: int = 35
    verbose: bool = False
    max_fit_calls: int = 10000
    operators: Optional[Sequence[str]] = None
    regressor_kwargs: Dict[str, Any] = field(default_factory=dict)
