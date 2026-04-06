from __future__ import annotations

import copy
import numbers
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Optional

from .config import DEFAULT_BAYES_OPERATORS, BayesConfig
from .dependencies import load_bayes_core
from .results import SymDAGResult


@contextmanager
def _configured_core(core, config: BayesConfig):
    operators = list(config.operators) if config.operators is not None else list(DEFAULT_BAYES_OPERATORS)
    attrs = {
        "COMP": int(config.max_complexity),
        "PARTS": int(config.num_particles),
        "SAMP": int(config.num_mcmc_samples),
        "SR_OPERATORS": operators,
    }
    old_values = {name: copy.deepcopy(getattr(core, name)) for name in attrs}
    try:
        for name, value in attrs.items():
            setattr(core, name, value)
        yield
    finally:
        for name, value in old_values.items():
            setattr(core, name, value)


def _format_ordering(ordering: Optional[Iterable[Any]]):
    if ordering is None:
        return None
    out = []
    for node in ordering:
        if isinstance(node, numbers.Integral):
            out.append(f"X_{int(node)}")
        else:
            out.append(str(node))
    return out


def run_bayes(data, config: Optional[BayesConfig] = None, **overrides) -> SymDAGResult:
    config = BayesConfig() if config is None else config
    if overrides:
        config = replace(config, **overrides)

    core = load_bayes_core()
    checkpoint_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else None

    with _configured_core(core, config):
        raw = core.symdag_bayes_stochastic_v2(
            data,
            n_iter=config.n_iter,
            random_state=config.random_state,
            burnin=config.burnin,
            checkpoint_dir=checkpoint_dir,
            max_equation_evals=config.max_equation_evals,
            final_max_equation_evals=config.final_max_equation_evals,
            score_method=config.score_method,
            sel_type=config.model_selection,
            samples_per_fit=config.samples_per_fit,
            n_chains=config.n_chains,
            chain_n_jobs=config.chain_n_jobs,
            edge_threshold=config.edge_threshold,
            use_efdr_threshold=config.use_efdr_threshold,
            efdr_q=config.efdr_q,
            use_data_driven_threshold=config.use_data_driven_threshold,
            target_n_edges=config.target_n_edges,
            edgeprob=config.edgeprob,
            use_bn_warm_start=config.use_bn_warm_start,
        )

    return SymDAGResult(
        method="bayes",
        graph=raw["graph"],
        expressions=dict(raw.get("expressions", {})),
        ordering=_format_ordering(raw.get("ordering")),
        runtime=raw.get("time"),
        details=raw,
    )
