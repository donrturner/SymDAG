from __future__ import annotations

import copy
import time
from contextlib import contextmanager
from dataclasses import replace
from typing import Optional

import pandas as pd

from .config import GreedyConfig
from .dependencies import load_greedy_core
from .results import SymDAGResult


def _coerce_dataframe(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    frame = pd.DataFrame(data)
    frame.columns = [f"X_{i}" for i in range(frame.shape[1])]
    return frame


@contextmanager
def _configured_core(core, config: GreedyConfig):
    attrs = {
        "GREEDY_SAMPLE_SIZE": int(config.sample_size),
        "GREEDY_MAX_COMPLEXITY": int(config.max_complexity),
        "GREEDY_VERBOSE": bool(config.verbose),
        "GREEDY_MAX_FIT_CALLS": int(config.max_fit_calls),
        "GREEDY_OPERATORS": None if config.operators is None else list(config.operators),
        "GREEDY_EXTRA_KWARGS": dict(config.regressor_kwargs),
    }
    old_values = {name: copy.deepcopy(getattr(core, name)) for name in attrs}
    try:
        for name, value in attrs.items():
            setattr(core, name, value)
        yield
    finally:
        for name, value in old_values.items():
            setattr(core, name, value)


def run_greedy(data, config: Optional[GreedyConfig] = None, **overrides) -> SymDAGResult:
    config = GreedyConfig() if config is None else config
    if overrides:
        config = replace(config, **overrides)

    core = load_greedy_core()
    frame = _coerce_dataframe(data)
    graph_init = core.DAG()
    graph_init.add_nodes_from(list(frame.columns))

    start = time.perf_counter()
    with _configured_core(core, config):
        raw = core.graphsearch_par(frame, graph_init)
    runtime = time.perf_counter() - start

    expressions = {node: expr for (_, node), expr in raw.get("functs", {}).items()}
    ordering = list(core.nx.topological_sort(raw["graph"])) if core.nx.is_directed_acyclic_graph(raw["graph"]) else None
    details = dict(raw)
    details["runtime"] = runtime

    return SymDAGResult(
        method="greedy",
        graph=raw["graph"],
        expressions=expressions,
        ordering=ordering,
        runtime=runtime,
        details=details,
    )
