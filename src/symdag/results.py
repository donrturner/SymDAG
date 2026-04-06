from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

from .config import DataGenerationConfig


@dataclass
class SimulationResult:
    data: pd.DataFrame
    true_graph: Any
    true_ordering: List[str]
    true_edges: List[Tuple[str, str]]
    config: DataGenerationConfig


@dataclass
class SymDAGResult:
    method: str
    graph: Any
    expressions: Dict[str, Any] = field(default_factory=dict)
    ordering: Optional[List[Any]] = None
    runtime: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        n_nodes = int(self.graph.number_of_nodes()) if hasattr(self.graph, "number_of_nodes") else None
        n_edges = int(self.graph.number_of_edges()) if hasattr(self.graph, "number_of_edges") else None
        is_dag = bool(nx.is_directed_acyclic_graph(self.graph)) if hasattr(self.graph, "edges") else None
        return {
            "method": self.method,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "is_dag": is_dag,
            "runtime": self.runtime,
            "ordering": self.ordering,
        }
