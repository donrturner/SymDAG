from __future__ import annotations

from typing import Any, Dict

import networkx as nx
import numpy as np

try:
    from gadjid import sid
except Exception:
    sid = None


def compute_metrics(est_graph: Any, true_graph: Any, d: int) -> Dict[str, Any]:
    nodelist = [f"X_{i}" for i in range(int(d))]
    true_adj = nx.to_numpy_array(true_graph, weight=None, nonedge=0, nodelist=nodelist, dtype=np.int8)
    est_adj = nx.to_numpy_array(est_graph, weight=None, nonedge=0, nodelist=nodelist, dtype=np.int8)
    shd = int(np.count_nonzero(true_adj != est_adj))
    sid_vals = sid(true_adj, est_adj, edge_direction="from row to column") if sid is not None else None
    return {"shd": shd, "sid": sid_vals}
