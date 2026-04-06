from __future__ import annotations

from itertools import combinations
from typing import Any, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import patsy
from pgmpy.base import DAG
from pgmpy.models import LinearGaussianBayesianNetwork
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler

from .config import DataGenerationConfig
from .results import SimulationResult


EQ_LIST = [
    "2 * np.cos(np.pi * gdata[parent]) * gdata[parent] - (0.25 * gdata[parent]) ** 2",
    "2 * np.cos(np.pi * gdata[parent])",
    "2 * np.sin(np.pi * gdata[parent])",
    "(0.25 * gdata[parent]) ** 2 + 2 * np.cos(np.pi * gdata[parent])",
    "2 * np.sin(np.pi * np.log(np.square(gdata[parent])))",
    "2 * np.sin(gdata[parent] ** 3)",
]

EQ_MULT = [
    "gdata[parent] * np.sin(np.pi * gdata[parent])",
    "(gdata[parent] ** 2) * np.cos(np.pi * gdata[parent])",
    "np.exp(0.5 * gdata[parent]) * np.sin(gdata[parent])",
    "np.tanh(gdata[parent]) * (gdata[parent] ** 3)",
    "(np.abs(gdata[parent]) + 0.1) * np.log1p(np.square(gdata[parent]))",
    "np.sin(np.square(gdata[parent])) * np.cos(gdata[parent])",
    "(0.5 + np.square(gdata[parent])) * np.sin(2 * np.pi * gdata[parent])",
    "np.cos(np.pi * gdata[parent]) * np.exp(-0.2 * np.square(gdata[parent]))",
]

EQ_PARENT_INPUT_CLIP = 8.0
EQ_TERM_OUTPUT_CLIP = 20.0
EQ_NODE_CLIP = 10.0
EQ_NODE_MIN_STD = 1e-8


def erdos_renyi_dag_from_order(d: int, edgeprob: float, seed: int | None = None):
    rng = np.random.default_rng(seed)
    order = rng.permutation(d).tolist()
    pos = {node: i for i, node in enumerate(order)}
    graph = nx.DiGraph()
    graph.add_nodes_from(range(d))
    for u in range(d):
        for v in range(d):
            if pos[u] > pos[v] and rng.random() < edgeprob:
                graph.add_edge(u, v)
    return graph, order


def gpgen(X, sigma: float = 1.0, jitter: float = 1e-6, seed: int | None = None):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if not np.isfinite(X).all():
        raise ValueError("gpgen received non-finite X")
    kernel = sigma * RBF(length_scale=1.0)
    K = kernel(X, X)
    K = 0.5 * (K + K.T)
    K = K + jitter * np.eye(K.shape[0])
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.zeros(X.shape[0]), K, size=1)


def _pairwise_product_features(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if p < 2:
        return np.empty((n, 0), dtype=float)
    feats = []
    for i in range(p):
        for j in range(i + 1, p):
            feats.append((X[:, i] * X[:, j]).reshape(-1, 1))
    return np.hstack(feats) if feats else np.empty((n, 0), dtype=float)


def gpgen_int(
    X,
    sigma: float = 1.0,
    length_scale: float = 1.0,
    interaction_length_scale: float = 1.0,
    interaction_weight: float = 1.0,
    jitter: float = 1e-6,
    seed: int | None = None,
):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if not np.isfinite(X).all():
        raise ValueError("gpgen_int received non-finite X")
    n, p = X.shape
    if p == 0:
        raise ValueError("gpgen_int requires at least one parent variable")
    if p == 1:
        return gpgen(X, sigma=sigma, jitter=jitter, seed=seed)

    w = float(max(0.0, interaction_weight))
    kernel_main = RBF(length_scale=length_scale)
    K_main = kernel_main(X, X)
    Z = _pairwise_product_features(X)
    if Z.shape[1] > 0 and w > 0.0:
        kernel_int = RBF(length_scale=interaction_length_scale)
        K_int = kernel_int(Z, Z)
        K = (K_main + w * K_int) / (1.0 + w)
    else:
        K = K_main

    K = sigma * K
    K = 0.5 * (K + K.T)
    K = K + jitter * np.eye(n)
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.zeros(n), K, size=1)


def splinegen(parent_data: pd.DataFrame, num_knots: int = 4) -> np.ndarray:
    n_samples, _ = parent_data.shape
    dependent_vector = np.zeros(n_samples)
    df = num_knots + 3
    for col in parent_data.columns:
        basis_matrix = patsy.dmatrix(
            f"bs(x, df={df}, include_intercept=True) - 1",
            {"x": parent_data[col]},
            return_type="dataframe",
        )
        coeffs = np.random.randn(basis_matrix.shape[1])
        dependent_vector += np.dot(basis_matrix, coeffs)
    return dependent_vector


def _safe_numeric_vector(values, clip=None):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if clip is None:
        cap = np.finfo(np.float64).max
    else:
        cap = float(abs(clip))
    arr = np.nan_to_num(arr, nan=0.0, posinf=cap, neginf=-cap)
    if clip is not None:
        arr = np.clip(arr, -cap, cap)
    return arr


class _SafeEqFrame:
    def __init__(self, frame, clip):
        self._frame = frame
        self._clip = float(clip)

    def __getitem__(self, key):
        return _safe_numeric_vector(self._frame[key].to_numpy(dtype=float), clip=self._clip)


def _stabilize_eq_node(values):
    arr = _safe_numeric_vector(values, clip=EQ_TERM_OUTPUT_CLIP)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if not np.isfinite(std) or std < EQ_NODE_MIN_STD:
        return _safe_numeric_vector(arr - mean, clip=EQ_NODE_CLIP)
    return _safe_numeric_vector((arr - mean) / std, clip=EQ_NODE_CLIP)


def _eval_eq_expression(eq_expr, gdata, parent):
    safe_gdata = _SafeEqFrame(gdata, clip=EQ_PARENT_INPUT_CLIP)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        base = eval(eq_expr, {"np": np}, {"gdata": safe_gdata, "parent": parent})
    return _safe_numeric_vector(base, clip=EQ_TERM_OUTPUT_CLIP)


def _eq_multi_parent_signal(gdata, parents, rng):
    n = len(gdata)
    parent_values = {
        parent: _safe_numeric_vector(gdata[parent].to_numpy(dtype=float), clip=EQ_PARENT_INPUT_CLIP)
        for parent in parents
    }
    main_effect = np.zeros(n, dtype=float)
    for parent in parents:
        eq_expr = str(rng.choice(EQ_MULT))
        main_effect += rng.choice([-1.0, 1.0]) * _eval_eq_expression(eq_expr, gdata, parent)
    main_effect = _safe_numeric_vector(main_effect, clip=EQ_TERM_OUTPUT_CLIP)
    main_effect /= np.sqrt(len(parents))
    main_effect = _safe_numeric_vector(main_effect, clip=EQ_TERM_OUTPUT_CLIP)

    pair_effect = np.zeros(n, dtype=float)
    pair_count = 0
    for pa, pb in combinations(parents, 2):
        prod = _safe_numeric_vector(parent_values[pa] * parent_values[pb], clip=EQ_TERM_OUTPUT_CLIP)
        interaction = prod if rng.random() < 0.5 else np.sin(np.pi * prod)
        interaction = _safe_numeric_vector(interaction, clip=EQ_TERM_OUTPUT_CLIP)
        pair_effect += rng.choice([-1.0, 1.0]) * interaction
        pair_effect = _safe_numeric_vector(pair_effect, clip=EQ_TERM_OUTPUT_CLIP)
        pair_count += 1
    if pair_count:
        pair_effect /= np.sqrt(pair_count)
        pair_effect = _safe_numeric_vector(pair_effect, clip=EQ_TERM_OUTPUT_CLIP)

    if len(parents) >= 3 and rng.random() < 0.5:
        chosen = rng.choice(parents, size=3, replace=False)
        triple = np.ones(n, dtype=float)
        for parent in chosen:
            triple = _safe_numeric_vector(triple * parent_values[parent], clip=EQ_TERM_OUTPUT_CLIP)
        pair_effect += 0.5 * rng.choice([-1.0, 1.0]) * triple
        pair_effect = _safe_numeric_vector(pair_effect, clip=EQ_TERM_OUTPUT_CLIP)

    return _safe_numeric_vector(main_effect + pair_effect, clip=EQ_TERM_OUTPUT_CLIP)


def generate_dataset(
    n: int,
    d: int,
    edgeprob: float,
    sigma: float,
    gen_method: str,
    seed: int,
) -> Tuple[pd.DataFrame, Any, List[str], List[Tuple[str, str]]]:
    rng = np.random.default_rng(seed)
    test_int, topo_order_int = erdos_renyi_dag_from_order(d, edgeprob, seed=seed)
    test = nx.relabel_nodes(test_int, {i: f"X_{i}" for i in range(d)})
    topo_order = [f"X_{i}" for i in topo_order_int]

    gmodel = LinearGaussianBayesianNetwork(DAG(test))
    gmodel.cpds = gmodel.get_random_cpds()

    gen_order = list(nx.topological_sort(gmodel))
    gdata = pd.DataFrame(index=range(n), columns=[f"X_{i}" for i in range(d)], dtype=float)
    edgesave: List[Tuple[str, str]] = []
    used_eq: List[str] = []

    for node in gen_order:
        parents = list(gmodel.get_parents(node))
        if not parents:
            gdata[node] = rng.uniform(low=-1.0, high=1.0, size=n)
            continue

        X_par = gdata[parents].to_numpy(dtype=float)
        if np.isnan(X_par).any():
            raise RuntimeError(f"NaN in parent inputs for node={node}, parents={parents}")

        if gen_method == "gp":
            gdata[node] = gpgen(X_par, sigma=1.0, jitter=1e-6, seed=seed).ravel() + sigma * rng.normal(size=n)
        elif gen_method == "gpi":
            gdata[node] = gpgen_int(X_par, sigma=1.0, length_scale=1.0, jitter=1e-6, seed=seed).ravel()
            gdata[node] = gdata[node] + sigma * rng.normal(size=n)
        elif gen_method == "spl":
            gdata[node] = splinegen(gdata[parents], num_knots=4).reshape(-1) + sigma * rng.normal(size=n)
        elif gen_method == "eq":
            if len(parents) > 1:
                signal = _eq_multi_parent_signal(gdata, parents, rng)
                node_values = signal + sigma * rng.normal(size=n)
            else:
                available = [eq for eq in EQ_LIST if eq not in used_eq]
                eq_expr = rng.choice(available if available else EQ_LIST)
                used_eq.append(eq_expr)
                parent = parents[0]
                base = _eval_eq_expression(eq_expr, gdata, parent)
                node_values = rng.choice([-1, 1]) * base + sigma * rng.normal(size=n)
            gdata[node] = _stabilize_eq_node(node_values)
        else:
            raise ValueError("Unsupported gen_method. Use 'gp', 'gpi', 'spl', or 'eq'.")

        edgesave += [(parent, node) for parent in parents]

    data_arr = gdata.to_numpy(dtype=float)
    if not np.isfinite(data_arr).all():
        for col in gdata.columns:
            gdata[col] = _safe_numeric_vector(gdata[col].to_numpy(dtype=float), clip=EQ_NODE_CLIP)

    scaler = StandardScaler()
    gdata = pd.DataFrame(scaler.fit_transform(gdata), columns=gdata.columns)
    return gdata, DAG(gmodel), topo_order, sorted(edgesave)


def simulate_dataset(config: DataGenerationConfig) -> SimulationResult:
    edgeprob = config.resolved_edgeprob()
    data, true_graph, true_ordering, true_edges = generate_dataset(
        n=config.n,
        d=config.d,
        edgeprob=edgeprob,
        sigma=config.sigma,
        gen_method=config.gen_method,
        seed=config.seed,
    )
    return SimulationResult(
        data=data,
        true_graph=true_graph,
        true_ordering=true_ordering,
        true_edges=true_edges,
        config=config,
    )
