
#!/usr/bin/env python3
import argparse
import os
os.environ["TQDM_DISABLE"] = "1"
import pickle
import re
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Optional

import dill
import networkx as nx
import numpy as np
import pandas as pd
import patsy
from joblib import Parallel, delayed
from pgmpy.base import DAG
from pgmpy.models import LinearGaussianBayesianNetwork
from pysips import PysipsRegressor
from scipy.special import softmax
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler

try:
    from gadjid import sid
except Exception:
    sid = None

try:
    import pybnesian as pbn
except Exception:
    pbn = None


# ----------------------------
# User config (edit in place)
# ----------------------------
GEN_METHOD = "gp"
N = 2000
D = 20
SIGMA = 0.1
SPARSITY = 1
EDGEPROB = (2 * SPARSITY) / (D - 1)
BASE_SEED = 42

N_ITER = 3000
BURNIN = 2000
N_CHAINS = 1
CHAIN_N_JOBS = None
EDGE_THRESHOLD = 0.3
SAMPLES_PER_FIT = 5
MAX_EQUATION_EVALS = None
FINAL_MAX_EQUATION_EVALS = 4000
SCORE_METHOD = "best"
SEL_TYPE = "max_nml"
USE_CHECKPOINTS = True
USE_BN_WARM_START = True
USE_DATA_DRIVEN_THRESHOLD = False
TARGET_N_EDGES = None
USE_EFDR_THRESHOLD = True
EFDR_Q = 0.05

COMP = 25
PARTS = 100
SAMP = 5
SR_OPERATORS = ["+", "-", "*", "cos"]

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


def _target_edge_count_from_density(p, edgeprob):
    total = (p * (p - 1)) // 2
    if edgeprob is None:
        return None
    try:
        raw = float(edgeprob) * float(total)
    except Exception:
        return None
    if not np.isfinite(raw):
        return None
    return int(np.clip(np.rint(raw), 0, total))


def _select_edges_by_efdr(scores, q):
    vals = np.asarray(scores, dtype=float)
    if vals.size == 0:
        return {
            "picked_indices": np.array([], dtype=int),
            "k_selected": 0,
            "efdr_at_selection": 0.0,
            "expected_false_edges": 0.0,
            "q_used": float(q),
        }
    vals = np.where(np.isfinite(vals), vals, 0.0)
    vals = np.clip(vals, 0.0, 1.0)

    try:
        q_used = float(np.clip(float(q), 0.0, 1.0))
    except Exception:
        q_used = 0.10
    order = np.argsort(vals)[::-1]
    sorted_vals = vals[order]
    cum_expected_false = np.cumsum(1.0 - sorted_vals)
    ks = np.arange(1, vals.size + 1, dtype=float)
    efdr_curve = cum_expected_false / ks

    valid = np.where(efdr_curve <= q_used)[0]
    if valid.size == 0:
        return {
            "picked_indices": np.array([], dtype=int),
            "k_selected": 0,
            "efdr_at_selection": 0.0,
            "expected_false_edges": 0.0,
            "q_used": q_used,
        }

    k_selected = int(valid[-1] + 1)
    picked = order[:k_selected]
    efdr_at_selection = float(efdr_curve[k_selected - 1])
    expected_false_edges = float(cum_expected_false[k_selected - 1])
    return {
        "picked_indices": picked,
        "k_selected": k_selected,
        "efdr_at_selection": efdr_at_selection,
        "expected_false_edges": expected_false_edges,
        "q_used": q_used,
    }


def log_normal_likelihood(residuals, sigma_sq):
    n = len(residuals)
    return -n / 2 * np.log(2 * np.pi) - n / 2 * np.log(sigma_sq) - np.sum(residuals**2) / (2 * sigma_sq)


def logmeanexp(log_values):
    log_values = np.asarray(log_values, dtype=float)
    if log_values.size == 0:
        return -np.inf
    max_log = np.max(log_values)
    return max_log + np.log(np.mean(np.exp(log_values - max_log)))


def score_from_reg(reg, score_method="best"):
    if score_method == "best":
        return float(reg.best_likelihood_)
    if score_method == "logmeanexp":
        return float(logmeanexp(reg.likelihoods_))
    raise ValueError(f"Unknown score_method: {score_method}")


def phi_last_value(phis):
    if phis is None:
        return None
    phis = np.asarray(phis)
    if phis.size == 0:
        return None
    if phis.size >= 2:
        return float(phis[-2])
    return float(phis[-1])


def get_parents(pi, i, p):
    return [] if i >= p - 1 else pi[i + 1 :]


def _checkpoint_attr(obj, name):
    if obj is None:
        return None
    if hasattr(obj, name):
        return getattr(obj, name)
    if hasattr(obj, "attrs") and isinstance(obj.attrs, dict):
        return obj.attrs.get(name)
    return None


def _to_numeric_vector(value):
    if value is None:
        return np.array([], dtype=float)
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.array([], dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    return arr


def _extract_phi_samples(obj):
    if obj is None:
        return 0
    for name in ("phis_", "phis"):
        if hasattr(obj, name):
            try:
                return len(getattr(obj, name))
            except Exception:
                pass
    if hasattr(obj, "attrs") and isinstance(obj.attrs, dict):
        for key in ("phis_", "phis"):
            val = obj.attrs.get(key)
            if val is not None:
                try:
                    return len(val)
                except Exception:
                    pass
    return 0


def reset_checkpoint_state(filename, phi_threshold=0.8, min_phi_samples=3):
    state = {
        "skip_fit": False,
        "phi": 0.0,
        "checkpoint_obj": None,
        "phi_samples": 0,
    }
    if not os.path.exists(filename):
        return state

    objects = []
    try:
        with open(filename, "rb") as f:
            while True:
                try:
                    objects.append(dill.load(f))
                except EOFError:
                    break
    except Exception:
        return state

    if not objects:
        return state

    phi_obj = objects[-2] if len(objects) >= 2 else objects[-1]
    checkpoint_obj = objects[-1]

    phi = 0.0
    if hasattr(phi_obj, "attrs") and isinstance(phi_obj.attrs, dict):
        v = phi_obj.attrs.get("phi")
        if v is not None:
            phi = float(v)
    elif hasattr(phi_obj, "phi"):
        phi = float(phi_obj.phi)

    phi_samples = _extract_phi_samples(phi_obj)
    state["phi"] = phi
    state["phi_samples"] = phi_samples
    state["checkpoint_obj"] = checkpoint_obj
    if phi >= phi_threshold and phi_samples >= min_phi_samples:
        state["skip_fit"] = True
    return state

def _candidate_expressions(obj):
    for name in ("expressions_", "expressions", "equations_", "equations", "models_", "models"):
        raw = _checkpoint_attr(obj, name)
        if raw is None:
            continue
        try:
            seq = list(raw)
        except Exception:
            continue
        if not seq:
            continue
        out = []
        for item in seq:
            if isinstance(item, (str, bytes)) or hasattr(item, "free_symbols"):
                out.append(item)
            else:
                expr = None
                for key in ("expression", "equation", "model", "expr"):
                    expr = _checkpoint_attr(item, key)
                    if expr is not None:
                        break
                out.append(expr)
        if any(v is not None for v in out):
            return out
    return []


def _candidate_likelihoods(obj):
    for name in ("likelihoods_", "likelihoods", "model_likelihoods_", "model_likelihoods", "scores_", "scores"):
        arr = _to_numeric_vector(_checkpoint_attr(obj, name))
        if arr.size:
            return arr.tolist()
    return []


def _extract_particle_weights(obj):
    for name in ("weights", "weights_", "log_weights", "log_weights_", "_log_weights"):
        arr = _to_numeric_vector(_checkpoint_attr(obj, name))
        if not arr.size:
            continue
        if "log" in name:
            m = np.max(arr)
            return np.exp(arr - m)
        return arr
    return np.array([], dtype=float)


def _normalize_probs(weights):
    w = _to_numeric_vector(weights)
    if w.size == 0:
        return None
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.maximum(w, 0.0)
    s = np.sum(w)
    if not np.isfinite(s) or s <= 0:
        return None
    return w / s


def _default_expression(obj):
    if hasattr(obj, "get_expression"):
        try:
            return obj.get_expression()
        except Exception:
            pass
    for key in ("best_expression", "expression", "equation"):
        v = _checkpoint_attr(obj, key)
        if v is not None:
            return v
    return None


def _default_likelihood(obj, score_method="best"):
    try:
        if hasattr(obj, "best_likelihood_") and score_method == "best":
            return float(obj.best_likelihood_)
        if hasattr(obj, "likelihoods_") and score_method == "logmeanexp":
            return float(logmeanexp(obj.likelihoods_))
    except Exception:
        pass
    for key in ("best_likelihood_", "best_likelihood", "likelihood"):
        v = _checkpoint_attr(obj, key)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            pass
    return None


def sample_weighted_model(obj, score_method="best", rng=None):
    if rng is None:
        rng = np.random.default_rng()

    exprs = _candidate_expressions(obj)
    likes = _candidate_likelihoods(obj)
    n = max(len(exprs), len(likes))
    if n <= 0:
        return {
            "index": None,
            "expression": _default_expression(obj),
            "likelihood": _default_likelihood(obj, score_method=score_method),
            "sampled": False,
            "probs": None,
        }

    if len(exprs) < n:
        exprs = exprs + [None] * (n - len(exprs))
    likes_arr = np.array([np.nan] * n, dtype=float)
    for i, v in enumerate(likes[:n]):
        try:
            likes_arr[i] = float(v)
        except Exception:
            pass

    probs = None
    w = _extract_particle_weights(obj)
    if w.size == n:
        probs = _normalize_probs(w)
    if probs is None and np.isfinite(likes_arr).any():
        m = np.nanmax(likes_arr)
        z = np.exp(np.where(np.isfinite(likes_arr), likes_arr - m, -np.inf))
        s = np.sum(z)
        if np.isfinite(s) and s > 0:
            probs = z / s
    if probs is None:
        probs = np.full(n, 1.0 / n, dtype=float)

    idx = int(rng.choice(np.arange(n), p=probs))
    expr = exprs[idx] if idx < len(exprs) else None
    lik = likes_arr[idx] if idx < len(likes_arr) and np.isfinite(likes_arr[idx]) else None

    if expr is None:
        expr = _default_expression(obj)
    if lik is None:
        lik = _default_likelihood(obj, score_method=score_method)

    return {
        "index": idx,
        "expression": expr,
        "likelihood": lik,
        "sampled": True,
        "probs": probs,
    }


def extract_active_parents(expression, parent_indices):
    if expression is None:
        return set()
    text = str(expression)
    active_local = set()
    for pattern in (r"X_\{?(\d+)\}?", r"\bX(\d+)\b", r"\bx_?(\d+)\b"):
        for token in re.findall(pattern, text):
            active_local.add(int(token))
    if hasattr(expression, "free_symbols"):
        for sym in expression.free_symbols:
            name = str(sym)
            for pattern in (r"X_\{?(\d+)\}?", r"\bX(\d+)\b", r"\bx_?(\d+)\b"):
                for token in re.findall(pattern, name):
                    active_local.add(int(token))
    return {parent_indices[i] for i in active_local if 0 <= i < len(parent_indices)}


def _checkpoint_summary(checkpoint_obj, X, Y, score_method="best"):
    if checkpoint_obj is None:
        return None
    sampled = sample_weighted_model(checkpoint_obj, score_method=score_method)
    expr = sampled["expression"]
    lik = sampled["likelihood"]

    residuals = None
    variance = None
    if hasattr(checkpoint_obj, "predict"):
        try:
            y_pred = checkpoint_obj.predict(X)
            residuals = Y.reshape(-1, 1) - y_pred.reshape(-1, 1)
            variance = np.var(residuals) or 1e-10
        except Exception:
            pass

    if lik is None and residuals is not None:
        lik = log_normal_likelihood(residuals, variance)
    if lik is None:
        return None
    if residuals is None:
        residuals = (Y - np.mean(Y)).reshape(-1, 1)
        variance = np.var(residuals) or 1e-10

    return {
        "expression": expr,
        "likelihood": float(lik),
        "residuals": residuals,
        "variance": variance,
        "sampled_model_index": sampled["index"],
    }

def symreg_fit(
    Y,
    X_parents,
    max_equation_evals=None,
    sel_type="max_nml",
    score_method="best",
    comp=None,
    parts=None,
    samp=None,
    samples_per_fit=1,
    max_time=None,
):
    if X_parents is None or (hasattr(X_parents, "shape") and X_parents.ndim == 2 and X_parents.shape[1] == 0):
        residuals = Y - np.mean(Y)
        variance = np.var(residuals) or 1e-10
        lik = log_normal_likelihood(residuals, variance)
        return {
            "residuals": residuals.reshape(-1, 1),
            "variance": variance,
            "expression": None,
            "regressor": None,
            "likelihood": lik,
            "phi_last": None,
            "sampled_model_index": None,
        }

    if comp is None:
        comp = COMP
    if parts is None:
        parts = PARTS
    if samp is None:
        samp = SAMP

    if max_time is not None and max_equation_evals is not None:
        raise ValueError("max_time and max_equation_evals cannot both be set")
    if max_equation_evals is None and max_time is None:
        max_equation_evals = max(1, int(parts) * int(samp) * int(samples_per_fit))

    X = X_parents.reshape(-1, 1) if X_parents.ndim == 1 else X_parents
    reg = PysipsRegressor(
        operators=list(SR_OPERATORS),
        max_complexity=comp,
        num_particles=parts,
        num_mcmc_samples=samp,
        max_time=max_time,
        max_equation_evals=max_equation_evals,
        random_state=42,
        model_selection=sel_type
    )
    reg.fit(X, Y)

    y_pred = reg.predict(X)
    residuals = Y.reshape(-1, 1) - y_pred.reshape(-1, 1)
    variance = np.var(residuals) or 1e-10
    sampled = sample_weighted_model(reg, score_method=score_method)
    expr = sampled["expression"]
    lik = sampled["likelihood"]
    if lik is None:
        lik = score_from_reg(reg, score_method=score_method)

    return {
        "residuals": residuals,
        "variance": variance,
        "expression": expr,
        "regressor": reg,
        "likelihood": float(lik),
        "phi_last": phi_last_value(getattr(reg, "phis_", None)),
        "sampled_model_index": sampled["index"],
    }


def _make_checkpoint_path(checkpoint_dir: Path, variable_indices: int, parent_indices):
    parents_str = "-".join(str(x) for x in sorted(parent_indices))
    return checkpoint_dir / f"{variable_indices}_{parents_str}.checkpoint"


def symreg_fit_with_parents(
    Y,
    Y_matrix,
    parent_indices,
    variable_indices=None,
    checkpoint_dir: Optional[Path] = None,
    max_equation_evals=None,
    sel_type="max_nml",
    score_method="best",
    comp=None,
    parts=None,
    samp=None,
    samples_per_fit=1,
    max_time=None,
):
    if not parent_indices:
        residuals = Y - np.mean(Y)
        variance = np.var(residuals) or 1e-10
        lik = log_normal_likelihood(residuals, variance)
        return {
            "residuals": residuals.reshape(-1, 1),
            "variance": variance,
            "expression": None,
            "likelihood": lik,
            "active_parents": set(),
            "phi_last": None,
            "regressor": None,
            "sampled_model_index": None,
        }

    if comp is None:
        comp = COMP
    if parts is None:
        parts = PARTS
    if samp is None:
        samp = SAMP

    if max_time is not None and max_equation_evals is not None:
        raise ValueError("max_time and max_equation_evals cannot both be set")
    if max_equation_evals is None and max_time is None:
        max_equation_evals = max(1, int(parts) * int(samp) * int(samples_per_fit))

    X_parents = Y_matrix[:, parent_indices]
    X = X_parents.reshape(-1, 1) if X_parents.ndim == 1 else X_parents

    ckpt_path = None
    checkpoint_state = None
    if checkpoint_dir is not None and variable_indices is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = _make_checkpoint_path(checkpoint_dir, int(variable_indices), parent_indices)
        checkpoint_state = reset_checkpoint_state(str(ckpt_path))

    if checkpoint_state and checkpoint_state.get("skip_fit"):
        cached = _checkpoint_summary(checkpoint_state.get("checkpoint_obj"), X, Y, score_method=score_method)
        if cached is not None:
            expr = cached["expression"]
            active_parents = extract_active_parents(expr, parent_indices)
            return {
                "residuals": cached["residuals"],
                "variance": cached["variance"],
                "expression": expr,
                "regressor": checkpoint_state.get("checkpoint_obj"),
                "likelihood": cached["likelihood"],
                "active_parents": active_parents,
                "phi_last": checkpoint_state.get("phi"),
                "sampled_model_index": cached.get("sampled_model_index"),
            }

    reg = PysipsRegressor(
        operators=list(SR_OPERATORS),
        max_complexity=comp,
        num_particles=parts,
        num_mcmc_samples=samp,
        checkpoint_file=str(ckpt_path) if ckpt_path is not None else None,
        max_time=max_time,
        max_equation_evals=max_equation_evals,
        random_state=42,
        model_selection=sel_type
    )
    reg.fit(X, Y)
    y_pred = reg.predict(X)
    residuals = Y.reshape(-1, 1) - y_pred.reshape(-1, 1)
    variance = np.var(residuals) or 1e-10

    sampled = sample_weighted_model(reg, score_method=score_method)
    expr = sampled["expression"]
    lik = sampled["likelihood"]
    if lik is None:
        lik = score_from_reg(reg, score_method=score_method)

    active_parents = extract_active_parents(expr, parent_indices)
    return {
        "residuals": residuals,
        "variance": variance,
        "expression": expr,
        "regressor": reg,
        "likelihood": float(lik),
        "active_parents": active_parents,
        "phi_last": phi_last_value(getattr(reg, "phis_", None)),
        "sampled_model_index": sampled["index"],
    }

def _clone_parent_map(parent_map):
    return {k: set(v) for k, v in parent_map.items()}


def _node_name_to_index(node, p):
    if isinstance(node, (int, np.integer)):
        idx = int(node)
        return idx if 0 <= idx < p else None
    s = str(node)
    m = re.search(r"(\d+)$", s)
    if m is None:
        return None
    idx = int(m.group(1))
    return idx if 0 <= idx < p else None


def _extract_model_edges(model):
    for attr in ("arcs", "edges"):
        if not hasattr(model, attr):
            continue
        raw = getattr(model, attr)
        try:
            items = list(raw() if callable(raw) else raw)
        except Exception:
            continue
        if items:
            return items

    if hasattr(model, "to_networkx"):
        try:
            g = model.to_networkx()
            return list(g.edges())
        except Exception:
            pass
    return []


def _pybnesian_warm_start_order(data, seed):
    p = data.shape[1]
    if pbn is None:
        return None, {"used": False, "reason": "pybnesian_not_installed"}

    cols = [f"X_{i}" for i in range(p)]
    df = pd.DataFrame(np.asarray(data), columns=cols)
    attempts = []
    learner = None

    learner_candidates = []
    for name in ("GreedyHillClimbing", "HillClimbing"):
        ctor = getattr(pbn, name, None)
        if ctor is not None:
            learner_candidates.append((name, ctor))

    if not learner_candidates:
        return None, {"used": False, "reason": "no_hill_climbing_api_in_pybnesian"}

    for learner_name, ctor in learner_candidates:
        try:
            learner = ctor()
            break
        except Exception as exc:
            attempts.append(f"{learner_name}() failed: {exc}")
            learner = None
    if learner is None:
        return None, {"used": False, "reason": "learner_init_failed", "attempts": attempts}

    start = None
    start_builders = []
    for name in ("GaussianNetwork", "BayesianNetwork", "SemiparametricBN"):
        ctor = getattr(pbn, name, None)
        if ctor is not None:
            start_builders.append((f"{name}(cols)", lambda ctor=ctor: ctor(cols)))
    for label, fn in start_builders:
        try:
            start = fn()
            if start is not None:
                break
        except Exception as exc:
            attempts.append(f"{label} failed: {exc}")
    if start is None:
        return None, {"used": False, "reason": "start_network_init_failed", "attempts": attempts}

    score = None
    score_builders = []
    for name in ("BIC", "BGe", "CVLikelihood", "ValidatedLikelihood"):
        ctor = getattr(pbn, name, None)
        if ctor is not None:
            score_builders.append((f"{name}(df)", lambda ctor=ctor: ctor(df)))
    for label, fn in score_builders:
        try:
            score = fn()
            if score is not None:
                break
        except Exception as exc:
            attempts.append(f"{label} failed: {exc}")
    if score is None:
        return None, {"used": False, "reason": "score_init_failed", "attempts": attempts}

    operators = None
    arc_ops_ctor = getattr(pbn, "ArcOperatorSet", None)
    if arc_ops_ctor is not None:
        try:
            operators = arc_ops_ctor()
        except Exception as exc:
            attempts.append(f"ArcOperatorSet() failed: {exc}")

    if operators is None:
        pool_ctor = getattr(pbn, "OperatorPool", None)
        add_ctor = getattr(pbn, "AddArc", None)
        rem_ctor = getattr(pbn, "RemoveArc", None)
        flip_ctor = getattr(pbn, "FlipArc", None)
        if pool_ctor is not None and add_ctor is not None and rem_ctor is not None and flip_ctor is not None:
            try:
                operators = pool_ctor([add_ctor(), rem_ctor(), flip_ctor()])
            except Exception as exc:
                attempts.append(f"OperatorPool([AddArc,RemoveArc,FlipArc]) failed: {exc}")
    if operators is None:
        return None, {"used": False, "reason": "operators_init_failed", "attempts": attempts}

    calls = []
    if hasattr(learner, "estimate"):
        calls.append(("estimate(operators, score, start)", lambda: learner.estimate(operators, score, start)))
        calls.append(
            (
                "estimate(operators, score, start, max_iters=10000)",
                lambda: learner.estimate(operators, score, start, max_iters=10000),
            )
        )
        # Legacy API fallback (older pybnesian variants).
        calls.append(("estimate(df) [legacy]", lambda: learner.estimate(df)))
    if hasattr(learner, "learn"):
        calls.append(("learn(operators, score, start)", lambda: learner.learn(operators, score, start)))
        calls.append(("learn(df) [legacy]", lambda: learner.learn(df)))

    model = None
    for label, fn in calls:
        try:
            model = fn()
            if model is not None:
                break
        except Exception as exc:
            attempts.append(f"{label} failed: {exc}")
    if model is None:
        return None, {"used": False, "reason": "bn_fit_failed", "attempts": attempts}

    raw_edges = _extract_model_edges(model)
    g = nx.DiGraph()
    g.add_nodes_from(range(p))
    for edge in raw_edges:
        if not isinstance(edge, (tuple, list)) or len(edge) != 2:
            continue
        u_raw, v_raw = edge
        u = _node_name_to_index(u_raw, p)
        v = _node_name_to_index(v_raw, p)
        if u is None or v is None or u == v:
            continue
        g.add_edge(u, v)

    if not nx.is_directed_acyclic_graph(g):
        return None, {"used": False, "reason": "bn_output_not_dag"}

    topo = list(nx.topological_sort(g))
    if len(topo) != p:
        return None, {"used": False, "reason": "topological_sort_failed"}

    # Internal pi expects potential parents later in the list, so reverse topo order.
    pi = list(reversed(topo))
    return pi, {"used": True, "reason": "pybnesian_hillclimb", "n_edges": int(g.number_of_edges())}


def _initial_ordering(data, seed, use_bn_warm_start):
    p = data.shape[1]
    rng = np.random.default_rng(seed)
    if not use_bn_warm_start:
        return list(rng.permutation(p)), {"used": False, "reason": "disabled"}

    pi, info = _pybnesian_warm_start_order(data, seed)
    if pi is not None and len(pi) == p:
        return pi, info

    fallback = list(rng.permutation(p))
    info = dict(info or {})
    info["fallback"] = "random_permutation"
    return fallback, info


def _fit_node_with_order(
    data,
    var,
    parents,
    checkpoint_dir,
    max_equation_evals,
    sel_type,
    score_method,
    samples_per_fit,
):
    fit = symreg_fit_with_parents(
        data[:, var],
        data,
        sorted(parents),
        var,
        checkpoint_dir=checkpoint_dir,
        max_equation_evals=max_equation_evals,
        sel_type=sel_type,
        score_method=score_method,
        samples_per_fit=samples_per_fit,
    )
    fit["_parent_indices"] = tuple(sorted(parents))
    return fit


def _rb_parent_summary_from_fit(fit, parent_indices, score_method="best"):
    default_like = float(fit.get("likelihood", -np.inf))
    active_fallback = {int(p): 1.0 for p in fit.get("active_parents", set())}
    reg = fit.get("regressor")

    # Parent-index mismatch means fit came from another structural state.
    fit_parent_indices = tuple(fit.get("_parent_indices", tuple(parent_indices)))
    if tuple(parent_indices) != fit_parent_indices:
        return {"rb_likelihood": default_like, "parent_probs": active_fallback, "rb_mode": "fallback_parent_mismatch"}

    if reg is None:
        return {"rb_likelihood": default_like, "parent_probs": active_fallback, "rb_mode": "no_regressor"}

    exprs = _candidate_expressions(reg)
    likes_raw = _candidate_likelihoods(reg)
    likes = np.asarray(likes_raw, dtype=float) if len(likes_raw) else np.array([], dtype=float)
    weights = _extract_particle_weights(reg)

    can_rb_structure = len(exprs) > 0 and weights.size == len(exprs)
    can_rb_like = likes.size > 0 and weights.size == likes.size

    rb_like = default_like
    if can_rb_like:
        probs_like = _normalize_probs(weights)
        if probs_like.size == likes.size and np.isfinite(likes).any():
            mask = np.isfinite(likes)
            p_valid = probs_like[mask]
            z = np.sum(p_valid)
            if np.isfinite(z) and z > 0:
                rb_like = float(np.dot(p_valid / z, likes[mask]))

    if not can_rb_structure:
        return {"rb_likelihood": rb_like, "parent_probs": active_fallback, "rb_mode": "fallback_no_structure_alignment"}

    probs = _normalize_probs(weights)
    if probs.size != len(exprs):
        return {"rb_likelihood": rb_like, "parent_probs": active_fallback, "rb_mode": "fallback_prob_alignment"}

    parent_probs = defaultdict(float)
    parsed_any = False
    for expr, prob in zip(exprs, probs):
        if not np.isfinite(prob) or prob <= 0:
            continue
        if expr is None:
            continue
        parsed_any = True
        active = extract_active_parents(expr, parent_indices)
        for parent in active:
            parent_probs[int(parent)] += float(prob)

    if not parsed_any:
        return {"rb_likelihood": rb_like, "parent_probs": active_fallback, "rb_mode": "fallback_unparsed"}

    # Keep probabilities in [0,1] for numerical safety.
    clipped = {k: float(np.clip(v, 0.0, 1.0)) for k, v in parent_probs.items() if np.isfinite(v) and v > 0}
    if not clipped:
        return {"rb_likelihood": rb_like, "parent_probs": {}, "rb_mode": "rb_struct_zero"}
    return {"rb_likelihood": rb_like, "parent_probs": clipped, "rb_mode": "rb_struct"}


def _edge_expectation_from_state(pi, current_fits, p, score_method="best"):
    edge_expect = np.zeros((p, p), dtype=np.float64)
    rb_mode_counts = defaultdict(int)
    for idx, child in enumerate(pi):
        parent_indices = get_parents(pi, idx, p)
        fit = current_fits.get(child)
        if fit is None:
            continue
        rb = _rb_parent_summary_from_fit(fit, parent_indices, score_method=score_method)
        rb_mode_counts[rb.get("rb_mode", "unknown")] += 1

        valid_ancestors = set(parent_indices)
        for parent, prob in rb["parent_probs"].items():
            if parent in valid_ancestors:
                edge_expect[parent, child] += float(prob)
    return edge_expect, dict(rb_mode_counts)


def _initialize_chain_state(data, pi, checkpoint_dir, max_equation_evals, sel_type, score_method, samples_per_fit):
    p = len(pi)
    jobs = [(var, get_parents(pi, idx, p)) for idx, var in enumerate(pi)]
    fits = Parallel(n_jobs=min(max(1, p), 4))(
        delayed(_fit_node_with_order)(
            data,
            var,
            parents,
            checkpoint_dir,
            max_equation_evals,
            sel_type,
            score_method,
            samples_per_fit,
        )
        for var, parents in jobs
    )

    current_active_parents = {}
    current_likelihood = {}
    current_fits = {}
    total_likelihood = 0.0
    for (var, _), fit in zip(jobs, fits):
        current_active_parents[var] = fit["active_parents"]
        current_likelihood[var] = fit["likelihood"]
        current_fits[var] = fit
        total_likelihood += fit["likelihood"]
    return current_active_parents, current_likelihood, current_fits, float(total_likelihood)


def symdag_mcmc_step(
    data,
    pi,
    current_active_parents,
    current_likelihood,
    current_fits,
    total_likelihood,
    rng,
    checkpoint_dir,
    max_equation_evals,
    sel_type,
    score_method,
    samples_per_fit,
):
    p = len(pi)
    i = int(rng.integers(0, p - 1))
    var_i, var_j = pi[i], pi[i + 1]
    old_li = float(current_likelihood[var_i])
    old_lj = float(current_likelihood[var_j])

    parents_i = get_parents(pi, i, p)
    parents_j = get_parents(pi, i + 1, p)
    parents_j_swap = [var_i] + parents_j

    fits = Parallel(n_jobs=4)(
        delayed(_fit_node_with_order)(
            data,
            var,
            parents,
            checkpoint_dir,
            max_equation_evals,
            sel_type,
            score_method,
            samples_per_fit,
        )
        for var, parents in ([var_i, parents_i], [var_j, parents_j], [var_j, parents_j_swap], [var_i, parents_j])
    )

    log_p_stay = fits[0]["likelihood"] + fits[1]["likelihood"]
    log_p_swap = fits[2]["likelihood"] + fits[3]["likelihood"]
    probs = softmax([log_p_stay, log_p_swap])
    accept_swap = bool(rng.random() < probs[1])

    if accept_swap:
        pi[i], pi[i + 1] = pi[i + 1], pi[i]
        new_li = float(fits[3]["likelihood"])
        new_lj = float(fits[2]["likelihood"])
        new_pi = set(fits[3]["active_parents"])
        new_pj = set(fits[2]["active_parents"])
    else:
        new_li = float(fits[0]["likelihood"])
        new_lj = float(fits[1]["likelihood"])
        new_pi = set(fits[0]["active_parents"])
        new_pj = set(fits[1]["active_parents"])

    current_active_parents[var_i] = new_pi
    current_active_parents[var_j] = new_pj
    current_likelihood[var_i] = new_li
    current_likelihood[var_j] = new_lj
    if accept_swap:
        current_fits[var_i] = fits[3]
        current_fits[var_j] = fits[2]
    else:
        current_fits[var_i] = fits[0]
        current_fits[var_j] = fits[1]
    total_likelihood += new_li + new_lj - old_li - old_lj
    return float(total_likelihood)


def _adjacency_from_state(pi, active_parents, p):
    adj = np.zeros((p, p), dtype=np.int32)
    for idx, child in enumerate(pi):
        valid_ancestors = set(pi[idx + 1 :])
        parents = set(active_parents.get(child, set())).intersection(valid_ancestors)
        for parent in parents:
            adj[parent, child] = 1
    return adj


def _run_symdag_chain(
    data,
    n_iter,
    burnin,
    checkpoint_dir,
    max_equation_evals,
    score_method,
    sel_type,
    samples_per_fit,
    seed,
    use_bn_warm_start=False,
):
    p = data.shape[1]
    rng = np.random.default_rng(seed)
    pi, warm_start_info = _initial_ordering(data, seed, use_bn_warm_start=use_bn_warm_start)
    initial_ordering = list(pi)
    current_active_parents, current_likelihood, current_fits, total_likelihood = _initialize_chain_state(
        data, pi, checkpoint_dir, max_equation_evals, sel_type, score_method, samples_per_fit
    )

    ordering_counts = defaultdict(int)
    ordering_best_score = defaultdict(lambda: -np.inf)
    best_ordering = tuple(pi)
    best_score = float(total_likelihood)
    best_active_parents = _clone_parent_map(current_active_parents)

    edge_weight_sums = np.zeros((p, p), dtype=np.float64)
    position_sums = np.zeros(p, dtype=np.float64)
    sample_count = 0
    rb_mode_counts = defaultdict(int)

    for iteration in range(n_iter):
        total_likelihood = symdag_mcmc_step(
            data,
            pi,
            current_active_parents,
            current_likelihood,
            current_fits,
            total_likelihood,
            rng,
            checkpoint_dir,
            max_equation_evals,
            sel_type,
            score_method,
            samples_per_fit,
        )
        if iteration >= burnin:
            key = tuple(pi)
            ordering_counts[key] += 1
            ordering_best_score[key] = max(ordering_best_score[key], total_likelihood)
            if total_likelihood > best_score:
                best_score = float(total_likelihood)
                best_ordering = key
                best_active_parents = _clone_parent_map(current_active_parents)
            edge_expect, rb_modes_iter = _edge_expectation_from_state(pi, current_fits, p, score_method=score_method)
            edge_weight_sums += edge_expect
            for k, v in rb_modes_iter.items():
                rb_mode_counts[k] += int(v)
            for pos, node in enumerate(pi):
                position_sums[node] += pos
            sample_count += 1

    return {
        "best_ordering": list(best_ordering),
        "best_score": float(best_score),
        "best_active_parents": best_active_parents,
        "ordering_counts": dict(ordering_counts),
        "ordering_best_score": dict(ordering_best_score),
        "edge_weight_sums": edge_weight_sums,
        "position_sums": position_sums,
        "sample_count": int(sample_count),
        "initial_ordering": initial_ordering,
        "warm_start_info": warm_start_info,
        "rb_mode_counts": dict(rb_mode_counts),
    }

def _build_graph_from_posterior(
    data,
    edge_posterior,
    mean_positions,
    edge_threshold,
    use_efdr_threshold,
    efdr_q,
    use_data_driven_threshold,
    target_n_edges,
    final_max_equation_evals,
    sel_type,
    score_method,
    samples_per_fit,
):
    p = data.shape[1]
    G = DAG()
    G.add_nodes_from([f"X_{i}" for i in range(p)])
    expressions = {}
    consensus_ordering = list(np.argsort(mean_positions))
    candidate_edges = []
    candidate_scores = []

    for u in range(p):
        for v in range(p):
            if u == v:
                continue
            if mean_positions[u] <= mean_positions[v]:
                continue
            candidate_edges.append((u, v))
            candidate_scores.append(float(edge_posterior[u, v]))

    selected_edges = set()
    try:
        effective_threshold = float(edge_threshold)
    except Exception:
        effective_threshold = float("-inf")
    selection_mode = "fixed_threshold"
    efdr_q_used = None
    efdr_at_selection = None
    expected_false_edges = None

    if use_efdr_threshold:
        selection_mode = "efdr"
        efdr_pick = _select_edges_by_efdr(candidate_scores, efdr_q)
        picked = efdr_pick["picked_indices"]
        if picked.size > 0:
            selected_edges = {candidate_edges[int(i)] for i in picked.tolist()}
            effective_threshold = float(min(candidate_scores[int(i)] for i in picked.tolist()))
        else:
            selected_edges = set()
            effective_threshold = np.inf
        target_n_edges = int(efdr_pick["k_selected"])
        efdr_q_used = float(efdr_pick["q_used"])
        efdr_at_selection = float(efdr_pick["efdr_at_selection"])
        expected_false_edges = float(efdr_pick["expected_false_edges"])
    elif use_data_driven_threshold:
        selection_mode = "target_edge_count"
        total_candidates = len(candidate_edges)
        if target_n_edges is None:
            target_n_edges = 0
        k = int(np.clip(target_n_edges, 0, total_candidates))
        if k > 0 and total_candidates > 0:
            order = np.argsort(np.asarray(candidate_scores))[::-1]
            picked = order[:k]
            selected_edges = {candidate_edges[i] for i in picked}
            effective_threshold = float(min(candidate_scores[i] for i in picked))
        else:
            selected_edges = set()
            effective_threshold = np.inf

    if selection_mode == "fixed_threshold":
        for u, v in candidate_edges:
            if edge_posterior[u, v] >= edge_threshold:
                selected_edges.add((u, v))

    for u, v in sorted(selected_edges):
        G.add_edge(f"X_{u}", f"X_{v}")

    for var in consensus_ordering:
        node = f"X_{var}"
        parents_list = list(G.get_parents(node))
        if not parents_list:
            expressions[node] = None
        else:
            par_idx = [int(pn.split("_")[1]) for pn in parents_list]
            fit = symreg_fit(
                data[:, var],
                data[:, par_idx],
                max_equation_evals=final_max_equation_evals,
                sel_type=sel_type,
                score_method=score_method,
                samples_per_fit=samples_per_fit,
            )
            expressions[node] = fit["expression"]

    return G, expressions, consensus_ordering, {
        "mode": selection_mode,
        "target_n_edges": int(target_n_edges) if target_n_edges is not None else None,
        "n_selected_edges": int(len(selected_edges)),
        "effective_edge_threshold": float(effective_threshold),
        "efdr_q": efdr_q_used,
        "efdr_at_selection": efdr_at_selection,
        "expected_false_edges": expected_false_edges,
    }


def symdag_stochastic(
    Y,
    n_iter=1000,
    random_state=None,
    burnin=500,
    checkpoint_dir: Optional[Path] = None,
    max_equation_evals=None,
    final_max_equation_evals=3000,
    score_method="best",
    sel_type="max_nml",
    samples_per_fit=1,
    n_chains=1,
    chain_n_jobs=None,
    edge_threshold=0.5,
    use_efdr_threshold=False,
    efdr_q=0.10,
    use_data_driven_threshold=False,
    target_n_edges=None,
    edgeprob=None,
    use_bn_warm_start=False,
):
    start = time.perf_counter()
    data = Y.values if isinstance(Y, pd.DataFrame) else Y
    p = data.shape[1]

    base_seed = int(np.random.SeedSequence().entropy) if random_state is None else int(random_state)
    if n_chains < 1:
        raise ValueError("n_chains must be >= 1")
    child_seeds = [base_seed + 10007 * i for i in range(n_chains)]

    if n_chains == 1:
        chain_results = [
            _run_symdag_chain(
                data,
                n_iter,
                burnin,
                checkpoint_dir,
                max_equation_evals,
                score_method,
                sel_type,
                samples_per_fit,
                child_seeds[0],
                use_bn_warm_start,
            )
        ]
    else:
        jobs = n_chains if chain_n_jobs is None else chain_n_jobs
        chain_results = Parallel(n_jobs=jobs)(
            delayed(_run_symdag_chain)(
                data,
                n_iter,
                burnin,
                checkpoint_dir,
                max_equation_evals,
                score_method,
                sel_type,
                samples_per_fit,
                seed,
                use_bn_warm_start,
            )
            for seed in child_seeds
        )

    total_samples = int(sum(res["sample_count"] for res in chain_results))
    if total_samples <= 0:
        raise ValueError("No post-burnin samples collected; increase n_iter or reduce burnin.")

    edge_weight_sums = np.zeros((p, p), dtype=np.float64)
    position_sums = np.zeros(p, dtype=np.float64)
    rb_mode_counts = defaultdict(int)
    for res in chain_results:
        edge_weight_sums += res["edge_weight_sums"]
        position_sums += res["position_sums"]
        for k, v in res.get("rb_mode_counts", {}).items():
            rb_mode_counts[k] += int(v)

    # Output-stage Rao-Blackwellization: average expected edge indicators across samples.
    edge_posterior = edge_weight_sums / float(total_samples)
    mean_positions = position_sums / float(total_samples)

    if target_n_edges is None:
        target_n_edges = _target_edge_count_from_density(p, edgeprob)

    G, expressions, consensus_ordering, edge_selection = _build_graph_from_posterior(
        data,
        edge_posterior,
        mean_positions,
        edge_threshold,
        use_efdr_threshold,
        efdr_q,
        use_data_driven_threshold,
        target_n_edges,
        final_max_equation_evals,
        sel_type,
        score_method,
        samples_per_fit,
    )

    best_chain_idx = int(np.argmax([res["best_score"] for res in chain_results]))
    total_time = time.perf_counter() - start
    try:
        edge_threshold_requested = float(edge_threshold)
    except Exception:
        edge_threshold_requested = None
    return {
        "graph": G,
        "ordering": consensus_ordering,
        "ordering_score": float(chain_results[best_chain_idx]["best_score"]),
        "best_chain_index": best_chain_idx,
        "expressions": expressions,
        "ordering_counts": chain_results[best_chain_idx]["ordering_counts"],
        "ordering_best_score": chain_results[best_chain_idx]["ordering_best_score"],
        "chain_best_scores": [float(res["best_score"]) for res in chain_results],
        "chain_best_orderings": [res["best_ordering"] for res in chain_results],
        "chain_initial_orderings": [res.get("initial_ordering") for res in chain_results],
        "chain_warm_start_info": [res.get("warm_start_info") for res in chain_results],
        "chain_rb_mode_counts": [res.get("rb_mode_counts", {}) for res in chain_results],
        "rb_mode_counts": dict(rb_mode_counts),
        "edge_aggregation": "output_stage_rb",
        "edge_posterior": edge_posterior,
        "mean_positions": mean_positions,
        "edge_threshold": float(edge_selection["effective_edge_threshold"]),
        "edge_threshold_requested": edge_threshold_requested,
        "edge_selection_mode": edge_selection["mode"],
        "target_n_edges": edge_selection["target_n_edges"],
        "n_selected_edges": edge_selection["n_selected_edges"],
        "efdr_q": edge_selection["efdr_q"],
        "efdr_at_selection": edge_selection["efdr_at_selection"],
        "expected_false_edges": edge_selection["expected_false_edges"],
        "samples_collected": total_samples,
        "time": total_time,
    }

def erdos_renyi_dag_from_order(d, edgeprob, seed=None):
    rng = np.random.default_rng(seed)
    order = rng.permutation(d).tolist()
    pos = {node: i for i, node in enumerate(order)}
    G = nx.DiGraph()
    G.add_nodes_from(range(d))
    for u in range(d):
        for v in range(d):
            if pos[u] > pos[v] and rng.random() < edgeprob:
                G.add_edge(u, v)
    return G, order


def gpgen(X, sigma=1.0, jitter=1e-6, seed=None):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if not np.isfinite(X).all():
        raise ValueError("gpgen received non-finite X")
    kernel = sigma * RBF(length_scale=1.0)
    K = kernel(X, X)
    K = 0.5 * (K + K.T)
    K = K + jitter * np.eye(K.shape[0])
    mean = np.zeros(X.shape[0])
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, K, size=1)


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
    sigma=1.0,
    length_scale=1.0,
    interaction_length_scale=1.0,
    interaction_weight=1.0,
    jitter=1e-6,
    seed=None,
):
    """
    GP with explicit interaction structure for multi-parent nodes.
    For p>=2 parents, sample from a covariance that mixes:
      - main-effect kernel on raw parent inputs
      - interaction kernel on pairwise product features
    The weighted sum of PSD kernels remains PSD.
    """
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

    mean = np.zeros(n)
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean, K, size=1)


def splinegen(parent_data: pd.DataFrame, num_knots=4) -> np.ndarray:
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
        parent_contribution = np.dot(basis_matrix, coeffs)
        dependent_vector += parent_contribution
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
        centered = arr - mean
        return _safe_numeric_vector(centered, clip=EQ_NODE_CLIP)
    z = (arr - mean) / std
    return _safe_numeric_vector(z, clip=EQ_NODE_CLIP)


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

    # Preserve nonlinear single-parent structure, then add explicit interactions.
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
        xa = parent_values[pa]
        xb = parent_values[pb]
        prod = _safe_numeric_vector(xa * xb, clip=EQ_TERM_OUTPUT_CLIP)
        if rng.random() < 0.5:
            interaction = prod
        else:
            interaction = np.sin(np.pi * prod)
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


def generate_dataset(n, d, edgeprob, sigma, gen_method, seed):
    rng = np.random.default_rng(seed)
    test_int, topo_order_int = erdos_renyi_dag_from_order(d, edgeprob, seed=seed)
    test = nx.relabel_nodes(test_int, {i: f"X_{i}" for i in range(d)})
    topo_order = [f"X_{i}" for i in topo_order_int]

    gmodel = LinearGaussianBayesianNetwork(DAG(test))
    gmodel.cpds = gmodel.get_random_cpds()

    gen_order = list(nx.topological_sort(gmodel))
    gdata = pd.DataFrame(index=range(n), columns=[f"X_{i}" for i in range(d)], dtype=float)
    edgesave = []
    used_eq = []

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
            gdata[node] = gpgen_int(X_par, sigma=1.0, length_scale=1.0, jitter=1e-6, seed=seed).ravel() + sigma * rng.normal(size=n)
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
            raise ValueError(f"Unsupported gen_method: {gen_method}. Use 'gp', 'gpi', 'spl', or 'eq'.")

        edgesave += [(parent, node) for parent in parents]

    data_arr = gdata.to_numpy(dtype=float)
    if not np.isfinite(data_arr).all():
        for col in gdata.columns:
            gdata[col] = _safe_numeric_vector(gdata[col].to_numpy(dtype=float), clip=EQ_NODE_CLIP)

    scaler = StandardScaler()
    gdata = pd.DataFrame(scaler.fit_transform(gdata), columns=gdata.columns)
    return gdata, DAG(gmodel), topo_order, sorted(edgesave)


def compute_metrics(est_graph, true_graph, d):
    nodelist = [f"X_{i}" for i in range(d)]
    true_adj = nx.to_numpy_array(true_graph, weight=None, nonedge=0, nodelist=nodelist, dtype=np.int8)
    est_adj = nx.to_numpy_array(est_graph, weight=None, nonedge=0, nodelist=nodelist, dtype=np.int8)
    shd = int(np.count_nonzero(true_adj != est_adj))
    sid_vals = sid(true_adj, est_adj, edge_direction="from row to column") if sid is not None else None
    return shd, sid_vals


def make_checkpoint_dir(gen_method, n, d, sigma, array_id):
    method_tag = f"{gen_method}_ER{int(SPARSITY)}"
    return (
        Path("checkpoints")
        / method_tag
        / f"n{int(n)}_d{int(d)}_sigma{float(sigma):g}"
        / str(array_id)
    )


def make_output_path(gen_method, n, d, sigma, array_id):
    method_tag = f"{gen_method}_ER{int(SPARSITY)}"
    outdir = Path("output") / method_tag / f"n{int(n)}_d{int(d)}_sigma{float(sigma):g}_rb"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / f"{array_id}.pickle"


def parse_args():
    parser = argparse.ArgumentParser(description="Run SymDAG sampler experiment.")
    parser.add_argument("--array_id", default=os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    return parser.parse_args()


def main():
    args = parse_args()
    array_id = str(args.array_id)
    seed = BASE_SEED + int(array_id)

    ckpt_dir = make_checkpoint_dir(GEN_METHOD, N, D, SIGMA, array_id)
    if USE_CHECKPOINTS:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    else:
        ckpt_dir = None

    t0 = time.perf_counter()
    data, true_graph, true_ordering, true_edges = generate_dataset(
        n=N,
        d=D,
        edgeprob=EDGEPROB,
        sigma=SIGMA,
        gen_method=GEN_METHOD,
        seed=seed,
    )

    result = symdag_stochastic(
        data,
        n_iter=N_ITER,
        random_state=seed,
        burnin=BURNIN,
        checkpoint_dir=ckpt_dir,
        max_equation_evals=MAX_EQUATION_EVALS,
        final_max_equation_evals=FINAL_MAX_EQUATION_EVALS,
        score_method=SCORE_METHOD,
        sel_type=SEL_TYPE,
        samples_per_fit=SAMPLES_PER_FIT,
        n_chains=N_CHAINS,
        chain_n_jobs=CHAIN_N_JOBS,
        edge_threshold=EDGE_THRESHOLD,
        use_efdr_threshold=USE_EFDR_THRESHOLD,
        efdr_q=EFDR_Q,
        use_data_driven_threshold=USE_DATA_DRIVEN_THRESHOLD,
        target_n_edges=TARGET_N_EDGES,
        edgeprob=EDGEPROB,
        use_bn_warm_start=USE_BN_WARM_START,
    )

    est_graph = result["graph"]
    est_edges = sorted(list(est_graph.edges()))
    shd, sid_vals = compute_metrics(est_graph, true_graph, D)
    elapsed = time.perf_counter() - t0

    payload = {
        "array_id": array_id,
        "seed": seed,
        "config": {
            "gen_method": GEN_METHOD,
            "n": N,
            "d": D,
            "sigma": SIGMA,
            "sparsity": SPARSITY,
            "edgeprob": EDGEPROB,
            "n_iter": N_ITER,
            "burnin": BURNIN,
            "n_chains": N_CHAINS,
            "chain_n_jobs": CHAIN_N_JOBS,
            "edge_threshold": EDGE_THRESHOLD,
            "use_efdr_threshold": USE_EFDR_THRESHOLD,
            "efdr_q": EFDR_Q,
            "use_data_driven_threshold": USE_DATA_DRIVEN_THRESHOLD,
            "target_n_edges": TARGET_N_EDGES,
            "samples_per_fit": SAMPLES_PER_FIT,
            "max_equation_evals": MAX_EQUATION_EVALS,
            "final_max_equation_evals": FINAL_MAX_EQUATION_EVALS,
            "score_method": SCORE_METHOD,
            "sel_type": SEL_TYPE,
            "use_checkpoints": USE_CHECKPOINTS,
            "use_bn_warm_start": USE_BN_WARM_START,
        },
        "true_ordering": true_ordering,
        "estimated_ordering": list(result["ordering"]),
        "true_edges": true_edges,
        "estimated_edges": est_edges,
        "shd": shd,
        "sid": sid_vals,
        "runtime": elapsed,
        "sampler_output": result,
    }

    output_path = make_output_path(GEN_METHOD, N, D, SIGMA, array_id)
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"array_id={array_id}")
    print(f"checkpoint_dir={ckpt_dir}")
    print(f"output_path={output_path}")
    print(f"runtime={elapsed:.3f}s, shd={shd}, sid={sid_vals}")


if __name__ == "__main__":
    main()
