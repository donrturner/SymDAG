# -*- coding: utf-8 -*-
"""Legacy SymDAG-Greedy core adapted from the original notebook export."""

import gc
import inspect
import os
import pickle
import re
import sys
import time
from contextlib import redirect_stdout
from itertools import permutations

import networkx as nx
import numpy as np
import pandas as pd
import patsy
import psutil
from joblib import Parallel, delayed, parallel_backend
from pgmpy.base import DAG
from pgmpy.models import LinearGaussianBayesianNetwork
from rils_rols.rils_rols import RILSROLSRegressor
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sympy import preorder_traversal

os.environ["PYTHONWARNINGS"] = "ignore"

GREEDY_SAMPLE_SIZE = 1
GREEDY_MAX_COMPLEXITY = 35
GREEDY_VERBOSE = False
GREEDY_MAX_FIT_CALLS = 10000
GREEDY_OPERATORS = None
GREEDY_EXTRA_KWARGS = {}


def _build_regressor():
    kwargs = {
        "sample_size": GREEDY_SAMPLE_SIZE,
        "max_complexity": GREEDY_MAX_COMPLEXITY,
        "verbose": GREEDY_VERBOSE,
        "max_fit_calls": GREEDY_MAX_FIT_CALLS,
    }
    kwargs.update(dict(GREEDY_EXTRA_KWARGS))

    sig = inspect.signature(RILSROLSRegressor.__init__)
    if GREEDY_OPERATORS is not None:
        for candidate in ("operators", "allowed_operators", "function_set"):
            if candidate in sig.parameters:
                kwargs[candidate] = list(GREEDY_OPERATORS)
                break
        else:
            raise ValueError(
                "The installed rils-rols backend does not expose an operator keyword. "
                "Leave operators=None to use the backend defaults."
            )
    return RILSROLSRegressor(**kwargs)

def complexity_sympy(model):
    c=0
    for _ in preorder_traversal(model):
        c += 1
    return c

# https://www.sciencedirect.com/science/article/pii/S0021999124005096
def BIC (err, d, n):
    return n * err + d * np.log(n)

def score_save (A, B):
    #  os.environ['OMP_NUM_THREADS'] = '1'
    with open(os.devnull, 'w') as f:
      with redirect_stdout(f):
        regressor = _build_regressor()
    regressor.fit(A, B)
    preds = regressor.predict(A)
    mse = mean_squared_error(B, preds)
    d = complexity_sympy(regressor.model_string())
    vars = re.findall(r'x\d+', str(regressor.model_string()))
    indices = list({int(v[1:]) for v in vars})
    # If a variable is selected out, we don't add the edge
    if len(indices) != A.shape[1] and A.values.sum() != A.shape[0]:
      regressor = None
      return 1000000000000000000
    regressor = None
    bic_val = BIC(err=np.log(mse), d=d, n=A.shape[0])
    gc.collect()
    return bic_val

def score_par (data, key):
  with open(os.devnull, 'w') as f:
    with redirect_stdout(f):
      regressor = _build_regressor()
  regressor.fit(data[:,list(key[0])], data[:,key[1]].reshape(-1, 1))
  preds = regressor.predict(data[:,list(key[0])])
  mse = mean_squared_error(data[:,key[1]].reshape(-1, 1), preds)
  # model_string = simplify(regressor.model_string())
  d = complexity_sympy(regressor.model_string())
  # d = complexity_sympy(model_string)
  vars = re.findall(r'x\d+', regressor.model)
  # vars = re.findall(r'x\d+', str(model_string))
  indices = list({int(v[1:]) for v in vars})
  # If a variable is selected out, we don't add the edge
  if len(indices) != data[:,list(key[0])].shape[1]:
    regressor=preds=mse=d=vars=indices = None
    gc.collect()
    return 1000000000000000000
  bic_val = BIC(err=np.log(mse), d=d, n=data[:,list(key[0])].shape[0])
  regressor=preds=mse=d=vars=indices = None
  gc.collect()
  return bic_val

def neg_normal_ML (A):
  return score_save(pd.DataFrame(np.repeat(1, A.shape[0]).reshape(-1, 1)), A)

def possible_edge_mods(model: DAG):
    nodes = model.nodes()
    current_edges = set(model.edges())

    # All unordered pairs of distinct nodes
    all_possible_edges = set(permutations(nodes, 2))
    additions = []
    deletions = list(current_edges)
    reversals = []

    for u, v in all_possible_edges:
        # Edge addition: (u, v) is not in the model and doesn't create a cycle
        if (u, v) not in current_edges:
            temp_model = model.copy()
            try:
                temp_model.add_edge(u, v)
                if nx.is_directed_acyclic_graph(temp_model):
                    additions.append((u, v))
            except ValueError:
                pass

    for u, v in current_edges:
        # Edge reversal: remove (u,v) and try to add (v,u)
        temp_model = model.copy()
        temp_model.remove_edge(u, v)
        try:
            temp_model.add_edge(v, u)
            if nx.is_directed_acyclic_graph(temp_model):
                reversals.append((u, v))  # meaning (u,v) → (v,u)
        except ValueError:
            continue

    return {
        'additions': additions,
        'deletions': deletions,
        'reversals': reversals
    }

# Should edit to include data as input perhaps?
def mod_bic_save (model: DAG, data, regression_cache):
  bbic = 0
  for node in model.nodes():
    parents = model.get_parents(node)
    if parents:
      key = (tuple(sorted(parents)), node)
      if key in regression_cache:
        bbic += regression_cache[key]
      else:
        ppar = data[parents]
        regression_cache[key] = score_save(ppar, data[node].values.reshape(-1, 1))
        bbic += regression_cache[key]
    else:
      key = (node, node)
      if key in regression_cache:
        bbic += regression_cache[key]
      else:
        regression_cache[key] = score_save(pd.DataFrame(np.repeat(1, len(data[node].values)).reshape(-1, 1)), data[node].values.reshape(-1, 1))
        bbic += regression_cache[key]
  gc.collect()
  return {'bbic': bbic, 'regression_cache': regression_cache}

# Greedy search algorithm function for finding true graph
def graphsearch_par (data, graph_init):
    print("Performing Greedy Search...")
    procs = len(psutil.Process().cpu_affinity())
    regression_cache = {}
    init = mod_bic_save(graph_init, data, regression_cache)
    regression_cache = init['regression_cache']
    bic_init = init['bbic']
    imp = True
    pgraph = graph_init.copy()

    while imp:
      bbic = []
      possible_mods = possible_edge_mods(pgraph)
      testgraphs = []
      idx = 0
      # Testing all possible graph additions
      for mod in possible_mods['additions']:
        testgraphs.append(pgraph.copy())
        testgraphs[idx].add_edge(mod[0], mod[1])
        idx += 1


      # Testing all possible graph deletions
      for mod in possible_mods['deletions']:
        testgraphs.append(pgraph.copy())
        testgraphs[idx].remove_edge(mod[0], mod[1])
        idx += 1

      # Testing all possible graph reversals
      for mod in possible_mods['reversals']:
        testgraphs.append(pgraph.copy())
        testgraphs[idx].remove_edge(mod[0], mod[1])
        testgraphs[idx].add_edge(mod[1], mod[0])
        idx += 1

      regs_uncache = []
      for model in testgraphs:
        for node in model.nodes():
          parents = model.get_parents(node)
          if parents:
            # ppar = [data[n] for n in parents]
            # ppar = np.hstack(ppar)
            # # Reshape ppar to be 2D
            # ppar = ppar.reshape(-1, len(parents))
            key = (tuple(sorted(parents)), node)
            if key in regression_cache:
              continue
            else:
              regs_uncache.append(key)

      indices = []
      for key in regs_uncache:
        indices.append((tuple([data.columns.get_loc(parent) for parent in key[0]]), data.columns.get_loc(key[1])))

      passdat = data.values

      with parallel_backend("loky", n_jobs=procs * 2, inner_max_num_threads=1):
        out = Parallel()(delayed(score_par)(passdat, key) for key in indices)
      for i in range(len(indices)):
        regression_cache[regs_uncache[i]] = out[i]
      for model in testgraphs:
        bbic.append(mod_bic_save(model, data, regression_cache)['bbic'])

      # Choosing best change to graph
      bbic = np.array(bbic)
      if bbic.min() < bic_init:
        bic_init = bbic.min()
        flat = [item for sublist in possible_mods.values() for item in sublist]
        if bbic.argmin() < len(possible_mods['additions']):
          pgraph.add_edge(flat[bbic.argmin()][0], flat[bbic.argmin()][1])
        elif bbic.argmin() < len(possible_mods['additions']) + len(possible_mods['deletions']):
          pgraph.remove_edge(flat[bbic.argmin()][0], flat[bbic.argmin()][1])
        else:
          pgraph.remove_edge(flat[bbic.argmin()][0], flat[bbic.argmin()][1])
          pgraph.add_edge(flat[bbic.argmin()][1], flat[bbic.argmin()][0])
        continue
      imp = False
      functs = {}
      for node in pgraph.nodes():
        success = False
        while not success:
          par = pgraph.get_parents(node)
          if par:
            key = (tuple(sorted(par)), node)
            with open(os.devnull, 'w') as f:
              with redirect_stdout(f):
                regressor = _build_regressor()
            regressor.fit(data[par], data[node])
            vars = re.findall(r'x\d+', str(regressor.model_string()))
            indices = list({int(v[1:]) for v in vars})
            # If a variable is selected out, we don't add the edge
            if len(indices) != len(par):
              for p in list(set(range(len(par))) - set(indices)):
                pgraph.remove_edge(par[p], node)
              continue
            functs[key] = regressor.model_string()
          success = True
    return {'graph': pgraph, 'bic': bic_init, 'functs': functs, 'cache': regression_cache}

equations = [
    "2 * np.cos(np.pi * gdata[edge[0]]) * gdata[edge[0]] - (0.25 * gdata[edge[0]]) ** 2",
    "2 * np.cos(np.pi * gdata[edge[0]])",
    "2 * np.sin(np.pi * gdata[edge[0]])",
    "(0.25 * gdata[edge[0]]) ** 2 + 2 * np.cos(np.pi * gdata[edge[0]])",
    "2 * np.sin(np.pi * np.log(np.square(gdata[edge[0]])))",
    "2 * np.sin(gdata[edge[0]] ** 3)"
]

# Matthew's Correlation Coefficient
def MCC(adj_sim, adj_true):
  tp = np.sum(np.logical_and(adj_sim == 1, adj_true == 1))
  tn = np.sum(np.logical_and(adj_sim == 0, adj_true == 0))
  fp = np.sum(np.logical_and(adj_sim == 1, adj_true == 0))
  fn = np.sum(np.logical_and(adj_sim == 0, adj_true == 1))
  if tp + fp == 0 and fn == 0:
    return 1
  elif (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
    return 0
  return (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

def gpgen(X, sigma = 1):
  kernel = sigma * RBF(length_scale=1.0)
  K = kernel(X, X)
  mean = np.zeros(X.shape[0])
  return np.random.multivariate_normal(mean, K, size=1)

def materngen(X, nu):
  kernel = Matern(length_scale=1.0, nu=nu)
  K = kernel(X, X)
  mean = np.zeros(X.shape[0])
  return np.random.multivariate_normal(mean, K, size=1)

def splinegen(parent_data: pd.DataFrame, num_knots: int) -> np.ndarray:
  n_samples, n_parents = parent_data.shape
  dependent_vector = np.zeros(n_samples)

  df = num_knots + 3

  for col in parent_data.columns:
    basis_matrix = patsy.dmatrix(
      f"bs(x, df={df}, include_intercept=True) - 1",
      {"x": parent_data[col]},
      return_type='dataframe'
    )
    coeffs = np.random.randn(basis_matrix.shape[1])
    parent_contribution = np.dot(basis_matrix, coeffs)
    dependent_vector += parent_contribution

  return dependent_vector

def search_sim_par(n, d, edgeprob, sigma, gen_method = 'gp'):
    start = time.perf_counter()
    # Data generation
    edgesave = []
    test = nx.scale_free_graph(d, 0.1, 0.05, 0.85, delta_in=0, delta_out=0.2)
    test = nx.relabel_nodes(test, {i: f'X_{i}' for i in range(d)})
    while not nx.is_directed_acyclic_graph(test):
      try:
        # Find a cycle
        cycle = nx.find_cycle(test, orientation='original')
        # Remove a random edge from the cycle
        edge_to_remove = np.random.choice(len(cycle))
        edge_to_remove = cycle[edge_to_remove]
        test.remove_edge(edge_to_remove[0], edge_to_remove[1])
      except nx.NetworkXNoCycle:
      # This should not be reached if the while condition is correct
        break
    gmodel = LinearGaussianBayesianNetwork(DAG(test.to_directed()))
    gmodel.cpds = gmodel.get_random_cpds()
    # gmodel = LinearGaussianBayesianNetwork.get_random(n_nodes = d, edge_prob=edgeprob, node_names=[f'X_{i}' for i in range(d)])
    # immoral = gmodel.get_immoralities()
    # while not all([len(p) == 0 for p in immoral.values()]):
    #   gmodel = LinearGaussianBayesianNetwork.get_random(n_nodes = d, edge_prob=edgeprob)
    #   immoral = gmodel.get_immoralities()
    gdata = gmodel.simulate(n)
    scaler = StandardScaler()
    gdata = pd.DataFrame(scaler.fit_transform(gdata), columns=gdata.columns)
    functs = []
    for root in gmodel.get_roots():
      gdata[root] = np.random.uniform(low=-1, high=1, size=n)
    gdata = pd.DataFrame(scaler.fit_transform(gdata), columns=gdata.columns)
    for node in [f'X_{i}' for i in range(d)]:
      ancestors = gmodel.get_ancestral_graph(node).edges
      if ancestors:
        for edge in ancestors:
          if edge in edgesave:
            continue
          else:
            parents = gmodel.get_parents(edge[1])
            if gen_method == 'gp':
              gdata[edge[1]] = gpgen(gdata[parents]).reshape(-1, 1)
              gdata = pd.DataFrame(scaler.fit_transform(gdata), columns=gdata.columns)
              gdata[edge[1]] += sigma * np.random.randn(n)
              edgesave += [(parent, edge[1]) for parent in parents]
              continue
            elif gen_method == 'spl':
              gdata[edge[1]] = splinegen(gdata[parents], 4).reshape(-1, 1)
              gdata = pd.DataFrame(scaler.fit_transform(gdata), columns=gdata.columns)
              gdata[edge[1]] += sigma * np.random.randn(n)
              edgesave = edgesave + [(parent, edge[1]) for parent in parents]
              continue
            elif gen_method == 'gamma':
              gdata[edge[1]] = gpgen(gdata[parents]).reshape(-1, 1)
              gdata = pd.DataFrame(scaler.fit_transform(gdata), columns=gdata.columns)
              gdata[edge[1]] += np.random.gamma(shape = 1, scale = sigma, size = n)
              edgesave = edgesave + [(parent, edge[1]) for parent in parents]
              continue
            if len(parents) > 1:
              parmult = 0
              colmult = 1
              for parent in parents:
                parmult += np.random.choice([-1, 1]) * np.square(gdata[parent])
                colmult *= gdata[parent]
                edgesave.append((parent, edge[1]))
              gdata[edge[1]] = parmult + colmult + sigma * np.random.randn(n)
              gdata = pd.DataFrame(scaler.fit_transform(gdata), columns=gdata.columns)
              functs.append(parents)
              continue
            if not [item for item in equations if item not in functs]:
                funct = np.random.choice(equations)
            else:
                funct = np.random.choice([item for item in equations if item not in functs])
            functs.append(funct)
            gdata[edge[1]] = np.random.choice([-1, 1]) * eval(funct) + sigma * np.random.randn(n)
            gdata = pd.DataFrame(scaler.fit_transform(gdata), columns=gdata.columns)
            edgesave.append(edge)
    graph_init = DAG()
    graph_init.add_nodes_from([f'X_{i}' for i in range(d)])
    if d <= 4:
      search = completesearch(gdata, graph_init)
    else:
      search = graphsearch_par(gdata, graph_init)
    total_time = time.perf_counter() - start
    return {'sim graph': search['graph'], 'true graph': DAG(gmodel), 'bic': search['bic'], 'data': gdata, 'cache': search['cache'], 'functs': functs, 'runtime': total_time, 'est_fn': search['functs']}

if __name__ == '__main__':
  # mp.set_start_method('spawn', force=True)
  if len(sys.argv) > 1:
    task_id = int(sys.argv[1])
  else:
    task_id = -1 # Default or error state

  n = 150
  d = 30
  edgeprob = 1 / d
  # noise variance
  sigma = 0.1
  gen_method = 'eq'
  
  out = search_sim_par(n, d, edgeprob, sigma, gen_method = gen_method)

  # Define subdirectory
  output_subdir = f'output/scalefree/{gen_method}/new/output_n{n}_d{d}_sigma{sigma}'
  # Create subdirectory if it doesn't exist
  os.makedirs(output_subdir, exist_ok=True)

  # Define filename with subdirectory
  filename = os.path.join(output_subdir, f'output_n{n}_d{d}_sigma{sigma}_task{task_id}.pickle')

  with open(filename, 'wb') as handle:
    pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
