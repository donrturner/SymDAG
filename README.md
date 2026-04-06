# SymDAG

SymDAG is a Python package for interpretable nonlinear causal discovery via symbolic regression.

The package currently exposes two related methods:

- `greedy` for SymDAG-Greedy
- `bayes` for SymDAG-Bayes

Both methods aim to estimate a directed acyclic graph together with readable symbolic mechanism estimates. SymDAG-Greedy embeds symbolic regression inside a score-based DAG search, while SymDAG-Bayes uses stochastic ordering updates and posterior-weighted local symbolic models to produce posterior edge-support summaries in addition to representative equations.

## What The Package Includes

- a unified Python API for `greedy` and `bayes`
- a command-line interface exposed as `symdag`
- simulated-data generation utilities using the current SymDAG-Bayes data generators
- examples for both methods on simulated data
- configurable symbolic-regression settings, including Bayesian hyperparameters such as burn-in, eFDR thresholding, particle settings, and symbolic operators

The simulated-data generators currently support:

- `gp` for Gaussian-process mechanisms
- `gpi` for Gaussian-process mechanisms with interaction structure
- `spl` for spline mechanisms
- `eq` for symbolic-function mechanisms

## Repository Layout

```text
pyproject.toml
README.md
examples/
src/symdag/
```

## Installation

Create and activate a Python 3.10+ environment first, then install the package from the repository root.

Install both backends:

```bash
pip install -e .[full]
```

Install only the Bayesian backend:

```bash
pip install -e .[bayes]
```

Install only the greedy backend:

```bash
pip install -e .[greedy]
```

Core dependencies are declared in `pyproject.toml`. The optional backend extras pull in the symbolic-regression libraries used by the current implementation:

- SymDAG-Greedy uses `rils-rols`
- SymDAG-Bayes uses `pysips`

## Quick Start

```python
from symdag import BayesConfig, DataGenerationConfig, GreedyConfig
from symdag import compute_metrics, run_symdag, simulate_dataset

simulation = simulate_dataset(
    DataGenerationConfig(
        n=150,
        d=10,
        sigma=0.1,
        gen_method="gp",
        seed=42,
        sparsity=1.0,
    )
)

bayes_result = run_symdag(
    simulation.data,
    method="bayes",
    config=BayesConfig(
        n_iter=1000,
        burnin=500,
        samples_per_fit=5,
        efdr_q=0.05,
        use_efdr_threshold=True,
        operators=["+", "-", "*", "cos"],
        num_particles=100,
        num_mcmc_samples=5,
    ),
)

greedy_result = run_symdag(
    simulation.data,
    method="greedy",
    config=GreedyConfig(
        max_complexity=35,
        max_fit_calls=10000,
    ),
)

print(compute_metrics(bayes_result.graph, simulation.true_graph, simulation.config.d))
print(greedy_result.summary())
```

## Command Line Usage

Run SymDAG-Bayes on simulated data:

```bash
symdag --method bayes --n 150 --d 10 --sigma 0.1 --gen-method gp --n-iter 1000 --burnin 500 --use-efdr-threshold --efdr-q 0.05 --operators + - * cos
```

Run SymDAG-Greedy on simulated data:

```bash
symdag --method greedy --n 100 --d 8 --sigma 0.1 --gen-method gp --max-complexity 35 --max-fit-calls 10000
```

Run on a CSV instead of simulated data:

```bash
symdag --method bayes --input-csv my_data.csv --n-iter 1000 --burnin 500
```

## Configuration

The public API exposes dataclass-based configuration objects:

- `DataGenerationConfig`
- `GreedyConfig`
- `BayesConfig`

Important Bayesian settings include:

- `n_iter`
- `burnin`
- `samples_per_fit`
- `efdr_q`
- `use_efdr_threshold`
- `edge_threshold`
- `num_particles`
- `num_mcmc_samples`
- `max_equation_evals`
- `final_max_equation_evals`
- `operators`

Important greedy settings include:

- `sample_size`
- `max_complexity`
- `max_fit_calls`
- `operators`

## Symbolic Regression Operators

- SymDAG-Bayes defaults to the current project operator set: `["+", "-", "*", "cos"]`
- SymDAG-Greedy defaults to the backend RILS-ROLS operator grammar used by the current implementation
- Both backends expose an `operators` setting
- For greedy, operator overrides are passed through only if the installed `rils-rols` build exposes an operator keyword

## Examples

- `examples/simulated_bayes.py`
- `examples/simulated_greedy.py`

## Current Status

This package is a packaging-oriented integration of the current project code. It keeps the active SymDAG-Greedy and SymDAG-Bayes implementations together under one installable interface and uses the current Bayesian simulation/data-generation workflow for examples and smoke tests.
