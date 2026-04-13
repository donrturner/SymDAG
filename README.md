# SymDAG

SymDAG is a Python package for interpretable nonlinear causal discovery with symbolic regression.

In the current implementation, SymDAG:

- assumes nonlinear additive-noise structural equations
- searches graph space through stochastic updates to a latent topological ordering
- fits local symbolic models with PySIPS and aggregates posterior edge support across retained samples
- returns a thresholded DAG, posterior edge summaries, and representative symbolic equations

## What The Package Includes

- one public fitting API, `run_symdag`
- one public configuration object, `SymDAGConfig`
- a command-line interface exposed as `symdag`
- simulated-data utilities derived from the current project implementation
- an example script on simulated data
- user-tunable hyperparameters for burn-in, retained samples, eFDR thresholding, particle settings, equation-evaluation budgets, warm starts, and symbolic operators

## Repository Layout

```text
pyproject.toml
README.md
examples/
src/symdag/
```

## Installation

Create and activate a Python 3.10+ environment first, then install the package from the repository root.

Install SymDAG:

```bash
pip install git+"https://github.com/donrturner/SymDAG/"
```

Install optional helpers for warm starts and SID metrics:

```bash
pip install -e .[full]
```

## Quick Start

```python
from symdag import DataGenerationConfig, SymDAGConfig
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

result = run_symdag(
    simulation.data,
    config=SymDAGConfig(
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

print(result.summary())
print(compute_metrics(result.graph, simulation.true_graph, simulation.config.d))
```

## Command Line Usage

Run SymDAG on simulated data:

```bash
symdag --n 150 --d 10 --sigma 0.1 --gen-method gp --n-iter 1000 --burnin 500 --use-efdr-threshold --efdr-q 0.05 --operators "+" "-" "*" "cos"
```

Run on a CSV instead of simulated data:

```bash
symdag --input-csv my_data.csv --n-iter 1000 --burnin 500 --num-particles 100 --num-mcmc-samples 5
```

## Configuration

The public API exposes two main dataclass-based configuration objects:

- `DataGenerationConfig`
- `SymDAGConfig`

Important `SymDAGConfig` settings include:

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
- `max_complexity`
- `use_bn_warm_start`
- `random_state`

The defaults follow the current project implementation closely, including the smaller symbolic grammar described in the manuscript.

## Symbolic Regression Operators

SymDAG defaults to the operator set:

```python
["+", "-", "*", "cos"]
```

That matches the default symbolic-regression grammar used in the current implementation. Users can supply their own operators through `SymDAGConfig(operators=...)` or with `symdag --operators ...`.

## Simulated Data

The packaged data generator is adapted directly from the current project implementation and supports:

- `gp` for Gaussian-process mechanisms
- `gpi` for Gaussian-process mechanisms with interaction structure
- `spl` for spline mechanisms
- `eq` for symbolic-function mechanisms

Use `simulate_dataset(DataGenerationConfig(...))` to generate a standardized dataset together with the true DAG, ordering, and edge list.

## Example

- `examples/simulated_symdag.py`

## Current Status

This package is a packaging-oriented integration of the current SymDAG methodology. It exposes the stochastic symbolic-regression workflow under one installable interface and keeps the simulated-data utilities and hyperparameter controls aligned with the active implementation.
