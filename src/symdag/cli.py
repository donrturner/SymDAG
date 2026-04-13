from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import pandas as pd

from .api import run_symdag
from .config import DataGenerationConfig, SymDAGConfig
from .metrics import compute_metrics
from .simulation import simulate_dataset


def build_parser() -> argparse.ArgumentParser:
    defaults = SymDAGConfig()
    parser = argparse.ArgumentParser(description="Run SymDAG on a CSV file or simulated dataset.")
    parser.add_argument("--input-csv", help="Optional CSV file. If omitted, simulated data is generated.")
    parser.add_argument("--gen-method", default="gp", choices=("gp", "gpi", "spl", "eq"))
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sparsity", type=float, default=1.0)
    parser.add_argument("--edgeprob", type=float)
    parser.add_argument("--operators", nargs="+")

    parser.add_argument("--n-iter", type=int, default=3000)
    parser.add_argument("--burnin", type=int, default=2000)
    parser.add_argument("--samples-per-fit", type=int, default=5)
    parser.add_argument("--num-particles", type=int, default=100)
    parser.add_argument("--num-mcmc-samples", type=int, default=5)
    parser.add_argument("--max-equation-evals", type=int)
    parser.add_argument("--final-max-equation-evals", type=int, default=4000)
    parser.add_argument("--edge-threshold", type=float, default=0.3)
    parser.add_argument("--efdr-q", type=float, default=0.05)
    parser.add_argument("--use-efdr-threshold", dest="use_efdr_threshold", action="store_true")
    parser.add_argument("--no-efdr-threshold", dest="use_efdr_threshold", action="store_false")
    parser.add_argument("--use-data-driven-threshold", action="store_true")
    parser.add_argument("--target-n-edges", type=int)
    parser.add_argument("--n-chains", type=int, default=1)
    parser.add_argument("--chain-n-jobs", type=int)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--no-bn-warm-start", action="store_true")
    parser.add_argument("--model-selection", default="max_nml")
    parser.add_argument("--score-method", default="best")
    parser.add_argument("--max-complexity", type=int, default=defaults.max_complexity)
    parser.set_defaults(use_efdr_threshold=defaults.use_efdr_threshold)
    return parser


def _load_data(args):
    if args.input_csv:
        return pd.read_csv(args.input_csv), None

    sim_config = DataGenerationConfig(
        n=args.n,
        d=args.d,
        sigma=args.sigma,
        gen_method=args.gen_method,
        seed=args.seed,
        sparsity=args.sparsity,
        edgeprob=args.edgeprob,
    )
    simulation = simulate_dataset(sim_config)
    return simulation.data, simulation


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    data, simulation = _load_data(args)

    config = SymDAGConfig(
        n_iter=args.n_iter,
        burnin=args.burnin,
        random_state=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        max_equation_evals=args.max_equation_evals,
        final_max_equation_evals=args.final_max_equation_evals,
        score_method=args.score_method,
        model_selection=args.model_selection,
        samples_per_fit=args.samples_per_fit,
        n_chains=args.n_chains,
        chain_n_jobs=args.chain_n_jobs,
        edge_threshold=args.edge_threshold,
        use_efdr_threshold=args.use_efdr_threshold,
        efdr_q=args.efdr_q,
        use_data_driven_threshold=args.use_data_driven_threshold,
        target_n_edges=args.target_n_edges,
        edgeprob=args.edgeprob,
        use_bn_warm_start=not args.no_bn_warm_start,
        operators=args.operators or None,
        max_complexity=args.max_complexity,
        num_particles=args.num_particles,
        num_mcmc_samples=args.num_mcmc_samples,
    )

    result = run_symdag(data, config=config)
    payload = {
        "result": result.summary(),
        "config": asdict(config),
    }
    if simulation is not None:
        payload["simulation"] = {
            "config": asdict(simulation.config),
            "metrics": compute_metrics(result.graph, simulation.true_graph, simulation.config.d),
        }
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
