from symdag import DataGenerationConfig, SymDAGConfig, compute_metrics, run_symdag, simulate_dataset


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
