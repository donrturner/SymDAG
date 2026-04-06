from symdag import DataGenerationConfig, GreedyConfig, compute_metrics, run_symdag, simulate_dataset


simulation = simulate_dataset(
    DataGenerationConfig(
        n=100,
        d=8,
        sigma=0.1,
        gen_method="gp",
        seed=42,
        sparsity=1.0,
    )
)

result = run_symdag(
    simulation.data,
    method="greedy",
    config=GreedyConfig(
        max_complexity=35,
        max_fit_calls=10000,
        operators=None,
    ),
)

print(result.summary())
print(compute_metrics(result.graph, simulation.true_graph, simulation.config.d))
