"""Benchmark execution speed for Bat Algorithm."""
import time
import numpy as np

from src.core.bat_algorithm import BatAlgorithm
from src.core.parameters import BatParameters
from src.evaluation.benchmark_funcs import SphereEvaluator


def test_performance_benchmark():
    dim = 20
    evaluator = SphereEvaluator(target=np.zeros(dim, dtype=np.int64))
    lower_bounds = np.full(dim, -50, dtype=np.int64)
    upper_bounds = np.full(dim, 50, dtype=np.int64)

    params = BatParameters(
        population_size=100,
        max_iterations=200,
        random_seed=42,
    )

    optimizer = BatAlgorithm(evaluator, lower_bounds, upper_bounds, params)
    t0 = time.perf_counter()
    res = optimizer.optimize()
    elapsed = time.perf_counter() - t0

    # 100 bats * 200 iterations = 20,000 evaluations across 20 dimensions
    # Optimized kernel must complete in under 1.5 seconds
    assert elapsed < 1.5
    assert res.iterations_run == 200
