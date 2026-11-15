"""Unit tests for Bat Algorithm core logic."""
import numpy as np
import pytest

from src.core.bat_algorithm import BatAlgorithm
from src.core.parameters import BatParameters
from src.evaluation.benchmark_funcs import SphereEvaluator, RastriginEvaluator


def test_parameter_validation():
    # Valid parameters
    params = BatParameters(population_size=20, max_iterations=50)
    params.validate()

    # Invalid cases
    with pytest.raises(ValueError):
        BatParameters(population_size=-5).validate()
    with pytest.raises(ValueError):
        BatParameters(f_min=2.0, f_max=1.0).validate()
    with pytest.raises(ValueError):
        BatParameters(a_init=1.5).validate()
    with pytest.raises(ValueError):
        BatParameters(alpha=-0.1).validate()


def test_bat_algorithm_sphere_minimization():
    target = np.array([3, 4, 2], dtype=np.int64)
    evaluator = SphereEvaluator(target=target)

    lower_bounds = np.array([1, 1, 1], dtype=np.int64)
    upper_bounds = np.array([10, 10, 10], dtype=np.int64)

    params = BatParameters(
        population_size=30,
        max_iterations=80,
        random_seed=42,
    )
    optimizer = BatAlgorithm(evaluator, lower_bounds, upper_bounds, params)
    result = optimizer.optimize()

    # Check that solution is inside bounds and integer
    assert np.all(result.best_solution >= lower_bounds)
    assert np.all(result.best_solution <= upper_bounds)
    assert np.issubdtype(result.best_solution.dtype, np.integer)
    assert result.best_fitness >= 0.0

    # Bat algorithm should converge close to the exact integer target
    np.testing.assert_array_equal(result.best_solution, target)
    assert result.best_fitness == 0.0


def test_bat_algorithm_convergence_history():
    evaluator = RastriginEvaluator()
    lower_bounds = np.array([-5, -5], dtype=np.int64)
    upper_bounds = np.array([5, 5], dtype=np.int64)

    callback_records = []
    def callback(t, best_f, best_sol):
        callback_records.append((t, best_f))

    params = BatParameters(population_size=20, max_iterations=30, random_seed=123)
    optimizer = BatAlgorithm(evaluator, lower_bounds, upper_bounds, params)
    result = optimizer.optimize(iteration_callback=callback)

    assert len(result.convergence_history) == 31  # initial + 30 iterations
    assert len(callback_records) == 30
    # Convergence history must be non-increasing
    for i in range(len(result.convergence_history) - 1):
        assert result.convergence_history[i] >= result.convergence_history[i + 1]
