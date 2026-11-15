"""Unit tests for Jackson Queueing Network evaluator."""
import numpy as np
import pytest

from src.core.bat_algorithm import BatAlgorithm
from src.core.parameters import BatParameters
from src.evaluation.jackson_adapter import (
    JacksonNetworkConfig,
    JacksonNetworkEvaluator,
)


@pytest.fixture
def sample_network():
    """Create sample 3-node network as described in document section 3.2."""
    gamma = np.array([4.0, 2.0, 1.0])
    mu = np.array([5.0, 4.0, 3.0])
    P = np.array([
        [0.0, 0.3, 0.2],
        [0.2, 0.0, 0.3],
        [0.1, 0.2, 0.0],
    ])
    c_min = np.array([1, 1, 1])
    c_max = np.array([5, 5, 5])
    config = JacksonNetworkConfig(
        num_nodes=3,
        gamma=gamma,
        mu=mu,
        P=P,
        c_min=c_min,
        c_max=c_max,
        alpha=1.0,
        beta=0.2,
    )
    return JacksonNetworkEvaluator(config)


def test_traffic_equations_solution(sample_network):
    lambdas = sample_network.lambdas
    assert len(lambdas) == 3
    # Check that (I - P^T) * lambda == gamma
    P = sample_network.config.P
    gamma = sample_network.config.gamma
    A = np.eye(3) - P.T
    residual = np.dot(A, lambdas) - gamma
    np.testing.assert_allclose(residual, 0.0, atol=1e-8)
    assert np.all(lambdas > 0.0)


def test_stability_and_characteristics(sample_network):
    # If c is too small, rho >= 1 -> unstable
    unstable_c = np.array([1, 1, 1])
    assert not sample_network.is_feasible(unstable_c)
    chars_unstable = sample_network.calculate_characteristics(unstable_c)
    assert not chars_unstable.is_stable
    assert chars_unstable.objective_value == sample_network.penalty_value

    # If c is sufficiently large -> stable
    stable_c = np.array([4, 4, 4])
    assert sample_network.is_feasible(stable_c)
    chars_stable = sample_network.calculate_characteristics(stable_c)
    assert chars_stable.is_stable
    assert np.all(chars_stable.rhos < 1.0)
    assert chars_stable.w_network > 0.0
    assert chars_stable.l_network > 0.0
    assert chars_stable.objective_value < sample_network.penalty_value


def test_bat_algorithm_optimizes_jackson_network(sample_network):
    c_min = sample_network.config.c_min
    c_max = sample_network.config.c_max

    params = BatParameters(
        population_size=25,
        max_iterations=40,
        random_seed=777,
    )
    optimizer = BatAlgorithm(sample_network, c_min, c_max, params)
    result = optimizer.optimize()

    assert result.feasible
    assert result.best_fitness < sample_network.penalty_value
    # Solution must be stable
    chars = sample_network.calculate_characteristics(result.best_solution)
    assert chars.is_stable
    assert np.all(chars.rhos < 1.0)
