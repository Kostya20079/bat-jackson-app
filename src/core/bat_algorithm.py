"""Bat Algorithm optimization implementation."""
import time
from typing import Callable, Optional
import numpy as np

from src.core.parameters import BatParameters
from src.core.types import OptimizationResult, ObjectiveFunctionProtocol
from src.core.bat_kernels import (
    update_bat_kinematics,
    project_discrete_bounds,
    local_random_walk,
)


class BatAlgorithm:
    """
    Bat Algorithm (BA) optimizer for discrete/integer search spaces.
    Optimized for minimizing queueing network costs and waiting times.
    """

    def __init__(
        self,
        objective_evaluator: ObjectiveFunctionProtocol,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        params: Optional[BatParameters] = None,
    ):
        self.evaluator = objective_evaluator
        self.lower_bounds = np.asarray(lower_bounds, dtype=np.int64)
        self.upper_bounds = np.asarray(upper_bounds, dtype=np.int64)
        self.dim = len(self.lower_bounds)

        if len(self.upper_bounds) != self.dim:
            raise ValueError("lower_bounds and upper_bounds must have the same dimension")
        if np.any(self.lower_bounds > self.upper_bounds):
            raise ValueError("All lower_bounds must be <= upper_bounds")

        self.params = params or BatParameters()
        self.params.validate()

        self.rng = np.random.default_rng(self.params.random_seed)

    def optimize(
        self,
        iteration_callback: Optional[Callable[[int, float, np.ndarray], None]] = None,
    ) -> OptimizationResult:
        """
        Execute Bat Algorithm optimization.
        
        Args:
            iteration_callback: Optional callback func(iter_num, current_best_fitness, best_solution)
        """
        start_time = time.perf_counter()
        p = self.params
        n = p.population_size
        d = self.dim

        # 1. Initialize population
        # Continuous positions initialized uniformly across discrete bounds
        continuous_positions = self.rng.uniform(
            self.lower_bounds - 0.49,
            self.upper_bounds + 0.49,
            size=(n, d)
        )
        velocities = np.zeros((n, d), dtype=np.float64)
        frequencies = np.zeros(n, dtype=np.float64)
        loudness = np.full(n, p.a_init, dtype=np.float64)
        pulse_rates = np.full(n, p.r_init, dtype=np.float64)
        initial_pulse_rates = np.full(n, p.r_init, dtype=np.float64)

        # Discretize initial positions
        discrete_positions = project_discrete_bounds(
            continuous_positions, self.lower_bounds, self.upper_bounds
        )

        fitnesses = np.empty(n, dtype=np.float64)
        for i in range(n):
            fitnesses[i] = self.evaluator.evaluate(discrete_positions[i])

        # Find best initial bat
        best_idx = int(np.argmin(fitnesses))
        best_fitness = float(fitnesses[best_idx])
        best_discrete = discrete_positions[best_idx].copy()
        best_continuous = continuous_positions[best_idx].copy()

        history = [best_fitness]

        # 2. Main iterative loop
        for t in range(1, p.max_iterations + 1):
            # Frequency update: f_i = f_min + (f_max - f_min) * beta
            beta = self.rng.uniform(0.0, 1.0, size=n)
            frequencies = p.f_min + (p.f_max - p.f_min) * beta

            # Kinematics update (velocity & continuous position)
            continuous_positions = update_bat_kinematics(
                continuous_positions, velocities, best_continuous, frequencies
            )

            # Local walk / exploitation
            rand_pulse = self.rng.uniform(0.0, 1.0, size=n)
            mean_a = float(np.mean(loudness))

            # Candidate discrete proposals
            new_discrete = project_discrete_bounds(
                continuous_positions, self.lower_bounds, self.upper_bounds
            )

            for i in range(n):
                candidate_sol = new_discrete[i]
                if rand_pulse[i] > pulse_rates[i]:
                    rand_eps = self.rng.uniform(-1.0, 1.0, size=d)
                    candidate_sol = local_random_walk(
                        best_discrete, mean_a, self.lower_bounds, self.upper_bounds, rand_eps
                    )

                f_candidate = self.evaluator.evaluate(candidate_sol)

                # Acceptance criterion: if rand < A_i and fitness improved
                rand_loud = self.rng.uniform(0.0, 1.0)
                if rand_loud < loudness[i] and f_candidate <= fitnesses[i]:
                    discrete_positions[i] = candidate_sol
                    fitnesses[i] = f_candidate
                    continuous_positions[i] = candidate_sol.astype(np.float64)

                    # Update loudness and emission rate
                    loudness[i] = p.alpha * loudness[i]
                    pulse_rates[i] = initial_pulse_rates[i] * (1.0 - np.exp(-p.gamma * t))

                    # Update global best
                    if f_candidate < best_fitness:
                        best_fitness = f_candidate
                        best_discrete = candidate_sol.copy()
                        best_continuous = candidate_sol.astype(np.float64)

            history.append(best_fitness)

            if iteration_callback is not None:
                iteration_callback(t, best_fitness, best_discrete)

        elapsed = time.perf_counter() - start_time
        return OptimizationResult(
            best_solution=best_discrete,
            best_fitness=best_fitness,
            convergence_history=history,
            iterations_run=p.max_iterations,
            execution_time_sec=elapsed,
            feasible=self.evaluator.is_feasible(best_discrete),
            metadata={
                "population_size": p.population_size,
                "dimension": d,
                "final_mean_loudness": float(np.mean(loudness)),
            }
        )
