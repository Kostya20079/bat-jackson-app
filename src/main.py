"""
Entry point for the Bat Algorithm Jackson Network application.
Supports running in GUI mode (default) or CLI mode.
"""
import sys
import os
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.core.parameters import BatParameters
from src.core.bat_algorithm import BatAlgorithm
from src.evaluation.jackson_adapter import (
    JacksonNetworkConfig,
    JacksonNetworkEvaluator,
)


def run_cli_optimization(population: int = 30, iterations: int = 100):
    """Run optimization directly in command-line mode."""
    print("=" * 60)
    print("BAT ALGORITHM OPTIMIZATION - CLI MODE")
    print("=" * 60)

    # 3-node academic network configuration
    gamma = np.array([4.0, 2.0, 1.0])
    mu = np.array([5.0, 4.0, 3.0])
    P = np.array([
        [0.0, 0.3, 0.2],
        [0.2, 0.0, 0.3],
        [0.1, 0.2, 0.0],
    ])
    c_min = np.array([1, 1, 1])
    c_max = np.array([8, 8, 8])

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

    evaluator = JacksonNetworkEvaluator(config)
    params = BatParameters(
        population_size=population,
        max_iterations=iterations,
        random_seed=42,
    )

    print(f"Nodes: 3 | Arrival rates: {gamma} | Service rates: {mu}")
    print(f"Solving traffic equations -> Arrival rates lambda: {np.round(evaluator.lambdas, 3)}")
    print(f"Running Bat Algorithm (N={population}, T={iterations})...")

    optimizer = BatAlgorithm(evaluator, c_min, c_max, params)
    result = optimizer.optimize()

    chars = evaluator.calculate_characteristics(result.best_solution)

    print("\n" + "-" * 60)
    print("RESULTS:")
    print(f"Optimal channels c*: {result.best_solution}")
    print(f"Best fitness f(c*): {result.best_fitness:.4f}")
    print(f"Mean network response time W_sieci: {chars.w_network:.4f}")
    print(f"Mean network customers L_sieci: {chars.l_network:.4f}")
    print(f"Total channel cost C(c): {chars.channel_cost:.1f}")
    print(f"Channel utilizations rho: {np.round(chars.rhos, 4)}")
    print(f"Network stability: {'STABLE' if chars.is_stable else 'UNSTABLE'}")
    print(f"Execution time: {result.execution_time_sec * 1000:.2f} ms")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Bat Algorithm Queueing Network Optimizer")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode instead of GUI")
    parser.add_argument("--pop", type=int, default=30, help="Population size")
    parser.add_argument("--iters", type=int, default=100, help="Max iterations")

    args = parser.parse_args()

    if args.cli:
        run_cli_optimization(population=args.pop, iterations=args.iters)
    else:
        try:
            from src.gui.app import BatOptimizationApp
            app = BatOptimizationApp()
            app.mainloop()
        except Exception as e:
            print(f"Could not initialize GUI ({e}). Falling back to CLI mode...")
            run_cli_optimization(population=args.pop, iterations=args.iters)


if __name__ == "__main__":
    main()
