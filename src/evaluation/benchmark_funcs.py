"""Standard benchmark functions for validating metaheuristic optimization."""
import numpy as np
from src.evaluation.base import BaseEvaluator


class SphereEvaluator(BaseEvaluator):
    """
    Continuous/integer sphere function:
    f(x) = sum( (x_i - target_i)^2 )
    """
    def __init__(self, target: np.ndarray | None = None):
        self.target = target

    def evaluate(self, solution: np.ndarray) -> float:
        sol = np.asarray(solution, dtype=np.float64)
        if self.target is not None:
            sol = sol - self.target
        return float(np.sum(sol ** 2))

    def is_feasible(self, solution: np.ndarray) -> bool:
        return True


class RastriginEvaluator(BaseEvaluator):
    """
    Discrete/continuous Rastrigin benchmark function (highly multimodal):
    f(x) = 10*n + sum( x_i^2 - 10*cos(2*pi*x_i) )
    """
    def __init__(self, target: np.ndarray | None = None):
        self.target = target

    def evaluate(self, solution: np.ndarray) -> float:
        sol = np.asarray(solution, dtype=np.float64)
        if self.target is not None:
            sol = sol - self.target
        n = len(sol)
        return float(10 * n + np.sum(sol ** 2 - 10 * np.cos(2 * np.pi * sol)))

    def is_feasible(self, solution: np.ndarray) -> bool:
        return True
