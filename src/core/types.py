"""Data types and protocols for optimization."""
from dataclasses import dataclass, field
from typing import List, Protocol, runtime_checkable
import numpy as np


@dataclass
class OptimizationResult:
    """Stores the output and history of the optimization process."""
    best_solution: np.ndarray       # Optimal integer channels c_i
    best_fitness: float            # Minimal objective function value
    convergence_history: List[float] = field(default_factory=list)
    iterations_run: int = 0
    execution_time_sec: float = 0.0
    feasible: bool = True
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class ObjectiveFunctionProtocol(Protocol):
    """Protocol for objective function evaluation."""
    def evaluate(self, solution: np.ndarray) -> float:
        """Evaluate discrete solution vector and return fitness (minimization)."""
        ...

    def is_feasible(self, solution: np.ndarray) -> bool:
        """Check if solution satisfies system stability and constraints."""
        ...
