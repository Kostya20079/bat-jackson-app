"""Abstract base class for objective function evaluators."""
from abc import ABC, abstractmethod
import numpy as np


class BaseEvaluator(ABC):
    """Abstract base class establishing the contract for optimization targets."""

    @abstractmethod
    def evaluate(self, solution: np.ndarray) -> float:
        """Calculate fitness value to be minimized."""
        pass

    @abstractmethod
    def is_feasible(self, solution: np.ndarray) -> bool:
        """Verify whether solution meets all physical and stability constraints."""
        pass
