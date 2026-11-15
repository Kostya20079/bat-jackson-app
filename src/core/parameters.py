"""Configuration parameters for the Bat Algorithm."""
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class BatParameters:
    """Hyperparameters configuring Bat Algorithm execution."""
    population_size: int = 30
    max_iterations: int = 100
    f_min: float = 0.0
    f_max: float = 2.0
    a_init: float = 0.9      # A0: Initial loudness
    r_init: float = 0.1      # r0: Initial pulse emission rate
    alpha: float = 0.95      # Loudness decay coefficient
    gamma: float = 0.05      # Pulse rate increase coefficient
    random_seed: int | None = None

    def validate(self) -> None:
        """Validate parameter sanity."""
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.f_min < 0 or self.f_max <= self.f_min:
            raise ValueError("Invalid frequency bounds: requires 0 <= f_min < f_max")
        if not (0.0 < self.a_init <= 1.0):
            raise ValueError("a_init must be in (0, 1]")
        if not (0.0 <= self.r_init <= 1.0):
            raise ValueError("r_init must be in [0, 1]")
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
