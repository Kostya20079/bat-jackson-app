"""Numerical acceleration kernels for Bat Algorithm (NumPy and Numba)."""
import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback dummy decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@njit(fastmath=True)
def update_bat_kinematics(
    positions: np.ndarray,
    velocities: np.ndarray,
    best_position: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    """
    Update velocity and continuous position:
    v_i^{t} = v_i^{t-1} + (x_i^{t-1} - x_best) * f_i
    x_new = x_i^{t-1} + v_i^{t}
    """
    n_bats, n_dim = positions.shape
    new_positions = np.empty_like(positions)
    for i in range(n_bats):
        for d in range(n_dim):
            velocities[i, d] += (positions[i, d] - best_position[d]) * frequencies[i]
            new_positions[i, d] = positions[i, d] + velocities[i, d]
    return new_positions


@njit(fastmath=True)
def project_discrete_bounds(
    positions: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    """Round continuous position to nearest integer and clip to [c_min, c_max]."""
    n_bats, n_dim = positions.shape
    discrete = np.empty((n_bats, n_dim), dtype=np.int64)
    for i in range(n_bats):
        for d in range(n_dim):
            val = int(np.round(positions[i, d]))
            if val < lower_bounds[d]:
                val = lower_bounds[d]
            elif val > upper_bounds[d]:
                val = upper_bounds[d]
            discrete[i, d] = val
    return discrete


@njit(fastmath=True)
def local_random_walk(
    best_position: np.ndarray,
    mean_loudness: float,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    rand_eps: np.ndarray,
) -> np.ndarray:
    """
    Generate local candidate around best solution:
    x_new = x_best + epsilon * average(A)
    """
    n_dim = best_position.shape[0]
    out = np.empty(n_dim, dtype=np.int64)
    for d in range(n_dim):
        step = rand_eps[d] * mean_loudness
        continuous_val = best_position[d] + step
        rounded_val = int(np.round(continuous_val))
        if rounded_val < lower_bounds[d]:
            rounded_val = lower_bounds[d]
        elif rounded_val > upper_bounds[d]:
            rounded_val = upper_bounds[d]
        out[d] = rounded_val
    return out
