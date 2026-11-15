"""
Adapter for Open Jackson Queueing Network evaluation.
Designed for easy substitution when the Jackson Network module is delivered by teammate.
Follows equations from sections 3, 5, 6 of the academic specification.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from src.evaluation.base import BaseEvaluator


@dataclass
class JacksonNetworkConfig:
    """Input configuration for Open Jackson Queueing Network."""
    num_nodes: int
    gamma: np.ndarray       # External arrival rates (gamma_i)
    mu: np.ndarray          # Service rates per channel (mu_i)
    P: np.ndarray           # Transition probability matrix (p_ij)
    c_min: np.ndarray       # Lower bounds of channels
    c_max: np.ndarray       # Upper bounds of channels
    alpha: float = 1.0      # Weight for mean system response time W_sieci
    beta: float = 0.5       # Weight for total channel cost C(c)
    cost_weights: Optional[np.ndarray] = None  # k_i: unit cost per channel in node i

    def __post_init__(self):
        self.gamma = np.asarray(self.gamma, dtype=np.float64)
        self.mu = np.asarray(self.mu, dtype=np.float64)
        self.P = np.asarray(self.P, dtype=np.float64)
        self.c_min = np.asarray(self.c_min, dtype=np.int64)
        self.c_max = np.asarray(self.c_max, dtype=np.int64)
        if self.cost_weights is None:
            self.cost_weights = np.ones(self.num_nodes, dtype=np.float64)
        else:
            self.cost_weights = np.asarray(self.cost_weights, dtype=np.float64)

        self.validate()

    def validate(self) -> None:
        """Validate input parameters according to Jackson model requirements."""
        n = self.num_nodes
        if len(self.gamma) != n or len(self.mu) != n or self.P.shape != (n, n):
            raise ValueError(f"Dimensions mismatch with num_nodes={n}")
        if np.any(self.gamma < 0):
            raise ValueError("External arrival rates gamma must be non-negative")
        if np.any(self.mu <= 0):
            raise ValueError("Service rates mu must be strictly positive")
        if np.any(self.P < 0) or np.any(self.P > 1):
            raise ValueError("Probabilities in P must be in [0, 1]")
        row_sums = np.sum(self.P, axis=1)
        if np.any(row_sums > 1.0 + 1e-7):
            raise ValueError("Sum of transition probabilities in row cannot exceed 1")


@dataclass
class JacksonNetworkCharacteristics:
    """Network performance characteristics calculated for a specific channel configuration c."""
    lambdas: np.ndarray     # Total arrival rates lambda_i
    rhos: np.ndarray        # Channel utilizations rho_i
    p0: np.ndarray          # Idle probabilities P_0,i
    p_wait: np.ndarray      # Erlang C delay probabilities P_wait,i
    lq: np.ndarray          # Average queue lengths L_q,i
    wq: np.ndarray          # Average wait times W_q,i
    w: np.ndarray           # Average node response times W_i
    l: np.ndarray           # Average node customers L_i
    l_network: float        # Total average customers L_sieci
    gamma_total: float      # Total external arrivals Gamma
    w_network: float        # Total average network time W_sieci
    channel_cost: float     # C(c) = sum(k_i * c_i)
    objective_value: float  # f(c) = alpha * W_sieci + beta * C(c)
    is_stable: bool         # All rho_i < 1


class JacksonNetworkEvaluator(BaseEvaluator):
    """
    Evaluator connecting Bat Algorithm with Open Jackson Queueing Network characteristics.
    """

    def __init__(self, config: JacksonNetworkConfig, penalty_value: float = 1e6):
        self.config = config
        self.penalty_value = penalty_value
        self.lambdas = self._compute_arrival_rates()

    def _compute_arrival_rates(self) -> np.ndarray:
        """
        Solve traffic equations:
        (I - P^T) * lambda = gamma
        """
        n = self.config.num_nodes
        I = np.eye(n, dtype=np.float64)
        A = I - self.config.P.T
        try:
            lambdas = np.linalg.solve(A, self.config.gamma)
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Singular traffic matrix (I - P^T): {e}")

        if np.any(lambdas < -1e-9):
            raise ValueError("Computed negative arrival rates lambda in network")
        return np.maximum(lambdas, 0.0)

    def calculate_characteristics(self, c: np.ndarray) -> JacksonNetworkCharacteristics:
        """
        Calculate complete M/M/c Jackson network characteristics for channel vector c.
        Implements section 3.4 & 3.5 of document.
        """
        c = np.asarray(c, dtype=np.int64)
        n = self.config.num_nodes
        mu = self.config.mu
        lambdas = self.lambdas

        rhos = lambdas / (c * mu)
        is_stable = bool(np.all(rhos < 1.0))

        channel_cost = float(np.sum(self.config.cost_weights * c))

        if not is_stable:
            # Unstable configuration
            return JacksonNetworkCharacteristics(
                lambdas=lambdas,
                rhos=rhos,
                p0=np.zeros(n),
                p_wait=np.ones(n),
                lq=np.full(n, np.inf),
                wq=np.full(n, np.inf),
                w=np.full(n, np.inf),
                l=np.full(n, np.inf),
                l_network=np.inf,
                gamma_total=float(np.sum(self.config.gamma)),
                w_network=np.inf,
                channel_cost=channel_cost,
                objective_value=self.penalty_value,
                is_stable=False,
            )

        # M/M/c Erlang C calculations per node
        p0 = np.empty(n, dtype=np.float64)
        p_wait = np.empty(n, dtype=np.float64)
        lq = np.empty(n, dtype=np.float64)
        wq = np.empty(n, dtype=np.float64)
        w = np.empty(n, dtype=np.float64)
        l = np.empty(n, dtype=np.float64)

        for i in range(n):
            c_i = c[i]
            lam_i = lambdas[i]
            mu_i = mu[i]
            rho_i = rhos[i]

            if lam_i == 0.0:
                p0[i] = 1.0
                p_wait[i] = 0.0
                lq[i] = 0.0
                wq[i] = 0.0
                w[i] = 1.0 / mu_i
                l[i] = 0.0
                continue

            a_i = lam_i / mu_i

            # Sum_k=0^{c_i-1} a_i^k / k!
            sum_terms = sum((a_i ** k) / math.factorial(k) for k in range(c_i))
            tail_term = (a_i ** c_i) / (math.factorial(c_i) * (1.0 - rho_i))
            p0_i = 1.0 / (sum_terms + tail_term)
            p0[i] = p0_i

            # Erlang C: P_wait,i
            p_wait_i = tail_term * p0_i
            p_wait[i] = p_wait_i

            # Lq_i = P_wait * rho / (1 - rho)
            lq_i = p_wait_i * rho_i / (1.0 - rho_i)
            lq[i] = lq_i

            # Wq_i = Lq_i / lambda_i
            wq_i = lq_i / lam_i
            wq[i] = wq_i

            # W_i = Wq_i + 1/mu_i
            w_i = wq_i + (1.0 / mu_i)
            w[i] = w_i

            # L_i = lambda_i * W_i
            l[i] = lam_i * w_i

        l_network = float(np.sum(l))
        gamma_total = float(np.sum(self.config.gamma))

        if gamma_total > 0:
            w_network = l_network / gamma_total
        else:
            w_network = 0.0

        objective_value = (
            self.config.alpha * w_network + self.config.beta * channel_cost
        )

        return JacksonNetworkCharacteristics(
            lambdas=lambdas,
            rhos=rhos,
            p0=p0,
            p_wait=p_wait,
            lq=lq,
            wq=wq,
            w=w,
            l=l,
            l_network=l_network,
            gamma_total=gamma_total,
            w_network=w_network,
            channel_cost=channel_cost,
            objective_value=objective_value,
            is_stable=True,
        )

    def evaluate(self, solution: np.ndarray) -> float:
        """
        Evaluate objective function:
        f(c) = alpha * W_sieci + beta * C(c)
        Unstable configurations receive progressive penalty.
        """
        c = np.asarray(solution, dtype=np.int64)
        rhos = self.lambdas / (c * self.config.mu)

        if np.any(rhos >= 1.0):
            # Penalty proportional to excess utilization
            violation = np.sum(np.maximum(0.0, rhos - 0.99))
            return self.penalty_value + 1e4 * float(violation)

        chars = self.calculate_characteristics(c)
        return chars.objective_value

    def is_feasible(self, solution: np.ndarray) -> bool:
        """Check stability condition: all rho_i < 1.0."""
        c = np.asarray(solution, dtype=np.int64)
        if np.any(c <= 0):
            return False
        rhos = self.lambdas / (c * self.config.mu)
        return bool(np.all(rhos < 1.0))
