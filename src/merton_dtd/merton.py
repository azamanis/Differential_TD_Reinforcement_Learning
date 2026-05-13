from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch

from .config import HorizonConfig, MertonParams, PolicyParams


def _as_vector(x, name: str) -> np.ndarray:
    if isinstance(x, np.ndarray):
        vec = x.astype(float)
    elif isinstance(x, (list, tuple)):
        vec = np.asarray(x, dtype=float)
    else:
        vec = np.asarray([float(x)], dtype=float)
    if vec.ndim != 1:
        raise ValueError(f"{name} must be a scalar or 1D vector")
    return vec


def _as_covariance(sigma, n: int) -> np.ndarray:
    if isinstance(sigma, np.ndarray):
        mat = sigma.astype(float)
    elif isinstance(sigma, (list, tuple)):
        mat = np.asarray(sigma, dtype=float)
    else:
        sigma_val = float(sigma)
        if n == 1:
            return np.asarray([[sigma_val ** 2]], dtype=float)
        return np.diag(np.full(n, sigma_val ** 2))

    if mat.ndim == 1:
        if mat.size != n:
            raise ValueError("sigma vector length must match number of assets")
        return np.diag(mat ** 2)
    if mat.ndim == 2:
        if mat.shape != (n, n):
            raise ValueError("sigma covariance must be shape (n, n)")
        return mat
    raise ValueError("sigma must be scalar, vector, or covariance matrix")


def _infer_num_assets(params: MertonParams, policy: PolicyParams) -> int:
    sizes: list[int] = []

    if isinstance(params.mu, (list, tuple, np.ndarray)):
        sizes.append(int(np.asarray(params.mu).shape[0]))
    if isinstance(params.sigma, (list, tuple, np.ndarray)):
        sigma_arr = np.asarray(params.sigma)
        if sigma_arr.ndim == 1:
            sizes.append(int(sigma_arr.shape[0]))
        elif sigma_arr.ndim == 2:
            sizes.append(int(sigma_arr.shape[0]))
    if isinstance(policy.pi, (list, tuple, np.ndarray)):
        sizes.append(int(np.asarray(policy.pi).shape[0]))

    if not sizes:
        return 1
    if len(set(sizes)) != 1:
        raise ValueError("Inconsistent asset dimensions among mu, sigma, and pi")
    return sizes[0]


def portfolio_drift_and_variance(params: MertonParams, policy: PolicyParams) -> tuple[float, float]:
    n = _infer_num_assets(params, policy)
    mu = _as_vector(params.mu, "mu")
    if mu.size == 1 and n > 1:
        mu = np.full(n, float(mu.item()))

    pi = _as_vector(policy.pi, "pi")
    if pi.size == 1 and n > 1:
        pi = np.full(n, float(pi.item()))

    if mu.size != n or pi.size != n:
        raise ValueError("mu and pi must match the number of assets")

    sigma = _as_covariance(params.sigma, n)
    mu_excess = mu - params.r
    drift_excess = float(pi @ mu_excess)
    variance = float(pi @ sigma @ pi)
    return drift_excess, variance


def utility(consumption: torch.Tensor | np.ndarray | float, gamma: float):
    """CRRA utility U(c) = c^(1-gamma)/(1-gamma), gamma != 1."""
    if isinstance(consumption, np.ndarray):
        c = np.maximum(consumption, 1e-12)
        return np.power(c, 1.0 - gamma) / (1.0 - gamma)
    if isinstance(consumption, torch.Tensor):
        c = torch.clamp(consumption, min=1e-12)
        return torch.pow(c, 1.0 - gamma) / (1.0 - gamma)
    c = max(float(consumption), 1e-12)
    return c ** (1.0 - gamma) / (1.0 - gamma)


def exact_value_coefficient(params: MertonParams, policy: PolicyParams) -> float:
    """Closed-form value coefficient for a fixed constant policy.

    For a constant policy (pi, kappa), the exact value is
        V(w) = A * w^(1-gamma) / (1-gamma),
    with
        A = kappa^(1-gamma) / D,
        D = rho - (1-gamma) * [r + pi (mu-r) - kappa - 0.5 * gamma * pi^2 * sigma^2].

    D must be strictly positive for the infinite-horizon discounted value to be finite.
    """
    g = params.gamma
    drift_excess, variance = portfolio_drift_and_variance(params, policy)
    drift_term = params.r + drift_excess - policy.kappa - 0.5 * g * variance
    denom = params.rho - (1.0 - g) * drift_term
    if denom <= 0.0:
        raise ValueError(
            "This policy is not admissible for the chosen parameters: the infinite-horizon "
            "value is not finite because the denominator is non-positive."
        )
    return policy.kappa ** (1.0 - g) / denom


def exact_value(wealth: torch.Tensor | np.ndarray | float, params: MertonParams, policy: PolicyParams):
    coeff = exact_value_coefficient(params, policy)
    g = params.gamma
    if isinstance(wealth, np.ndarray):
        w = np.maximum(wealth, 1e-12)
        return coeff * np.power(w, 1.0 - g) / (1.0 - g)
    if isinstance(wealth, torch.Tensor):
        w = torch.clamp(wealth, min=1e-12)
        return coeff * torch.pow(w, 1.0 - g) / (1.0 - g)
    w = max(float(wealth), 1e-12)
    return coeff * (w ** (1.0 - g)) / (1.0 - g)


def optimal_policy_closed_form(params: MertonParams) -> Tuple[PolicyParams, float]:
    """Closed-form optimal constant policy and its value coefficient.

    The infinite-horizon CRRA Merton solution is:
        pi* = (mu-r) / (gamma sigma^2)
        kappa* = [rho - (1-gamma)(r + 0.5 * (mu-r)^2 / (gamma sigma^2))] / gamma
    The policy is admissible only if kappa* > 0.
    """
    g = params.gamma
    n = _infer_num_assets(params, PolicyParams(pi=0.0, kappa=1.0))
    mu = _as_vector(params.mu, "mu")
    if mu.size == 1 and n > 1:
        mu = np.full(n, float(mu.item()))
    sigma = _as_covariance(params.sigma, n)

    mu_excess = mu - params.r
    pi_star_vec = np.linalg.solve(sigma, mu_excess) / g
    sharpe_term = 0.5 * float(mu_excess @ np.linalg.solve(sigma, mu_excess)) / g
    kappa_star = (params.rho - (1.0 - g) * (params.r + sharpe_term)) / g
    if kappa_star <= 0.0:
        raise ValueError(
            "The chosen parameters do not produce a positive optimal consumption rate. "
            "Increase rho or modify (mu, r, sigma, gamma)."
        )
    pi_star = float(pi_star_vec[0]) if pi_star_vec.size == 1 else pi_star_vec.tolist()
    policy = PolicyParams(pi=pi_star, kappa=kappa_star)
    coeff = exact_value_coefficient(params, policy)
    return policy, coeff


def is_policy_admissible(params: MertonParams, policy: PolicyParams) -> bool:
    try:
        _ = exact_value_coefficient(params, policy)
        return True
    except ValueError:
        return False


def risky_weight_to_sharpe_ratio(params: MertonParams, pi: float) -> float:
    _drift_excess, variance = portfolio_drift_and_variance(params, PolicyParams(pi=pi, kappa=1.0))
    return math.sqrt(variance)


def exact_step(
    wealth: torch.Tensor,
    params: MertonParams,
    policy: PolicyParams,
    dt: float,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """One exact step under a constant policy.

    Since dW/W is affine in dt and dB under a constant policy, the wealth process is geometric:
        W_{t+dt} = W_t exp((a - 0.5 b^2)dt + b sqrt(dt) eps)
    with a = r + pi(mu-r) - kappa and b = pi sigma.
    """
    if noise is None:
        noise = torch.randn_like(wealth)
    drift_excess, variance = portfolio_drift_and_variance(params, policy)
    a = params.r + drift_excess - policy.kappa
    b = math.sqrt(max(variance, 0.0))
    return wealth * torch.exp((a - 0.5 * variance) * dt + b * math.sqrt(dt) * noise)


def reward_rate(wealth: torch.Tensor, params: MertonParams, policy: PolicyParams) -> torch.Tensor:
    consumption = policy.kappa * wealth
    return utility(consumption, params.gamma)


def exact_step_tensor(
    wealth: torch.Tensor,
    params: MertonParams,
    pi: torch.Tensor,
    kappa: torch.Tensor,
    dt: float,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample exact step under a state-dependent policy.

    Same closed-form GBM update as `exact_step`, but `pi` and `kappa` are
    tensors broadcastable to `wealth`. Valid because (pi, kappa) is constant
    over the [t, t+dt] interval — only sample-dependent.
    """
    if noise is None:
        noise = torch.randn_like(wealth)
    if not isinstance(params.mu, (int, float)):
        raise ValueError("exact_step_tensor only supports scalar mu/sigma")
    if not isinstance(params.sigma, (int, float)):
        raise ValueError("exact_step_tensor only supports scalar mu/sigma")
    a = params.r + pi * (params.mu - params.r) - kappa
    b = pi * params.sigma
    return wealth * torch.exp((a - 0.5 * b * b) * dt + b * math.sqrt(dt) * noise)


def reward_rate_tensor(
    wealth: torch.Tensor, params: MertonParams, kappa: torch.Tensor
) -> torch.Tensor:
    consumption = kappa * wealth
    return utility(consumption, params.gamma)


def _finite_horizon_D(params: MertonParams, policy: PolicyParams) -> float:
    """The constant D appearing in the A(t) ODE.

    A(t) satisfies dA/dt = D * A - kappa^{1-gamma}, A(T) = terminal_coef,
    where
        D = rho - (1-gamma) * (r + pi (mu-r) - kappa - 0.5 gamma pi^2 sigma^2).
    Admissibility requires D > 0.
    """
    g = params.gamma
    drift_excess, variance = portfolio_drift_and_variance(params, policy)
    drift_term = params.r + drift_excess - policy.kappa - 0.5 * g * variance
    D = params.rho - (1.0 - g) * drift_term
    if D <= 0.0:
        raise ValueError(
            "Finite-horizon Merton policy is not admissible: D <= 0."
        )
    return float(D)


def finite_horizon_A(
    t: torch.Tensor | np.ndarray | float,
    params: MertonParams,
    policy: PolicyParams,
    horizon: HorizonConfig,
):
    """Closed-form A(t) for the finite-horizon CRRA Merton problem.

    Solving dA/dt = D A - kappa^{1-gamma} backward from A(T) = terminal_coef:
        A(t) = terminal_coef * exp(-D (T-t))
             + (kappa^{1-gamma} / D) * (1 - exp(-D (T-t))).

    As T - t -> infinity, A(t) -> kappa^{1-gamma}/D, the infinite-horizon
    coefficient, regardless of `terminal_coef`.
    """
    g = params.gamma
    D = _finite_horizon_D(params, policy)
    A_inf = (policy.kappa ** (1.0 - g)) / D
    A_T = horizon.terminal_coef

    if isinstance(t, np.ndarray):
        tau = horizon.T - t
        return A_T * np.exp(-D * tau) + A_inf * (1.0 - np.exp(-D * tau))
    if isinstance(t, torch.Tensor):
        tau = horizon.T - t
        return A_T * torch.exp(-D * tau) + A_inf * (1.0 - torch.exp(-D * tau))
    tau = horizon.T - float(t)
    return A_T * math.exp(-D * tau) + A_inf * (1.0 - math.exp(-D * tau))


def exact_value_finite(
    t: torch.Tensor | np.ndarray | float,
    wealth: torch.Tensor | np.ndarray | float,
    params: MertonParams,
    policy: PolicyParams,
    horizon: HorizonConfig,
):
    """Closed-form V(t, W) for the finite-horizon CRRA Merton problem."""
    A = finite_horizon_A(t, params, policy, horizon)
    g = params.gamma
    if isinstance(wealth, np.ndarray):
        w = np.maximum(wealth, 1e-12)
        return A * np.power(w, 1.0 - g) / (1.0 - g)
    if isinstance(wealth, torch.Tensor):
        w = torch.clamp(wealth, min=1e-12)
        return A * torch.pow(w, 1.0 - g) / (1.0 - g)
    w = max(float(wealth), 1e-12)
    return A * (w ** (1.0 - g)) / (1.0 - g)


def terminal_value_fn(
    params: MertonParams,
    horizon: HorizonConfig,
):
    """Returns the terminal value function g(W) = terminal_coef * W^{1-gamma}/(1-gamma)."""
    g = params.gamma
    A_T = horizon.terminal_coef

    def g_fn(wealth: torch.Tensor) -> torch.Tensor:
        w = torch.clamp(wealth, min=1e-12)
        return A_T * torch.pow(w, 1.0 - g) / (1.0 - g)

    return g_fn
