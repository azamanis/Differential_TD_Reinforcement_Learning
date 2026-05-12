from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch

from .config import HorizonConfig, MertonParams, PolicyParams


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
    drift_term = (
        params.r
        + policy.pi * (params.mu - params.r)
        - policy.kappa
        - 0.5 * g * (policy.pi ** 2) * (params.sigma ** 2)
    )
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
    pi_star = (params.mu - params.r) / (g * params.sigma ** 2)
    sharpe_term = 0.5 * ((params.mu - params.r) ** 2) / (g * params.sigma ** 2)
    kappa_star = (params.rho - (1.0 - g) * (params.r + sharpe_term)) / g
    if kappa_star <= 0.0:
        raise ValueError(
            "The chosen parameters do not produce a positive optimal consumption rate. "
            "Increase rho or modify (mu, r, sigma, gamma)."
        )
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
    return pi * params.sigma


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
    a = params.r + policy.pi * (params.mu - params.r) - policy.kappa
    b = policy.pi * params.sigma
    return wealth * torch.exp((a - 0.5 * b * b) * dt + b * math.sqrt(dt) * noise)


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
    drift_term = (
        params.r
        + policy.pi * (params.mu - params.r)
        - policy.kappa
        - 0.5 * g * (policy.pi ** 2) * (params.sigma ** 2)
    )
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
