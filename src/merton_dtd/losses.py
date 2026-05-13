from __future__ import annotations

import math
from typing import Literal

import torch

from .config import MertonParams, PolicyParams
from .merton import exact_step, reward_rate

LossName = Literal[
    "td", "td_mean", "dtd", "beta_dtd", "rl_pinn", "dtd_mean", "beta_dtd_mean",
    "naive_dtd", "beta_naive_dtd",
]


def td_mean_residual(
    critic,
    wealth: torch.Tensor,
    params: MertonParams,
    policy: PolicyParams,
    dt: float,
    num_replicas: int,
    t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean-target TD: average ρ_disc·V(W_next^k) over K i.i.d. transitions, then
    take residual against V(W_t). Reduces gradient variance K-fold; fixed point
    unchanged (target is detached and zero-mean-noise — see dTD vs TD asymmetry)."""
    if num_replicas < 2:
        raise ValueError("td_mean_residual requires num_replicas >= 2")
    if t is not None:
        raise NotImplementedError("td_mean_residual does not yet support finite horizon")

    V = critic.value(wealth)
    reward_step = reward_rate(wealth, params, policy) * dt
    rho_disc = math.exp(-params.rho * dt)

    V_next_sum = torch.zeros_like(wealth)
    with torch.no_grad():
        for _ in range(num_replicas):
            noise = torch.randn_like(wealth)
            wealth_next = exact_step(wealth, params, policy, dt, noise)
            V_next_sum = V_next_sum + critic.value(wealth_next)
    V_next_mean = V_next_sum / num_replicas
    return reward_step + rho_disc * V_next_mean - V


def make_batch(
    wealth: torch.Tensor,
    params: MertonParams,
    policy: PolicyParams,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample one exact step under the fixed constant policy.

    Returns
    -------
    wealth      : W_t
    wealth_next : W_{t+dt}
    reward      : instantaneous reward rate r_t = U(c_t)
    """
    noise = torch.randn_like(wealth)
    wealth_next = exact_step(wealth, params, policy, dt, noise)
    reward = reward_rate(wealth, params, policy)  # reward rate, not yet multiplied by dt
    return wealth, wealth_next, reward


def td_residual(
    critic,
    wealth: torch.Tensor,
    wealth_next: torch.Tensor,
    reward: torch.Tensor,
    params: MertonParams,
    dt: float,
    t: torch.Tensor | None = None,
    t_next: torch.Tensor | None = None,
    terminal_value_next: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Standard one-step TD residual:
        delta_TD = reward_step + rho_disc * V(W_{t+dt}) - V(W_t)

    where
        reward_step = reward_rate * dt
        rho_disc  = exp(-rho * dt)
    """

    # rho is the continuous discount factor
    # in the paper they use gamma (but in the merton context gamma is the risk aversion param)
    V = critic.value(wealth, t)
    reward_step = reward * dt
    rho_disc = math.exp(-params.rho * dt)

    with torch.no_grad():
        if terminal_value_next is None:
            V_next = critic.value(wealth_next, t_next)
        else:
            V_next = terminal_value_next

    return reward_step + rho_disc * V_next - V


def dtd_prediction_and_target(
    critic,
    wealth: torch.Tensor,
    wealth_next: torch.Tensor,
    reward: torch.Tensor,
    params: MertonParams,
    dt: float,
    t: torch.Tensor | None = None,
    t_next: torch.Tensor | None = None,
    terminal_value_next: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Practical dTD decomposition matching the paper's useful form.

    For one-dimensional state W, define:
        prediction = ΔW * V_w(W_t) + 0.5 * (ΔW)^2 * V_ww(W_t)

        target = - reward_step + rho * dt * V(W_{t+dt})

    so the dTD error is:
        prediction - target

    Notes
    -----
    - reward passed in is an instantaneous reward rate, so we multiply by dt here.
    - V(W_{t+dt}) is treated as a target (detached / no-grad).
    - This is deliberately different from the old "raw residual" implementation.
    """
    if t is None:
        _, Vw, Vww = critic.value_and_derivatives(wealth)
        time_prediction = 0.0
    else:
        _, Vt, Vw, Vww = critic.value_and_derivatives(wealth, t)
        time_prediction = dt * Vt
    delta_w = wealth_next - wealth
    reward_step = reward * dt

    # Prediction side: derivative/local-dynamics terms at the current state
    # Note: time prediction is zero for infinite horizon
    prediction = time_prediction + delta_w * Vw + 0.5 * delta_w.square() * Vww

    # Target side: value/reward terms
    with torch.no_grad(): # we don't backprop througb target
        if terminal_value_next is None:
            V_next = critic.value(wealth_next, t_next)
        else:
            V_next = terminal_value_next

    # Since rho_disc = exp(-rho dt), we have -log(rho_disc) = rho dt
    # Reward step is Utility() * dt
    target = -reward_step + (params.rho * dt) * V_next
    
    return prediction, target


def dtd_residual(
    critic,
    wealth: torch.Tensor,
    wealth_next: torch.Tensor,
    reward: torch.Tensor,
    params: MertonParams,
    dt: float,
    t: torch.Tensor | None = None,
    t_next: torch.Tensor | None = None,
    terminal_value_next: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Practical dTD error:
        delta_dTD = prediction - target
    """
    pred, target = dtd_prediction_and_target(
        critic=critic,
        wealth=wealth,
        wealth_next=wealth_next,
        reward=reward,
        params=params,
        dt=dt,
        t=t,
        t_next=t_next,
        terminal_value_next=terminal_value_next,
    )
    return pred - target


def naive_dtd_residual(
    critic,
    wealth: torch.Tensor,
    wealth_next: torch.Tensor,
    reward: torch.Tensor,
    params: MertonParams,
    dt: float,
    t: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Naive-dTD (Table 1 of Settai et al.), discretization-friendly form:

        prediction = (rho * dt) * V(W_t)             [gradient flows]
        target     = r*dt + ΔW V_w(W_t) + 0.5 ΔW² V_ww(W_t)   [detached]

    The noise-coupled derivative terms live in the *target* (and are detached),
    so the prediction is noise-free and the semi-gradient is unbiased — fixed
    point coincides with the HJB equation. Cost: variance of the target scales
    as 1/Δt, so per-sample noise is much larger than for the rearranged dTD.
    """
    if t is None:
        V = critic.value(wealth)
        _, Vw, Vww = critic.value_and_derivatives(wealth)
        time_target = 0.0
    else:
        V = critic.value(wealth, t)
        _, Vt, Vw, Vww = critic.value_and_derivatives(wealth, t)
        time_target = (dt * Vt).detach()

    Vw = Vw.detach()
    Vww = Vww.detach()
    delta = wealth_next - wealth
    reward_step = reward * dt

    prediction = (params.rho * dt) * V
    target = reward_step + time_target + delta * Vw + 0.5 * delta.square() * Vww
    return prediction - target


def dtd_mean_residual(
    critic,
    wealth: torch.Tensor,
    params: MertonParams,
    policy: PolicyParams,
    dt: float,
    num_replicas: int,
    t: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Mean-residual dTD estimator: draw K i.i.d. transitions from the same W_t,
    average the per-replica residual, then square.

    Per-replica residual:
        delta_k = ΔW_k V_w + 0.5 ΔW_k² V_ww + r dt − ρ dt V(W + ΔW_k).

    Mean over k of delta_k has population limit Δt · HJB(W) and zero population
    contribution from the V_w-times-noise term, removing the BRM regularizer
    on V_w² that biases the per-sample MSE estimator.
    """
    from .merton import exact_step, reward_rate

    if num_replicas < 2:
        raise ValueError("dtd_mean_residual requires num_replicas >= 2")
    if t is not None:
        raise NotImplementedError("dtd_mean_residual does not yet support finite horizon")

    _, Vw, Vww = critic.value_and_derivatives(wealth)
    time_term = 0.0

    reward = reward_rate(wealth, params, policy)
    reward_step = reward * dt

    pred_sum = torch.zeros_like(wealth)
    V_next_sum = torch.zeros_like(wealth)
    for _ in range(num_replicas):
        noise = torch.randn_like(wealth)
        wealth_next = exact_step(wealth, params, policy, dt, noise)
        delta = wealth_next - wealth
        pred_sum = pred_sum + delta * Vw + 0.5 * delta.square() * Vww
        with torch.no_grad():
            V_next_sum = V_next_sum + critic.value(wealth_next)

    pred_mean = time_term + pred_sum / num_replicas
    V_next_mean = V_next_sum / num_replicas
    target_mean = -reward_step + (params.rho * dt) * V_next_mean
    return pred_mean - target_mean


def rl_pinn_residual(
    critic,
    wealth: torch.Tensor,
    wealth_next: torch.Tensor,
    reward: torch.Tensor,
    params: MertonParams,
    dt: float,
    t: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    PINN-style Bellman residual from the RL_PINN note.

        R_theta(W_t) = V_theta(W_t)
                       - (1/rho) [ U(c_t)
                                   + (DeltaW/dt)        * V_w(W_t)
                                   + 0.5 * (DeltaW)^2/dt * V_ww(W_t) ]

    For finite horizon (critic built with `time_horizon`), pass `t` and the
    Ito expansion of V(t+dt, W_{t+dt}) contributes a V_t term:

        R = V(t,W) - (1/rho)[ U + V_t + (DeltaW/dt) V_w + 0.5 (DeltaW)^2/dt V_ww ].
    """
    delta = wealth_next - wealth
    if t is None:
        V, Vw, Vww = critic.value_and_derivatives(wealth)
        rhs = reward + (delta / dt) * Vw + 0.5 * (delta * delta / dt) * Vww
    else:
        V, Vt, Vw, Vww = critic.value_and_derivatives(wealth, t)
        rhs = reward + Vt + (delta / dt) * Vw + 0.5 * (delta * delta / dt) * Vww
    return V - rhs / params.rho


def compute_loss(
    critic,
    wealth: torch.Tensor,
    wealth_next: torch.Tensor,
    reward: torch.Tensor,
    params: MertonParams,
    dt: float,
    loss_name: LossName,
    beta: float = 0.5,
    t: torch.Tensor | None = None,
    t_next: torch.Tensor | None = None,
    terminal_value_next: torch.Tensor | None = None,
    policy: "PolicyParams | None" = None,
    num_replicas: int = 1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Losses:
      - td       : mean(delta_TD^2)
      - dtd      : mean(delta_dTD^2)
      - beta_dtd : (1-beta) * mean(delta_TD^2) + beta * mean(delta_dTD^2)
    """
    td = td_residual(
        critic=critic,
        wealth=wealth,
        wealth_next=wealth_next,
        reward=reward,
        params=params,
        dt=dt,
        t=t,
        t_next=t_next,
        terminal_value_next=terminal_value_next,
    )

    dtd = dtd_residual(
        critic=critic,
        wealth=wealth,
        wealth_next=wealth_next,
        reward=reward,
        params=params,
        dt=dt,
        t=t,
        t_next=t_next,
        terminal_value_next=terminal_value_next,
    )

    td_mse = torch.mean(td.square())
    dtd_mse = torch.mean(dtd.square())
    pinn_mse = torch.tensor(float("nan"))
    dtd_mean_mse = torch.tensor(float("nan"))
    naive_dtd_mse = torch.tensor(float("nan"))

    if loss_name == "rl_pinn":
        pinn = rl_pinn_residual(
            critic=critic,
            wealth=wealth,
            wealth_next=wealth_next,
            reward=reward,
            params=params,
            dt=dt,
            t=t,
        )
        pinn_mse = torch.mean(pinn.square())
        loss = pinn_mse
    elif loss_name == "td":
        loss = td_mse
    elif loss_name == "td_mean":
        if policy is None:
            raise ValueError(f"loss_name={loss_name} requires policy")
        td_mean = td_mean_residual(
            critic=critic,
            wealth=wealth,
            params=params,
            policy=policy,
            dt=dt,
            num_replicas=num_replicas,
            t=t,
        )
        loss = torch.mean(td_mean.square())
    elif loss_name == "dtd":
        loss = dtd_mse
    elif loss_name == "beta_dtd":
        loss = (1.0 - beta) * td_mse + beta * dtd_mse
    elif loss_name in ("naive_dtd", "beta_naive_dtd"):
        naive = naive_dtd_residual(
            critic=critic,
            wealth=wealth,
            wealth_next=wealth_next,
            reward=reward,
            params=params,
            dt=dt,
            t=t,
        )
        naive_dtd_mse = torch.mean(naive.square())
        if loss_name == "naive_dtd":
            loss = naive_dtd_mse
        else:
            loss = (1.0 - beta) * td_mse + beta * naive_dtd_mse
    elif loss_name in ("dtd_mean", "beta_dtd_mean"):
        if policy is None:
            raise ValueError(f"loss_name={loss_name} requires policy")
        dtd_mean = dtd_mean_residual(
            critic=critic,
            wealth=wealth,
            params=params,
            policy=policy,
            dt=dt,
            num_replicas=num_replicas,
            t=t,
        )
        dtd_mean_mse = torch.mean(dtd_mean.square())
        if loss_name == "dtd_mean":
            loss = dtd_mean_mse
        else:
            loss = (1.0 - beta) * td_mse + beta * dtd_mean_mse
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")

    metrics = {
        "td_mse": float(td_mse.detach().cpu()),
        "dtd_mse": float(dtd_mse.detach().cpu()),
        "pinn_mse": float(pinn_mse.detach().cpu()),
        "dtd_mean_mse": float(dtd_mean_mse.detach().cpu()),
        "naive_dtd_mse": float(naive_dtd_mse.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }
    return loss, metrics
