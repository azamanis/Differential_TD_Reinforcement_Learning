from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

from .config import HorizonConfig, MertonParams, PolicyParams
from .merton import exact_value, exact_value_coefficient, exact_value_finite, utility


def wealth_grid(low: float, high: float, num: int) -> np.ndarray:
    return np.exp(np.linspace(np.log(low), np.log(high), num))


def evaluate_critic_on_grid(
    critic,
    params: MertonParams,
    policy: PolicyParams,
    low: float,
    high: float,
    num: int,
    device: str = "cpu",
    horizon: HorizonConfig | None = None,
    num_t: int = 21,
) -> dict[str, np.ndarray | float | dict]:
    """
    Evaluate critic against the closed-form value. Reports value-fit error
    (MAE/RMSE/MAPE), derivative-fit error (v_w_mae, v_w_norm), and the
    deterministic HJB residual L^pi V - rho V + U(kappa w).
    """
    if horizon is not None:
        return _evaluate_finite_horizon_critic_on_grid(
            critic=critic,
            params=params,
            policy=policy,
            horizon=horizon,
            low=low,
            high=high,
            num_w=num,
            num_t=num_t,
            device=device,
        )

    grid = wealth_grid(low, high, num)
    w = torch.tensor(grid, dtype=torch.float32, device=device, requires_grad=True)

    V, Vw, Vww = critic.value_and_derivatives(w)
    pred = V.detach().cpu().numpy()
    Vw_np = Vw.detach().cpu().numpy()
    Vww_np = Vww.detach().cpu().numpy()

    truth = exact_value(grid, params, policy)
    abs_err = np.abs(pred - truth)
    rel_err = abs_err / np.maximum(np.abs(truth), 1e-12)

    # Closed-form derivatives:  V(w) = A * w^{1-g} / (1-g)
    # =>  V_w  = A * w^{-g},   V_ww = -g * A * w^{-g-1}
    g = params.gamma
    A = exact_value_coefficient(params, policy)
    Vw_truth = A * np.power(grid, -g)
    Vww_truth = -g * A * np.power(grid, -g - 1.0)
    v_w_abs_err = np.abs(Vw_np - Vw_truth)

    # Deterministic HJB residual at the learned critic:
    #   L^pi V (w) = a w V_w(w) + 0.5 b^2 w^2 V_ww(w)
    #   HJB(w)    = L^pi V (w) - rho V(w) + U(kappa w)
    a = params.r + policy.pi * (params.mu - params.r) - policy.kappa
    b = policy.pi * params.sigma
    LV = a * grid * Vw_np + 0.5 * (b**2) * (grid**2) * Vww_np
    U = utility(policy.kappa * grid, g)
    hjb = LV - params.rho * pred + U

    out: dict[str, np.ndarray | float | dict] = {
        "wealth": grid,
        "pred": pred,
        "truth": truth,
        "Vw_pred": Vw_np,
        "Vw_truth": Vw_truth,
        "Vww_pred": Vww_np,
        "Vww_truth": Vww_truth,
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "mape": float(rel_err.mean()),
        "v_w_mae": float(v_w_abs_err.mean()),
        "v_w_norm": float(np.sqrt(np.mean(Vw_np**2))),
        "v_w_norm_true": float(np.sqrt(np.mean(Vw_truth**2))),
        "hjb_rmse": float(np.sqrt(np.mean(hjb**2))),
        "params": asdict(params),
        "policy": asdict(policy),
    }

    return out


def _evaluate_finite_horizon_critic_on_grid(
    critic,
    params: MertonParams,
    policy: PolicyParams,
    horizon: HorizonConfig,
    low: float,
    high: float,
    num_w: int,
    num_t: int = 21,
    device: str = "cpu",
) -> dict[str, np.ndarray | float | dict]:
    """Evaluate a TimeAwareMLPCritic against the closed-form V(t, W).

    Reports MAE, RMSE, MAPE on a (t, W) tensor product grid. Also returns
    arrays of pred and truth shaped (num_t, num_w) for plotting.
    """
    w_grid = wealth_grid(low, high, num_w)
    t_grid = np.linspace(0.0, horizon.T, num_t)
    TT, WW = np.meshgrid(t_grid, w_grid, indexing="ij")

    w_flat = torch.tensor(WW.reshape(-1), dtype=torch.float32, device=device)
    t_flat = torch.tensor(TT.reshape(-1), dtype=torch.float32, device=device)
    with torch.no_grad():
        V_flat = critic.value(w_flat, t_flat).cpu().numpy()
    pred = V_flat.reshape(TT.shape)

    truth = exact_value_finite(TT, WW, params, policy, horizon)
    abs_err = np.abs(pred - truth)
    rel_err = abs_err / np.maximum(np.abs(truth), 1e-12)

    return {
        "t_grid": t_grid,
        "wealth": w_grid,
        "pred": pred,
        "truth": truth,
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
        "mape": float(rel_err.mean()),
        "params": asdict(params),
        "policy": asdict(policy),
        "horizon": asdict(horizon),
    }
