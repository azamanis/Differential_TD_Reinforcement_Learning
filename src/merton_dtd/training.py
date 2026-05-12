from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam
from tqdm import trange

from .config import HorizonConfig, MertonParams, PolicyParams, TrainConfig
from .critic import VanillaMLPCritic
from .eval import evaluate_critic_on_grid
from .losses import compute_loss, make_batch
from .merton import terminal_value_fn
from .sampling import sample_log_uniform


def build_critic(
    params: MertonParams,
    device: str = "cpu",
    horizon: HorizonConfig | None = None,
) -> VanillaMLPCritic:
    time_horizon = horizon.T if horizon is not None else None
    return VanillaMLPCritic(params, time_horizon=time_horizon).to(device)


def _evaluate_streaming_critic(
    critic: VanillaMLPCritic,
    params: MertonParams,
    policy: PolicyParams,
    train_cfg: TrainConfig,
    horizon: HorizonConfig | None,
) -> dict[str, Any]:
    return evaluate_critic_on_grid(
        critic=critic,
        params=params,
        policy=policy,
        low=train_cfg.wealth_min,
        high=train_cfg.wealth_max,
        num=train_cfg.eval_points,
        device=train_cfg.device,
        horizon=horizon,
    )


def train_fixed_policy_critic(
    params: MertonParams,
    policy: PolicyParams,
    train_cfg: TrainConfig,
    loss_name: str = "beta_dtd",
    horizon: HorizonConfig | None = None,
    terminal_weight: float = 1.0,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    torch.manual_seed(train_cfg.seed)

    critic = build_critic(params, device=train_cfg.device, horizon=horizon)
    optimizer = Adam(critic.parameters(), lr=train_cfg.learning_rate)

    history: dict[str, list[float]] = {
        "step": [],
        "loss": [],
        "td_mse": [],
        "dtd_mse": [],
        "pinn_mse": [],
        "terminal_mse": [],
        "mae": [],
        "rmse": [],
        "mape": [],
        "v_w_mae": [],
        "v_w_norm": [],
        "hjb_rmse": [],
    }

    wealth = sample_log_uniform(
        batch_size=train_cfg.batch_size,
        low=train_cfg.wealth_min,
        high=train_cfg.wealth_max,
        device=train_cfg.device,
    )
    t = torch.zeros_like(wealth) if horizon is not None else None
    g_fn = terminal_value_fn(params, horizon) if horizon is not None else None

    iterator = trange(train_cfg.num_steps, desc=f"train-{loss_name}", leave=False)
    for step in iterator:
        wealth, wealth_next, reward = make_batch(wealth, params, policy, train_cfg.dt)
        t_next = None
        terminal_mask = None
        terminal_value_next = None
        if horizon is not None:
            t_next = torch.clamp(t + train_cfg.dt, max=horizon.T)
            terminal_mask = t_next >= horizon.T
            if torch.any(terminal_mask):
                assert g_fn is not None
                terminal_value_next = critic.value(wealth_next, t_next).detach()
                terminal_value_next = terminal_value_next.clone()
                terminal_value_next[terminal_mask] = g_fn(
                    wealth_next[terminal_mask]
                ).detach()

        optimizer.zero_grad(set_to_none=True)
        loss, metrics = compute_loss(
            critic=critic,
            wealth=wealth,
            wealth_next=wealth_next,
            reward=reward,
            params=params,
            dt=train_cfg.dt,
            loss_name=loss_name,
            beta=train_cfg.beta,
            t=t,
            t_next=t_next,
            terminal_value_next=terminal_value_next,
            policy=policy,
            num_replicas=train_cfg.num_replicas,
        )
        if horizon is not None:
            assert g_fn is not None
            terminal_wealth = sample_log_uniform(
                batch_size=train_cfg.batch_size,
                low=train_cfg.wealth_min,
                high=train_cfg.wealth_max,
                device=train_cfg.device,
            )
            terminal_time = torch.full_like(terminal_wealth, horizon.T)
            v_terminal = critic.value(terminal_wealth, terminal_time)
            terminal_mse = torch.mean(
                (v_terminal - g_fn(terminal_wealth)).square()
            )
            loss = loss + terminal_weight * terminal_mse
            metrics["loss"] = float(loss.detach().cpu())
            metrics["terminal_mse"] = float(terminal_mse.detach().cpu())
        else:
            metrics["terminal_mse"] = float("nan")

        loss.backward()
        optimizer.step()

        wealth = wealth_next.detach()
        if horizon is not None:
            assert t_next is not None
            assert terminal_mask is not None
            num_resets = int(terminal_mask.sum().item())
            if num_resets > 0:
                wealth = wealth.clone()
                wealth[terminal_mask] = sample_log_uniform(
                    batch_size=num_resets,
                    low=train_cfg.wealth_min,
                    high=train_cfg.wealth_max,
                    device=train_cfg.device,
                )
            t = torch.where(terminal_mask, torch.zeros_like(t_next), t_next).detach()

        if (step % train_cfg.log_every == 0) or (step == train_cfg.num_steps - 1):
            eval_metrics = _evaluate_streaming_critic(
                critic=critic,
                params=params,
                policy=policy,
                train_cfg=train_cfg,
                horizon=horizon,
            )
            history["step"].append(step)
            history["loss"].append(metrics["loss"])
            history["td_mse"].append(metrics["td_mse"])
            history["dtd_mse"].append(metrics["dtd_mse"])
            history["pinn_mse"].append(metrics.get("pinn_mse", float("nan")))
            history["terminal_mse"].append(metrics["terminal_mse"])
            history["mae"].append(float(eval_metrics["mae"]))
            history["rmse"].append(float(eval_metrics["rmse"]))
            history["mape"].append(float(eval_metrics["mape"]))
            history["v_w_mae"].append(float(eval_metrics.get("v_w_mae", float("nan"))))
            history["v_w_norm"].append(float(eval_metrics.get("v_w_norm", float("nan"))))
            history["hjb_rmse"].append(float(eval_metrics.get("hjb_rmse", float("nan"))))
            iterator.set_postfix(loss=f"{metrics['loss']:.3e}", mae=f"{eval_metrics['mae']:.3e}")

    summary = _evaluate_streaming_critic(
        critic=critic,
        params=params,
        policy=policy,
        train_cfg=train_cfg,
        horizon=horizon,
    )

    meta = {
        "params": asdict(params),
        "policy": asdict(policy),
        "train_cfg": asdict(train_cfg),
        "loss_name": loss_name,
    }
    if horizon is not None:
        meta["horizon"] = asdict(horizon)
        meta["terminal_weight"] = terminal_weight
    result = {"history": history, "summary": summary, "meta": meta}
    return critic, result


def save_checkpoint(critic, result: dict[str, Any], out_dir: str | Path) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": critic.state_dict(),
            "result": result,
        },
        out_path / "checkpoint.pt",
    )
