"""
Initialize the critic at (a supervised fit to) the closed-form V^pi and then
run pure dTD.

Outputs the standard training history dict + checkpoint to --out-dir, plus
a small "pretrain vs dtd" trajectory plot.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from tqdm import trange

from merton_dtd.config import MertonParams, PolicyParams, TrainConfig
from merton_dtd.eval import evaluate_critic_on_grid, wealth_grid
from merton_dtd.losses import compute_loss, make_batch
from merton_dtd.merton import exact_value
from merton_dtd.plotting import plot_training_curves, plot_value_fit
from merton_dtd.sampling import sample_log_uniform
from merton_dtd.training import build_critic, save_checkpoint


def supervised_pretrain(
    critic,
    params: MertonParams,
    policy: PolicyParams,
    train_cfg: TrainConfig,
    num_steps: int,
    target_mae: float,
) -> dict:
    """MSE-supervise the critic on the closed-form V^pi over a fixed dense
    log-uniform grid (full-batch gradient — no sampling noise). Stops when
    MAE drops below target_mae or num_steps elapses."""
    # Dense fixed grid of (W, V^π(W)) — no stochastic sampling so the optimizer
    # converges to floating-point precision (modulo NN expressivity).
    num_pts = max(4 * train_cfg.eval_points, 2048)
    grid_np = wealth_grid(train_cfg.wealth_min, train_cfg.wealth_max, num_pts)
    w_grid = torch.tensor(grid_np, dtype=torch.float32, device=train_cfg.device)
    target = torch.tensor(
        exact_value(grid_np, params, policy), dtype=torch.float32, device=train_cfg.device
    )

    optimizer = Adam(critic.parameters(), lr=train_cfg.learning_rate)
    history = {"step": [], "pretrain_mse": [], "mae": [], "v_w_mae": []}

    iterator = trange(num_steps, desc="pretrain", leave=False)
    for step in iterator:
        pred = critic.value(w_grid)
        loss = (pred - target).pow(2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % train_cfg.log_every == 0 or step == num_steps - 1:
            ev = evaluate_critic_on_grid(
                critic=critic,
                params=params,
                policy=policy,
                low=train_cfg.wealth_min,
                high=train_cfg.wealth_max,
                num=train_cfg.eval_points,
                device=train_cfg.device,
            )
            history["step"].append(step)
            history["pretrain_mse"].append(float(loss.detach().cpu()))
            history["mae"].append(float(ev["mae"]))
            history["v_w_mae"].append(float(ev["v_w_mae"]))
            iterator.set_postfix(mse=f"{loss.detach().cpu().item():.2e}", mae=f"{ev['mae']:.2e}")
            if ev["mae"] < target_mae:
                break
    return history


def run_dtd_from(
    critic,
    params: MertonParams,
    policy: PolicyParams,
    train_cfg: TrainConfig,
    loss_name: str,
) -> dict:
    optimizer = Adam(critic.parameters(), lr=train_cfg.learning_rate)
    history: dict[str, list[float]] = {
        "step": [], "loss": [], "mae": [], "rmse": [], "v_w_mae": [], "v_w_norm": [],
        "hjb_rmse": [], "dtd_mse": [], "td_mse": [],
    }
    wealth = sample_log_uniform(
        batch_size=train_cfg.batch_size,
        low=train_cfg.wealth_min,
        high=train_cfg.wealth_max,
        device=train_cfg.device,
    )

    iterator = trange(train_cfg.num_steps, desc=f"train-{loss_name}", leave=False)
    for step in iterator:
        wealth, wealth_next, reward = make_batch(wealth, params, policy, train_cfg.dt)
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
            policy=policy,
            num_replicas=train_cfg.num_replicas,
        )
        loss.backward()
        optimizer.step()
        wealth = wealth_next.detach()

        if step % train_cfg.log_every == 0 or step == train_cfg.num_steps - 1:
            ev = evaluate_critic_on_grid(
                critic=critic,
                params=params,
                policy=policy,
                low=train_cfg.wealth_min,
                high=train_cfg.wealth_max,
                num=train_cfg.eval_points,
                device=train_cfg.device,
            )
            history["step"].append(step)
            history["loss"].append(float(metrics["loss"]))
            history["mae"].append(float(ev["mae"]))
            history["rmse"].append(float(ev["rmse"]))
            history["v_w_mae"].append(float(ev["v_w_mae"]))
            history["v_w_norm"].append(float(ev["v_w_norm"]))
            history["hjb_rmse"].append(float(ev["hjb_rmse"]))
            history["dtd_mse"].append(float(metrics["dtd_mse"]))
            history["td_mse"].append(float(metrics["td_mse"]))
            iterator.set_postfix(loss=f"{metrics['loss']:.2e}", mae=f"{ev['mae']:.2e}")
    return history


def plot_drift(pretrain_hist: dict, dtd_hist: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))

    pre_steps = np.asarray(pretrain_hist["step"])
    pre_mae = np.asarray(pretrain_hist["mae"])
    dtd_steps = np.asarray(dtd_hist["step"]) + (pre_steps[-1] + 1 if len(pre_steps) else 0)
    boundary = pre_steps[-1] if len(pre_steps) else 0

    ax = axes[0]
    ax.plot(pre_steps, pre_mae, color="tab:green", label="pretrain (MSE on V^π)")
    ax.plot(dtd_steps, dtd_hist["mae"], color="tab:red", label="dTD from truth")
    ax.axvline(boundary, color="black", ls=":", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel("MAE vs V^π")
    ax.set_title("MAE: pretrain hits truth, then dTD takes over")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(pre_steps, pretrain_hist["v_w_mae"], color="tab:green", label="pretrain")
    ax.plot(dtd_steps, dtd_hist["v_w_mae"], color="tab:red", label="dTD")
    ax.axvline(boundary, color="black", ls=":", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel(r"MAE of $V_w$")
    ax.set_title(r"$V_w$ error under dTD")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sigma", type=float, default=0.10)
    p.add_argument("--pi", type=float, default=0.75)
    p.add_argument("--kappa", type=float, default=0.06125)
    p.add_argument("--r", type=float, default=0.02)
    p.add_argument("--mu", type=float, default=0.08)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--rho", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--dt", type=float, default=1.0 / 252.0)
    p.add_argument("--wealth-min", type=float, default=0.3)
    p.add_argument("--wealth-max", type=float, default=3.0)
    p.add_argument("--eval-points", type=int, default=200)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--pretrain-steps", type=int, default=8000)
    p.add_argument("--pretrain-target-mae", type=float, default=1.0)
    p.add_argument("--num-steps", type=int, default=12000)
    p.add_argument("--loss", type=str, default="dtd")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=str, default="results/dtd_from_truth")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    params = MertonParams(r=args.r, mu=args.mu, sigma=args.sigma, gamma=args.gamma, rho=args.rho)
    policy = PolicyParams(pi=args.pi, kappa=args.kappa)
    train_cfg = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        learning_rate=args.lr,
        dt=args.dt,
        wealth_min=args.wealth_min,
        wealth_max=args.wealth_max,
        eval_points=args.eval_points,
        beta=0.5,
        device=args.device,
        log_every=args.log_every,
    )

    critic = build_critic(params, device=args.device)

    pretrain_hist = supervised_pretrain(
        critic, params, policy, train_cfg, args.pretrain_steps, args.pretrain_target_mae
    )
    print(f"pretrain done: final MAE={pretrain_hist['mae'][-1]:.3e}, v_w_mae={pretrain_hist['v_w_mae'][-1]:.3e}")

    dtd_hist = run_dtd_from(critic, params, policy, train_cfg, args.loss)
    print(f"{args.loss} done: final MAE={dtd_hist['mae'][-1]:.3e}, v_w_mae={dtd_hist['v_w_mae'][-1]:.3e}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_grid = evaluate_critic_on_grid(
        critic=critic, params=params, policy=policy,
        low=train_cfg.wealth_min, high=train_cfg.wealth_max,
        num=train_cfg.eval_points, device=args.device,
    )
    result = {
        "history": dtd_hist,
        "pretrain_history": pretrain_hist,
        "summary": summary_grid,
        "meta": {
            "params": asdict(params),
            "policy": asdict(policy),
            "train_cfg": asdict(train_cfg),
            "loss_name": args.loss,
            "experiment": "dtd_from_truth",
        },
    }
    save_checkpoint(critic, result, out_dir)
    plot_drift(pretrain_hist, dtd_hist, out_dir / "drift.png")

    scalar = {k: float(v) for k, v in summary_grid.items() if isinstance(v, (int, float))}
    (out_dir / "summary.json").write_text(json.dumps(scalar, indent=2))


if __name__ == "__main__":
    main()
