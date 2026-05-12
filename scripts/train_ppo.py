from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from merton_dtd.config import (
    HorizonConfig,
    MertonParams,
    PPOConfig,
    TrainConfig,
)
from merton_dtd.merton import optimal_policy_closed_form
from merton_dtd.ppo import train_ppo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a PPO actor + critic on the Merton problem."
    )

    # Merton parameters
    p.add_argument("--r", type=float, default=0.02)
    p.add_argument("--mu", type=float, default=0.08)
    p.add_argument("--sigma", type=float, default=0.20)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--rho", type=float, default=0.08)

    # Critic loss (same options as train_critic.py except dtd_mean variants)
    p.add_argument(
        "--loss",
        type=str,
        default="beta_dtd",
        choices=["td", "dtd", "beta_dtd", "rl_pinn", "naive_dtd", "beta_naive_dtd"],
    )
    p.add_argument("--beta", type=float, default=0.5)

    # Shared train config
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-3, help="critic learning rate")
    p.add_argument("--dt", type=float, default=1.0 / 252.0)
    p.add_argument("--wealth-min", type=float, default=0.3)
    p.add_argument("--wealth-max", type=float, default=3.0)
    p.add_argument("--eval-points", type=int, default=200)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=str, default="results/train_ppo")

    # PPO hyperparameters
    p.add_argument("--n-iters", type=int, default=200)
    p.add_argument("--n-steps", type=int, default=64)
    p.add_argument("--n-envs", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--minibatch-size", type=int, default=1024)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--init-log-std", type=float, default=-1.0)
    p.add_argument("--kappa-floor", type=float, default=1e-3)
    p.add_argument("--actor-hidden-dim", type=int, default=64)
    p.add_argument("--actor-depth", type=int, default=2)
    p.add_argument("--actor-activation", type=str, default="tanh")
    p.add_argument("--state-dependent-std", action="store_true")

    # Finite horizon
    p.add_argument("--horizon", type=float, default=None)
    p.add_argument("--terminal-coef", type=float, default=0.0)
    p.add_argument("--terminal-weight", type=float, default=1.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    params = MertonParams(
        r=args.r, mu=args.mu, sigma=args.sigma, gamma=args.gamma, rho=args.rho
    )
    train_cfg = TrainConfig(
        seed=args.seed,
        batch_size=args.n_envs,
        num_steps=args.n_iters,
        learning_rate=args.lr,
        dt=args.dt,
        wealth_min=args.wealth_min,
        wealth_max=args.wealth_max,
        eval_points=args.eval_points,
        beta=args.beta,
        device=args.device,
        log_every=1,
        num_replicas=1,
    )
    ppo_cfg = PPOConfig(
        n_iters=args.n_iters,
        n_steps=args.n_steps,
        n_envs=args.n_envs,
        n_epochs=args.n_epochs,
        minibatch_size=args.minibatch_size,
        clip_eps=args.clip_eps,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        value_coef=args.value_coef,
        lr_actor=args.lr_actor,
        max_grad_norm=args.max_grad_norm,
        init_log_std=args.init_log_std,
        kappa_floor=args.kappa_floor,
        actor_hidden_dim=args.actor_hidden_dim,
        actor_depth=args.actor_depth,
        actor_activation=args.actor_activation,
        state_dependent_std=args.state_dependent_std,
    )
    horizon = (
        HorizonConfig(T=args.horizon, terminal_coef=args.terminal_coef)
        if args.horizon is not None
        else None
    )

    actor, critic, result = train_ppo(
        params=params,
        train_cfg=train_cfg,
        ppo_cfg=ppo_cfg,
        loss_name=args.loss,
        horizon=horizon,
        terminal_weight=args.terminal_weight,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "result": result,
        },
        out_dir / "checkpoint.pt",
    )
    (out_dir / "history.json").write_text(
        json.dumps(
            {k: [float(x) for x in v] for k, v in result["history"].items()},
            indent=2,
        )
    )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "params": asdict(params),
                "train_cfg": asdict(train_cfg),
                "ppo_cfg": asdict(ppo_cfg),
                "loss": args.loss,
                "horizon": asdict(horizon) if horizon is not None else None,
                "summary": result["summary"],
            },
            indent=2,
        )
    )

    try:
        ref_policy, _ = optimal_policy_closed_form(params)
        print(
            f"Closed-form optimum (infinite horizon): pi*={ref_policy.pi:.4f}, "
            f"kappa*={ref_policy.kappa:.4f}"
        )
    except ValueError:
        pass
    s = result["summary"]
    print(
        f"PPO finished. learned mean (pi, kappa) = "
        f"({s['final_pi_mean']:.4f}, {s['final_kappa_mean']:.4f}); "
        f"critic MAE vs ref policy = {s['final_mae']:.3e}"
    )
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
