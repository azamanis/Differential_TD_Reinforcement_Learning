"""
Train TD, pure dTD, and beta-dTD on the same fixed-policy Merton problem,
then produce a single side-by-side comparison: value fit vs truth (left)
and MAE-over-training (right).

Usage:
    python scripts/compare_methods.py --sigma 0.10 --out-dir results/compare/sigma0p10
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from merton_dtd.config import MertonParams, PolicyParams, TrainConfig
from merton_dtd.training import save_checkpoint, train_fixed_policy_critic


METHODS = [
    ("td",        "TD",         "tab:blue",   "-"),
    ("dtd",       "pure dTD",   "tab:red",    "-"),
    ("beta_dtd",  "beta-dTD",   "tab:purple", "-"),
]


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
    p.add_argument("--num-steps", type=int, default=12000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--dt", type=float, default=1.0 / 252.0)
    p.add_argument("--wealth-min", type=float, default=0.3)
    p.add_argument("--wealth-max", type=float, default=3.0)
    p.add_argument("--eval-points", type=int, default=200)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=str, default="results/compare/run")
    args = p.parse_args()

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
        beta=args.beta,
        device=args.device,
        log_every=args.log_every,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict] = {}

    for loss_name, label, _color, _ls in METHODS:
        print(f"=== Training {label} ({loss_name}) ===")
        critic, result = train_fixed_policy_critic(
            params=params, policy=policy, train_cfg=train_cfg, loss_name=loss_name,
        )
        save_checkpoint(critic, result, out_dir / loss_name)
        runs[loss_name] = result

    # ----- Comparison plot -----
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # Left: value fit against ground truth.
    wealth = np.asarray(next(iter(runs.values()))["summary"]["wealth"], dtype=float)
    truth = np.asarray(next(iter(runs.values()))["summary"]["truth"], dtype=float)
    axes[0].plot(wealth, truth, label="closed-form $V^\\pi$", color="black", linewidth=2.5)
    for loss_name, label, color, ls in METHODS:
        pred = np.asarray(runs[loss_name]["summary"]["pred"], dtype=float)
        axes[0].plot(wealth, pred, label=label, color=color, linestyle="--", linewidth=1.8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Wealth $W$")
    axes[0].set_ylabel("$V(W)$")
    axes[0].set_title(f"Value fit (σ={args.sigma}, β={args.beta})")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, which="both", alpha=0.3)

    # Right: MAE over training.
    for loss_name, label, color, ls in METHODS:
        h = runs[loss_name]["history"]
        axes[1].plot(h["step"], h["mae"], label=label, color=color, linewidth=2.0)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("MAE vs $V^\\pi$ (log scale)")
    axes[1].set_title("MAE over training")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")

    # Tiny scalar summary
    summary = {
        loss_name: {
            "final_mae": float(runs[loss_name]["summary"]["mae"]),
            "final_v_w_norm": float(runs[loss_name]["summary"].get("v_w_norm", float("nan"))),
        }
        for loss_name, _, _, _ in METHODS
    }
    summary["meta"] = {
        "params": asdict(params), "policy": asdict(policy),
        "train_cfg": asdict(train_cfg), "beta": args.beta,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
