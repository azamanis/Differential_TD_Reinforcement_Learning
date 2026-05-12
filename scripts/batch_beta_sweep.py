"""
Joint sweep over batch_size and β at fixed σ. Hypothesis: β-dTD's bias-variance
trade-off helps TD only when TD's gradient noise dominates (small batches).
At large batches, TD has plenty of signal and β-dTD's added bias is pure harm.

Records per-run best MAE (early stopping) and plots best-MAE vs β, one line
per batch size.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from merton_dtd.config import MertonParams, PolicyParams, TrainConfig
from merton_dtd.training import save_checkpoint, train_fixed_policy_critic


def loss_for_beta(beta: float) -> str:
    if beta == 0.0:
        return "td"
    if beta == 1.0:
        return "dtd"
    return "beta_dtd"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sigma", type=float, default=0.20)
    p.add_argument("--pi", type=float, default=0.75)
    p.add_argument("--kappa", type=float, default=0.06125)
    p.add_argument("--r", type=float, default=0.02)
    p.add_argument("--mu", type=float, default=0.08)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--rho", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--dt", type=float, default=1.0 / 252.0)
    p.add_argument("--wealth-min", type=float, default=0.3)
    p.add_argument("--wealth-max", type=float, default=3.0)
    p.add_argument("--eval-points", type=int, default=200)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--batches", type=str, default="64,256,2048")
    p.add_argument("--betas", type=str, default="0,0.05,0.1,0.25,0.5")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=str, default="results/batch_beta_sweep")
    args = p.parse_args()

    batches = [int(b) for b in args.batches.split(",")]
    betas = [float(b) for b in args.betas.split(",")]

    params = MertonParams(r=args.r, mu=args.mu, sigma=args.sigma, gamma=args.gamma, rho=args.rho)
    policy = PolicyParams(pi=args.pi, kappa=args.kappa)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # grid: best_mae[batch_idx, beta_idx], best_step[..], best_vwn[..]
    best_mae = np.full((len(batches), len(betas)), np.nan)
    best_step = np.full((len(batches), len(betas)), -1, dtype=int)
    best_vwn = np.full((len(batches), len(betas)), np.nan)
    truth_vwn_holder = [None]

    for bi, B in enumerate(batches):
        for ki, beta in enumerate(betas):
            loss_name = loss_for_beta(beta)
            train_cfg = TrainConfig(
                seed=args.seed, batch_size=B, num_steps=args.num_steps,
                learning_rate=args.lr, dt=args.dt,
                wealth_min=args.wealth_min, wealth_max=args.wealth_max,
                eval_points=args.eval_points, beta=beta,
                device=args.device, log_every=args.log_every,
            )
            print(f"=== B={B}  β={beta}  loss={loss_name} ===")
            critic, result = train_fixed_policy_critic(
                params=params, policy=policy, train_cfg=train_cfg, loss_name=loss_name,
            )
            run_dir = out_dir / f"B{B}_beta{beta:g}"
            save_checkpoint(critic, result, run_dir)

            h = result["history"]
            mae = np.asarray(h["mae"])
            i = int(np.argmin(mae))
            best_mae[bi, ki] = float(mae[i])
            best_step[bi, ki] = int(h["step"][i])
            best_vwn[bi, ki] = float(h["v_w_norm"][i])
            if truth_vwn_holder[0] is None:
                truth_vwn_holder[0] = float(result["summary"]["v_w_norm_true"])

    truth_vwn = truth_vwn_holder[0]

    # ----- Plot: best MAE vs β, one line per batch -----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))
    cmap = plt.cm.plasma(np.linspace(0.1, 0.8, len(batches)))
    for bi, B in enumerate(batches):
        axes[0].plot(betas, best_mae[bi], "o-", color=cmap[bi], label=f"B={B}", linewidth=2.0)
        axes[1].plot(betas, best_vwn[bi], "o-", color=cmap[bi], label=f"B={B}", linewidth=2.0)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("β"); axes[0].set_ylabel("min MAE during training")
    axes[0].set_title(f"Best MAE vs β at each batch size (σ={args.sigma}, {args.num_steps} steps)")
    axes[0].grid(True, which="both", alpha=0.3); axes[0].legend(fontsize=9)

    axes[1].axhline(truth_vwn, color="black", ls="--", label=f"truth $\\|V_w\\|$={truth_vwn:.0f}")
    axes[1].set_xlabel("β"); axes[1].set_ylabel(r"$\|V_w\|$ at best-MAE step")
    axes[1].set_title(r"$\|V_w\|$ at the best-MAE step")
    axes[1].grid(True, which="both", alpha=0.3); axes[1].legend(fontsize=9)

    fig.tight_layout()
    out_path = out_dir / "batch_beta_sweep.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")

    summary = {
        "batches": batches,
        "betas": betas,
        "best_mae": best_mae.tolist(),
        "best_step": best_step.tolist(),
        "best_v_w_norm": best_vwn.tolist(),
        "v_w_norm_truth": truth_vwn,
        "meta": {
            "params": asdict(params), "policy": asdict(policy),
            "num_steps": args.num_steps, "lr": args.lr, "dt": args.dt,
            "sigma": args.sigma,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Console table
    print()
    print(f"truth |V_w| = {truth_vwn:.2f}")
    print(f"{'batch':>6s}" + "".join(f"  β={b:<6g}" for b in betas))
    for bi, B in enumerate(batches):
        print(f"{B:>6d}" + "".join(f"  {best_mae[bi, ki]:8.3f}" for ki in range(len(betas))))


if __name__ == "__main__":
    main()
