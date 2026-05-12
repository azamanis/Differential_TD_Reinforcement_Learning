"""
Sweep over beta in beta-dTD = (1-beta) * L_TD + beta * L_dTD.
Train one critic per beta value, then plot:
  - MAE vs beta (with TD and pure dTD as endpoints)
  - v_w_norm vs beta
  - value fit for a few representative betas overlayed on truth
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
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--num-steps", type=int, default=12000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--dt", type=float, default=1.0 / 252.0)
    p.add_argument("--wealth-min", type=float, default=0.3)
    p.add_argument("--wealth-max", type=float, default=3.0)
    p.add_argument("--eval-points", type=int, default=200)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--betas", type=str, default="0,0.1,0.25,0.5,0.75,0.9,1.0")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=str, default="results/beta_sweep_plot")
    args = p.parse_args()

    betas = [float(b) for b in args.betas.split(",")]
    params = MertonParams(r=args.r, mu=args.mu, sigma=args.sigma, gamma=args.gamma, rho=args.rho)
    policy = PolicyParams(pi=args.pi, kappa=args.kappa)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    for beta in betas:
        # β=0 → pure TD; β=1 → pure dTD; in between → β-dTD.
        if beta == 0.0:
            loss_name, label = "td", "TD (β=0)"
        elif beta == 1.0:
            loss_name, label = "dtd", "pure dTD (β=1)"
        else:
            loss_name, label = "beta_dtd", f"β-dTD β={beta:g}"
        train_cfg = TrainConfig(
            seed=args.seed, batch_size=args.batch_size, num_steps=args.num_steps,
            learning_rate=args.lr, dt=args.dt,
            wealth_min=args.wealth_min, wealth_max=args.wealth_max,
            eval_points=args.eval_points, beta=beta,
            device=args.device, log_every=args.log_every,
        )
        print(f"=== Training {label} ===")
        critic, result = train_fixed_policy_critic(
            params=params, policy=policy, train_cfg=train_cfg, loss_name=loss_name,
        )
        save_checkpoint(critic, result, out_dir / f"beta_{beta:g}")
        runs.append({"beta": beta, "loss_name": loss_name, "label": label, "result": result})

    # ----- Per-run best-MAE (early stopping) -----
    truth_vwn = float(runs[0]["result"]["summary"]["v_w_norm_true"])
    for r in runs:
        h = r["result"]["history"]
        mae = np.asarray(h["mae"])
        i_best = int(np.argmin(mae))
        r["best_mae"] = float(mae[i_best])
        r["best_step"] = int(h["step"][i_best])
        r["best_vwn"] = float(h["v_w_norm"][i_best])

    bs = np.array(betas)
    best_mae = np.array([r["best_mae"] for r in runs])
    best_vwn = np.array([r["best_vwn"] for r in runs])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Left: best MAE per β.
    axes[0].plot(bs, best_mae, "o-", color="tab:red")
    axes[0].set_xlabel("β"); axes[0].set_ylabel("min MAE during training")
    axes[0].set_yscale("log")
    axes[0].set_title(f"Best MAE per β, early stopping (σ={args.sigma})")
    axes[0].grid(True, which="both", alpha=0.3)

    # Middle: V_w at the best-MAE step, with truth line.
    axes[1].plot(bs, best_vwn, "o-", color="tab:blue", label=r"$\|V_w\|$ at best-MAE step")
    axes[1].axhline(truth_vwn, color="black", ls="--", label=f"truth $\\|V_w\\|$ = {truth_vwn:.0f}")
    axes[1].set_xlabel("β"); axes[1].set_ylabel(r"$\|V_w\|$ (RMS over grid)")
    axes[1].set_title(r"$\|V_w\|$ at the best-MAE step")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    # Right: MAE-over-training, one line per β, with stars marking each min.
    cmap = plt.cm.viridis(np.linspace(0.0, 0.85, len(runs)))
    for r, color in zip(runs, cmap):
        h = r["result"]["history"]
        axes[2].plot(h["step"], h["mae"], color=color, label=r["label"], linewidth=1.4)
        axes[2].plot(r["best_step"], r["best_mae"], "*", color=color, markersize=10,
                     markeredgecolor="black", markeredgewidth=0.5)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Training step"); axes[2].set_ylabel("MAE vs $V^\\pi$")
    axes[2].set_title("MAE over training; ★ marks per-β minimum")
    axes[2].legend(fontsize=8); axes[2].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "beta_sweep.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")

    summary = {
        "betas": betas,
        "best_mae": best_mae.tolist(),
        "best_step": [r["best_step"] for r in runs],
        "best_v_w_norm": best_vwn.tolist(),
        "v_w_norm_truth": truth_vwn,
        "meta": {
            "params": asdict(params), "policy": asdict(policy),
            "num_steps": args.num_steps, "lr": args.lr, "dt": args.dt,
            "batch_size": args.batch_size, "sigma": args.sigma,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
