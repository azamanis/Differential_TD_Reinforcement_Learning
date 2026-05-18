"""
Plot MAE-over-training-step for TD vs β-dTD at a fixed
(σ, lr, B) cell from the HPO sweep. Averages over seeds only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_histories(runs_dir: Path, cell: str, tag: str, num_seeds: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (steps, mae_seeds) where mae_seeds has shape (num_seeds, T)."""
    steps = None
    maes = []
    for s in range(num_seeds):
        run_dir = runs_dir / f"{cell}{tag}_s{s}"
        h = json.loads((run_dir / "history.json").read_text())
        if steps is None:
            steps = np.asarray(h["step"], dtype=float)
        maes.append(np.asarray(h["mae"], dtype=float))
    return steps, np.stack(maes)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="results/hpo_sweep")
    p.add_argument("--cell", type=str, default="sig0.05_lr0.005_B256_",
                   help="run_id prefix up to the method tag")
    p.add_argument("--td-tag", type=str, default="td_beta0",
                   help="TD method tag appended to the cell prefix")
    p.add_argument("--beta-tag-template", type=str, default="beta_dtd_beta{beta}",
                   help="Template for β-dTD run tags; {beta} is replaced by each β value")
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--betas", type=str, default="0.25,0.5,0.75,0.9",
                   help="β-dTD configs to plot alongside TD")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    runs_dir = Path(args.out_dir) / "runs"
    out_path = Path(args.out or (Path(args.out_dir) / "plots" / f"{args.cell.rstrip('_')}_mae_over_time.png"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    # TD anchor (red)
    steps, td = load_histories(runs_dir, args.cell, args.td_tag, args.num_seeds)
    td_mean = td.mean(0)
    ax.plot(steps, td_mean, color="C3", lw=2.0, label=f"TD  (final {td_mean[-1]:.2f})")

    # β-dTD lines
    betas = [float(b) for b in args.betas.split(",")]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.80, len(betas)))
    for color, beta in zip(cmap, betas):
        beta_tag = args.beta_tag_template.format(beta=beta)
        _, bd = load_histories(runs_dir, args.cell, beta_tag, args.num_seeds)
        m = bd.mean(0)
        ax.plot(steps, m, color=color, lw=1.8, label=f"β-dTD β={beta:g}  (final {m[-1]:.2f})")

    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("MAE vs. closed-form $V^\\pi$ (log scale)")
    sigma_str = args.cell.split("_")[0].replace("sig", "σ=")
    lr_str = args.cell.split("_")[1].replace("lr", "lr=")
    B_str = args.cell.split("_")[2].replace("B", "B=")
    ax.set_title(f"MAE over training  ·  {sigma_str}, {lr_str}, {B_str}  ·  mean over {args.num_seeds} seeds")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
