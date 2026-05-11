"""
Decompose the dTD MSE into its analytic signal and noise-floor parts and
overlay the empirical dTD MSE measured on the eval grid.

Reads checkpoints produced by train_critic.py and writes a PNG per run.

Usage:
    python scripts/plot_dtd_decomposition.py results/dtd_decomp/dtd results/dtd_decomp/td
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_one(run_dir: Path) -> None:
    ckpt = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    h = ckpt["result"]["history"]
    loss_name = ckpt["result"]["meta"]["loss_name"]

    step = np.asarray(h["step"])
    signal = np.asarray(h["dtd_signal_part"])
    noise = np.asarray(h["dtd_noise_floor"])
    total = signal + noise

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(step, total, label="signal + noise", lw=1.5, ls="--", color="tab:gray")
    ax.plot(step, signal, label=r"signal = $\Delta t^2 \cdot \overline{HJB^2}$", color="tab:blue")
    ax.plot(step, noise, label=r"noise floor = $\Delta t \cdot \overline{b^2 W^2 V_w^2}$", color="tab:red")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("contribution to $E[\\delta_{dTD}^2]$")
    ax.set_title(f"dTD loss decomposition — loss={loss_name}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out = run_dir / "dtd_decomposition.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    for d in args.run_dirs:
        plot_one(d)


if __name__ == "__main__":
    main()
