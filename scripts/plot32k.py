from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def load_history(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one run's training history."""
    hist_path = run_dir / "history.json"
    if not hist_path.exists():
        raise FileNotFoundError(f"Missing history file: {hist_path}")

    h = json.loads(hist_path.read_text())

    steps = np.asarray(h["step"], dtype=float)
    mae = np.asarray(h["mae"], dtype=float)

    return steps, mae


def load_seed_group(
    runs_dir: Path,
    run_name_template: str,
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load several seeds for one method.

    Returns:
        steps: shape (T,)
        maes: shape (num_seeds, T)
    """
    steps_ref = None
    maes = []

    for s in seeds:
        run_dir = runs_dir / run_name_template.format(seed=s)
        steps, mae = load_history(run_dir)

        if steps_ref is None:
            steps_ref = steps
        else:
            if len(steps) != len(steps_ref) or not np.allclose(steps, steps_ref):
                raise ValueError(f"Step grid mismatch in {run_dir}")

        maes.append(mae)

    return steps_ref, np.stack(maes, axis=0)


def plot_mean_curve(
    ax,
    steps: np.ndarray,
    maes: np.ndarray,
    label: str,
    color=None,
    linestyle: str = "-",
    linewidth: float = 2.0,
) -> None:
    """Plot only mean over seeds, no SD band."""
    mean_mae = maes.mean(axis=0)

    ax.plot(
        steps,
        mean_mae,
        label=f"{label} (final {mean_mae[-1]:.2f})",
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default="results/beta_dtd_mean_at_hpo",
        help="Folder containing the K=32 mean-target results.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="results/beta_dtd_mean_at_hpo/mae_over_time_K32_no_sd_no_K1.png",
        help="Output path for the new figure.",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated seeds to average over.",
    )

    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]

    root = Path(args.root)
    runs_dir = root / "runs"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    # ------------------------------------------------------------
    # TD-mean, K = 32
    # ------------------------------------------------------------
    steps, maes = load_seed_group(
        runs_dir,
        "sig0.05_lr0.005_B256_td_mean_K32_s{seed}",
        seeds,
    )

    plot_mean_curve(
        ax,
        steps,
        maes,
        label="TD-mean, K=32",
        color="C3",
        linestyle="-",
        linewidth=2.4,
    )

    # ------------------------------------------------------------
    # beta-dTD-mean, K = 32
    # ------------------------------------------------------------
    betas = [0.25, 0.5, 0.75, 0.9]
    colors = plt.cm.viridis(np.linspace(0.15, 0.80, len(betas)))

    for beta, color in zip(betas, colors):
        steps, maes = load_seed_group(
            runs_dir,
            f"sig0.05_lr0.005_B256_beta_dtd_mean_K32_beta{beta:g}_s{{seed}}",
            seeds,
        )

        plot_mean_curve(
            ax,
            steps,
            maes,
            label=rf"$\beta$-dTD-mean, $\beta={beta:g}$, K=32",
            color=color,
            linestyle="-",
            linewidth=1.9,
        )

    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(r"MAE vs. closed-form $V^\pi$ (log scale)")
    ax.set_title(
        r"MAE over training, $K=32$, $\sigma=0.05$, lr=$0.005$, $B=256$ — mean over 3 seeds"
    )

    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()