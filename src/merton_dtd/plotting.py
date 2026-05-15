from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _nice_log_ticks(xmin: float, xmax: float) -> np.ndarray:
    """
    Return log-scale ticks spanning the full range [xmin, xmax].
    """
    xmin = max(float(xmin), 1e-12)
    xmax = max(float(xmax), xmin * 1.000001)

    e_min = int(np.floor(np.log10(xmin)))
    e_max = int(np.ceil(np.log10(xmax)))
    ticks = 10.0 ** np.arange(e_min, e_max + 1)

    # Keep only ticks inside the visible range, but ensure endpoints are covered
    ticks = ticks[(ticks >= xmin * 0.999) & (ticks <= xmax * 1.001)]
    if ticks.size == 0:
        ticks = np.array([xmin, xmax])
    return ticks


def _format_run_title(result: dict, base: str) -> str:
    def fmt(value) -> str:
        if isinstance(value, (int, float, np.integer, np.floating)):
            return f"{value:g}"
        return str(value)

    meta = result.get("meta", {})
    params = meta.get("params", {})
    train_cfg = meta.get("train_cfg", {})
    horizon = meta.get("horizon")

    loss = meta.get("loss_name", "unknown loss")
    sigma = params.get("sigma", "?")
    lr = train_cfg.get("learning_rate", "?")
    batch_size = train_cfg.get("batch_size", "?")
    num_steps = train_cfg.get("num_steps", "?")
    dt = train_cfg.get("dt", "?")
    num_replicas = train_cfg.get("num_replicas", 1)

    if horizon is None:
        horizon_part = "IH"
    else:
        T = horizon.get("T", "?")
        terminal_coef = horizon.get("terminal_coef", "?")
        horizon_part = f"FH T={fmt(T)}, terminal_coef={fmt(terminal_coef)}"

    details = (
        f"{horizon_part} | loss={loss}, K={num_replicas}, "
        f"sigma={sigma}, lr={fmt(lr)}, B={batch_size}, steps={num_steps}, dt={fmt(dt)}"
    )
    return f"{base}\n{details}"


def plot_value_fit(result: dict, out_file: str | Path) -> None:
    summary = result["summary"]
    wealth = np.asarray(summary["wealth"], dtype=float)
    truth = np.asarray(summary["truth"], dtype=float)
    pred = np.asarray(summary["pred"], dtype=float)

    if pred.ndim == 2:
        _plot_finite_horizon_value_fit(result, out_file)
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(wealth, truth, label="Exact closed-form value", linewidth=2.2)
    ax.plot(wealth, pred, label="Neural critic prediction", linestyle="--", linewidth=2.0)

    ax.set_xscale("log")
    ax.set_xlim(wealth.min(), wealth.max())

    ticks = _nice_log_ticks(wealth.min(), wealth.max())
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])

    ax.set_xlabel("Current wealth $W_t$")
    ax.set_ylabel("Value $V(W_t)$")
    ax.set_title(_format_run_title(result, "Exact value function vs learned critic"))
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(frameon=True)
    fig.tight_layout()

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_finite_horizon_value_fit(result: dict, out_file: str | Path) -> None:
    summary = result["summary"]
    wealth = np.asarray(summary["wealth"], dtype=float)
    time = np.asarray(summary["t_grid"], dtype=float)
    truth = np.asarray(summary["truth"], dtype=float)
    pred = np.asarray(summary["pred"], dtype=float)
    abs_err = np.abs(pred - truth)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    slice_idx = np.unique(
        np.round(np.linspace(0, len(time) - 1, min(4, len(time)))).astype(int)
    )
    for idx in slice_idx:
        label = f"t={time[idx]:.2f}"
        (truth_line,) = axes[0].plot(
            wealth, truth[idx], linewidth=2.0, label=f"Exact {label}"
        )
        axes[0].plot(
            wealth,
            pred[idx],
            color=truth_line.get_color(),
            linestyle="--",
            linewidth=1.8,
            label=f"Learned {label}",
        )

    axes[0].set_xscale("log")
    axes[0].set_xlim(wealth.min(), wealth.max())
    ticks = _nice_log_ticks(wealth.min(), wealth.max())
    axes[0].set_xticks(ticks)
    axes[0].set_xticklabels([f"{t:g}" for t in ticks])
    axes[0].set_xlabel("Current wealth $W_t$")
    axes[0].set_ylabel("Value $V(t, W_t)$")
    axes[0].set_title("Exact vs learned value at selected times")
    axes[0].grid(True, which="both", linestyle=":", alpha=0.5)
    axes[0].legend(frameon=True, fontsize=8, ncol=2)

    mesh = axes[1].pcolormesh(wealth, time, abs_err, shading="auto")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Current wealth $W_t$")
    axes[1].set_ylabel("Time $t$")
    axes[1].set_title("Absolute error over finite-horizon grid")
    fig.colorbar(mesh, ax=axes[1], label="|learned - exact|")

    fig.suptitle(_format_run_title(result, "Finite-horizon Merton critic fit"), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(result: dict, out_file: str | Path) -> None:
    history = result["history"]
    steps = np.asarray(history["step"], dtype=float)
    loss = np.asarray(history["loss"], dtype=float)
    mae = np.asarray(history["mae"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(steps, loss, label="Total training loss", linewidth=2.0)
    axes[0].set_yscale("log")
    axes[0].set_xlim(steps.min(), steps.max())
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Loss (log scale)")
    axes[0].set_title("Training loss")
    axes[0].grid(True, which="both", linestyle=":", alpha=0.5)
    axes[0].legend(frameon=True)

    # Right panel: MAE vs closed form.
    axes[1].plot(steps, mae, label="MAE", linewidth=2.0)
    axes[1].set_yscale("log")
    axes[1].set_xlim(steps.min(), steps.max())
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("MAE vs exact value (log scale)")
    axes[1].set_title("Critic accuracy against closed-form benchmark")
    axes[1].grid(True, which="both", linestyle=":", alpha=0.5)
    axes[1].legend(frameon=True)

    fig.suptitle(_format_run_title(result, "Merton fixed-policy critic training summary"), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
