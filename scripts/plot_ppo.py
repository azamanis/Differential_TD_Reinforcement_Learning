from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from merton_dtd.config import MertonParams
from merton_dtd.merton import optimal_policy_closed_form


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=str)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    history = json.loads((run_dir / "history.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())

    params_d = summary["params"]
    params = MertonParams(**params_d)
    try:
        ref_policy, _ = optimal_policy_closed_form(params)
        pi_ref, kappa_ref = ref_policy.pi, ref_policy.kappa
    except ValueError:
        pi_ref = kappa_ref = None

    its = history["iter"]

    fig, axes = plt.subplots(3, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.plot(its, history["mean_pi"], label="mean π", color="tab:blue")
    ax.fill_between(
        its,
        [m - s for m, s in zip(history["mean_pi"], history["std_pi"])],
        [m + s for m, s in zip(history["mean_pi"], history["std_pi"])],
        alpha=0.2, color="tab:blue",
    )
    if pi_ref is not None:
        ax.axhline(pi_ref, color="k", linestyle="--", label=f"π* = {pi_ref:.3f}")
    ax.set_title("actor π")
    ax.set_xlabel("iter")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(its, history["mean_kappa"], label="mean κ", color="tab:orange")
    ax.fill_between(
        its,
        [m - s for m, s in zip(history["mean_kappa"], history["std_kappa"])],
        [m + s for m, s in zip(history["mean_kappa"], history["std_kappa"])],
        alpha=0.2, color="tab:orange",
    )
    if kappa_ref is not None:
        ax.axhline(kappa_ref, color="k", linestyle="--", label=f"κ* = {kappa_ref:.4f}")
    ax.set_title("actor κ")
    ax.set_xlabel("iter")
    ax.legend()

    ax = axes[1, 0]
    if "mean_return" in history:
        ax.plot(its, history["mean_return"], color="tab:green", label="mean GAE return")
    if "mean_value" in history:
        ax.plot(its, history["mean_value"], color="tab:olive", label="mean V(s)", linestyle="--")
    if "mean_reward_rate" in history:
        ax2 = ax.twinx()
        ax2.plot(its, history["mean_reward_rate"], color="tab:gray", alpha=0.5,
                 label="mean U(c) rate")
        ax2.set_ylabel("U(c) rate", color="tab:gray")
        ax2.legend(loc="lower right")
    ax.set_title("return / value")
    ax.set_xlabel("iter")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    ax.semilogy(
        its,
        [max(abs(x), 1e-12) for x in history["critic_loss"]],
        color="tab:red",
    )
    ax.set_title("critic loss (|.|, log scale)")
    ax.set_xlabel("iter")

    ax = axes[2, 0]
    ax.semilogy(its, history["mae"], color="tab:purple")
    ax.set_title("critic MAE vs closed-form (ref policy)")
    ax.set_xlabel("iter")

    ax = axes[2, 1]
    ax.plot(its, history["entropy"], color="tab:brown", label="entropy")
    ax2 = ax.twinx()
    ax2.plot(its, history["kl_approx"], color="tab:gray", label="KL approx", alpha=0.6)
    ax.set_title("entropy & approx KL")
    ax.set_xlabel("iter")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    fig.suptitle(f"PPO diagnostics — {run_dir}")
    fig.tight_layout()

    out = Path(args.out) if args.out else run_dir / "diagnostics.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
