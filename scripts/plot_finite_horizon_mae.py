"""
Plot MAE-over-training-step for TD vs β-dTD at the finite-horizon config
where β-dTD was already shown to beat TD (T=1, σ=0.2, lr=2e-3, B=1024, dt=0.02).

Re-trains each method across N seeds (default 3) and plots mean ± 1 SD.

Pass `--t-target <t>` to restrict the reported MAE to a single time slice
(the closest grid point) instead of averaging over all (t, W).
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _run_one(cfg: dict[str, Any]) -> dict[str, list[float]]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    # silence tqdm
    import merton_dtd.training as _t
    class _SR:
        def __init__(self, n, **kw): self._it = iter(range(n))
        def __iter__(self): return self
        def __next__(self): return next(self._it)
        def set_postfix(self, *a, **kw): pass
        def close(self): pass
    _t.trange = lambda n, **kw: _SR(n, **kw)  # type: ignore[assignment]

    # Optional: restrict reported MAE to a single time slice.
    t_target = cfg.get("t_target")
    if t_target is not None:
        import merton_dtd.eval as _e
        _orig = _e._evaluate_finite_horizon_critic_on_grid

        def _patched(*args, **kw):
            out = _orig(*args, **kw)
            t_grid = out["t_grid"]
            i = int(np.argmin(np.abs(t_grid - t_target)))
            abs_err = np.abs(out["pred"] - out["truth"])
            out["mae"] = float(abs_err[i].mean())
            return out
        _e._evaluate_finite_horizon_critic_on_grid = _patched  # type: ignore[assignment]

    from merton_dtd.config import HorizonConfig, MertonParams, PolicyParams, TrainConfig
    from merton_dtd.training import train_fixed_policy_critic

    params = MertonParams(r=0.02, mu=0.08, sigma=cfg["sigma"], gamma=2.0, rho=0.08)
    policy = PolicyParams(pi=0.75, kappa=0.06125)
    train_cfg = TrainConfig(
        seed=cfg["seed"], batch_size=cfg["B"], num_steps=cfg["num_steps"],
        learning_rate=cfg["lr"], dt=cfg["dt"], wealth_min=0.3, wealth_max=3.0,
        eval_points=200, beta=cfg["beta"], device="cpu", log_every=cfg["log_every"],
    )
    horizon = HorizonConfig(T=cfg["T"], terminal_coef=0.0)
    _, result = train_fixed_policy_critic(
        params=params, policy=policy, train_cfg=train_cfg, loss_name=cfg["loss"],
        horizon=horizon, terminal_weight=1.0,
    )
    h = result["history"]
    return {"step": [float(x) for x in h["step"]],
            "mae":  [float(x) for x in h["mae"]],
            "tag":  cfg["tag"], "seed": cfg["seed"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--B", type=int, default=1024)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--num-steps", type=int, default=10000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--betas", type=str, default="0.25,0.5",
                   help="comma-separated β-dTD configs to plot")
    p.add_argument("--t-target", type=float, default=None,
                   help="if set, report MAE only at this t (closest grid point)")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()

    betas = [float(b) for b in args.betas.split(",")]

    # Build job list: TD (β=0) + each β-dTD config × num_seeds
    jobs: list[dict[str, Any]] = []
    for seed in range(args.num_seeds):
        common = dict(sigma=args.sigma, lr=args.lr, B=args.B, dt=args.dt,
                      num_steps=args.num_steps, log_every=args.log_every,
                      T=args.T, seed=seed, t_target=args.t_target)
        jobs.append({**common, "loss": "td", "beta": 0.0, "tag": "TD"})
        for b in betas:
            jobs.append({**common, "loss": "beta_dtd", "beta": b,
                          "tag": f"β-dTD β={b:g}"})

    print(f"Running {len(jobs)} jobs ({args.num_seeds} seeds × {1 + len(betas)} methods) "
          f"on {args.workers} workers...")
    histories: dict[str, list[dict]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(_run_one, jobs):
            histories.setdefault(res["tag"], []).append(res)

    # Plot
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    tags = ["TD"] + [f"β-dTD β={b:g}" for b in betas]
    cmap = ["C3"] + list(plt.cm.viridis(np.linspace(0.20, 0.80, len(betas))))
    for color, tag in zip(cmap, tags):
        runs = histories[tag]
        steps = np.asarray(runs[0]["step"], dtype=float)
        mae = np.stack([np.asarray(r["mae"], dtype=float) for r in runs])
        m, sd = mae.mean(0), mae.std(0)
        ax.plot(steps, m, color=color, lw=1.8, label=f"{tag}  (final {m[-1]:.3f})")
        ax.fill_between(steps, m - sd, m + sd, color=color, alpha=0.15)

    ax.set_yscale("log")
    ax.set_xlabel("training step")
    if args.t_target is not None:
        ax.set_ylabel(f"MAE at t≈{args.t_target:g} (log scale)")
        title_suffix = f"at t={args.t_target:g}"
    else:
        ax.set_ylabel(r"MAE vs. closed-form $V^\pi(t, W)$ (log scale)")
        title_suffix = "averaged over (t, W)"
    ax.set_title(
        f"Finite-horizon MAE {title_suffix}  ·  T={args.T}, σ={args.sigma}, "
        f"lr={args.lr:g}, B={args.B}, dt={args.dt}  ·  mean ± 1 SD over {args.num_seeds} seeds"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    default_out = (f"results/finite_horizon/td_vs_beta_dtd_mae_at_t{args.t_target:g}.png"
                   if args.t_target is not None
                   else "results/finite_horizon/td_vs_beta_dtd_mae_over_time.png")
    out_path = Path(args.out or default_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
