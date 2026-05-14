"""
HPO sweep over (loss, β, lr, batch_size, σ, seed) for fixed-policy Merton critic.

Hypothesis under test: is there a configuration where β-dTD beats TD on MAE, or
hits a usefully different bias/variance trade-off?

Layout:
    results/hpo_sweep/
        runs/<run_id>/{summary.json, history.json}
        runs.csv          # one row per run, scalar metrics
        meta.json         # grid + timing
        plots/*.png

Each worker process silences tqdm by monkey-patching `trange` to plain `range`.

After this, run analyze script (e.g. `scripts/analyze_hpo.py`) to read the per-run summaries and produce analysis plots.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any


class _SilentRange:
    def __init__(self, n: int, **kw: Any) -> None:
        self._it = iter(range(n))

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def set_postfix(self, *a: Any, **kw: Any) -> None:
        pass

    def close(self) -> None:
        pass


def _worker(cfg: dict[str, Any]) -> dict[str, Any]:
    # Prevent thread contention between parallel workers.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)

    # Silence tqdm in this worker before training imports it via the iterator.
    import merton_dtd.training as _t

    _t.trange = lambda n, **kw: _SilentRange(n, **kw)  # type: ignore[assignment]

    from merton_dtd.config import HorizonConfig, MertonParams, PolicyParams, TrainConfig
    from merton_dtd.training import train_fixed_policy_critic

    params = MertonParams(
        r=cfg["r"],
        mu=cfg["mu"],
        sigma=cfg["sigma"],
        gamma=cfg["gamma"],
        rho=cfg["rho"],
    )
    policy = PolicyParams(pi=cfg["pi"], kappa=cfg["kappa"])
    train_cfg = TrainConfig(
        seed=cfg["seed"],
        batch_size=cfg["batch_size"],
        num_steps=cfg["num_steps"],
        learning_rate=cfg["lr"],
        dt=cfg["dt"],
        wealth_min=cfg["wealth_min"],
        wealth_max=cfg["wealth_max"],
        eval_points=cfg["eval_points"],
        beta=cfg["beta"],
        device="cpu",
        log_every=cfg["log_every"],
    )
    horizon = (
        HorizonConfig(T=cfg["horizon"], terminal_coef=cfg["terminal_coef"])
        if cfg["horizon"] is not None
        else None
    )

    t0 = time.time()
    _, result = train_fixed_policy_critic(
        params=params,
        policy=policy,
        train_cfg=train_cfg,
        loss_name=cfg["loss"],
        horizon=horizon,
        terminal_weight=cfg["terminal_weight"],
    )
    elapsed = time.time() - t0

    run_dir = Path(cfg["out_dir"]) / "runs" / cfg["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)

    s = result["summary"]
    scalar_keys = [
        "mae",
        "rmse",
        "mape",
        "v_w_mae",
        "v_w_norm",
        "v_w_norm_true",
        "hjb_rmse",
    ]
    summary_out = {
        "run_id": cfg["run_id"],
        "config": cfg,
        "elapsed_sec": elapsed,
        "params": asdict(params),
        "policy": asdict(policy),
        "train_cfg": asdict(train_cfg),
    }
    if horizon is not None:
        summary_out["horizon"] = asdict(horizon)
    for k in scalar_keys:
        if k in s:
            summary_out[k] = float(s[k])

    (run_dir / "summary.json").write_text(json.dumps(summary_out, indent=2))

    # history: list-of-lists per metric. Serialize as JSON.
    h = result["history"]
    history_out = {k: [float(v) for v in vs] for k, vs in h.items() if vs}
    (run_dir / "history.json").write_text(json.dumps(history_out))

    # Best-MAE-during-training (early-stopping view).
    mae_hist = history_out.get("mae", [])
    if mae_hist:
        best_i = int(min(range(len(mae_hist)), key=lambda i: mae_hist[i]))
        best_mae = mae_hist[best_i]
        best_step = int(history_out["step"][best_i])
        best_vwn = float(history_out["v_w_norm"][best_i])
    else:
        best_mae = float("nan")
        best_step = -1
        best_vwn = float("nan")

    row = {
        "run_id": cfg["run_id"],
        "loss": cfg["loss"],
        "beta": cfg["beta"],
        "sigma": cfg["sigma"],
        "lr": cfg["lr"],
        "batch_size": cfg["batch_size"],
        "seed": cfg["seed"],
        "num_steps": cfg["num_steps"],
        "final_mae": summary_out.get("mae", float("nan")),
        "final_v_w_norm": summary_out.get("v_w_norm", float("nan")),
        "v_w_norm_true": summary_out.get("v_w_norm_true", float("nan")),
        "best_mae": best_mae,
        "best_step": best_step,
        "best_v_w_norm": best_vwn,
        "hjb_rmse": summary_out.get("hjb_rmse", float("nan")),
        "elapsed_sec": elapsed,
    }
    return row


def loss_for_beta(beta: float) -> str:
    if beta == 0.0:
        return "td"
    if beta == 1.0:
        return "dtd"
    return "beta_dtd"


def build_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    sigmas = [float(s) for s in args.sigmas.split(",")]
    lrs = [float(x) for x in args.lrs.split(",")]
    batches = [int(b) for b in args.batches.split(",")]
    betas = [float(b) for b in args.betas.split(",")]
    seeds = list(range(args.num_seeds))

    # TD anchor: β=0 only, no need to repeat across betas.
    # β-dTD: betas > 0.
    configs: list[dict[str, Any]] = []
    for sigma, lr, B, seed in product(sigmas, lrs, batches, seeds):
        # TD anchor
        configs.append(
            {
                "loss": "td",
                "beta": 0.0,
                "sigma": sigma,
                "lr": lr,
                "batch_size": B,
                "seed": seed,
            }
        )
        # β-dTD across betas (excluding 0; 1.0 collapses)
        for beta in betas:
            if beta <= 0.0:
                continue
            configs.append(
                {
                    "loss": loss_for_beta(beta),
                    "beta": beta,
                    "sigma": sigma,
                    "lr": lr,
                    "batch_size": B,
                    "seed": seed,
                }
            )

    # Common fields
    for i, c in enumerate(configs):
        c.update(
            {
                "r": args.r,
                "mu": args.mu,
                "gamma": args.gamma,
                "rho": args.rho,
                "pi": args.pi,
                "kappa": args.kappa,
                "num_steps": args.num_steps,
                "dt": args.dt,
                "wealth_min": 0.3,
                "wealth_max": 3.0,
                "eval_points": 200,
                "log_every": args.log_every,
                "out_dir": args.out_dir,
                "horizon": args.horizon,
                "terminal_coef": args.terminal_coef,
                "terminal_weight": args.terminal_weight,
            }
        )
        c["run_id"] = (
            f"sig{c['sigma']:g}_lr{c['lr']:g}_B{c['batch_size']}_"
            f"{c['loss']}_beta{c['beta']:g}_s{c['seed']}"
        )
    return configs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--r", type=float, default=0.02)
    p.add_argument("--mu", type=float, default=0.08)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--rho", type=float, default=0.08)
    p.add_argument("--pi", type=float, default=0.75)
    p.add_argument("--kappa", type=float, default=0.06125)

    p.add_argument("--sigmas", type=str, default="0.05,0.20")
    p.add_argument("--lrs", type=str, default="5e-4,2e-3,5e-3")
    p.add_argument("--batches", type=str, default="256,1024,4096")
    p.add_argument("--betas", type=str, default="0.1,0.25,0.5,0.75")
    p.add_argument("--num-seeds", type=int, default=3)
    p.add_argument("--num-steps", type=int, default=15000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--dt", type=float, default=1.0 / 252.0)

    # Finite-horizon options. If --horizon is passed, the critic becomes V(t, W)
    # and is evaluated against the finite-horizon closed form.
    p.add_argument("--horizon", type=float, default=None, help="terminal time T (finite horizon)")
    p.add_argument("--terminal-coef", type=float, default=0.0, help="finite horizon: bequest coefficient")
    p.add_argument("--terminal-weight", type=float, default=1.0, help="finite horizon: weight on terminal MSE")

    p.add_argument("--out-dir", type=str, default="results/hpo_sweep")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--limit", type=int, default=0, help="cap on number of runs (0=all)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    configs = build_grid(args)
    if args.limit > 0:
        configs = configs[: args.limit]

    meta = {
        "num_runs": len(configs),
        "workers": args.workers,
        "grid": {
            "sigmas": args.sigmas,
            "lrs": args.lrs,
            "batches": args.batches,
            "betas": args.betas,
            "num_seeds": args.num_seeds,
            "num_steps": args.num_steps,
            "dt": args.dt,
            "horizon": args.horizon,
            "terminal_coef": args.terminal_coef,
            "terminal_weight": args.terminal_weight,
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"Launching {len(configs)} runs on {args.workers} workers, "
        f"num_steps={args.num_steps}"
    )

    rows: list[dict[str, Any]] = []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, c): c for c in configs}
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            if i % 5 == 0 or i == len(configs):
                elapsed = time.time() - t_start
                eta = elapsed / i * (len(configs) - i)
                print(
                    f"  [{i:>4d}/{len(configs)}] {elapsed:6.0f}s elapsed, "
                    f"~{eta:6.0f}s remaining  | last: {row['run_id']}  "
                    f"best_mae={row['best_mae']:.2f}"
                )

    # Write CSV
    keys = list(rows[0].keys())
    csv_path = out_dir / "runs.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"wrote {csv_path}  ({len(rows)} rows, {time.time() - t_start:.0f}s total)")


if __name__ == "__main__":
    main()
