"""
Read per-run summary.json files from results/hpo_sweep/runs/, aggregate over
seeds, and produce analysis plots.

Designed to run mid-sweep — silently skips cells with missing seeds.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np


def load_rows(runs_dir: Path) -> list[dict]:
    rows = []
    for sd in sorted(runs_dir.iterdir()):
        f = sd / "summary.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        cfg = r["config"]
        # Also pull best-MAE from history.
        h_path = sd / "history.json"
        best_mae = float("nan")
        best_step = -1
        best_vwn = float("nan")
        if h_path.exists():
            h = json.loads(h_path.read_text())
            mae = h.get("mae", [])
            if mae:
                i = int(np.argmin(mae))
                best_mae = float(mae[i])
                best_step = int(h["step"][i])
                best_vwn = float(h["v_w_norm"][i])
        rows.append({
            "loss": cfg["loss"],
            "beta": float(cfg["beta"]),
            "sigma": float(cfg["sigma"]),
            "lr": float(cfg["lr"]),
            "B": int(cfg["batch_size"]),
            "seed": int(cfg["seed"]),
            "final_mae": float(r.get("mae", float("nan"))),
            "final_vwn": float(r.get("v_w_norm", float("nan"))),
            "vwn_true": float(r.get("v_w_norm_true", float("nan"))),
            "hjb_rmse": float(r.get("hjb_rmse", float("nan"))),
            "best_mae": best_mae,
            "best_step": best_step,
            "best_vwn": best_vwn,
        })
    return rows


def agg_over_seeds(rows: list[dict]) -> dict:
    """Group by (sigma, lr, B, loss, beta) and aggregate."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["sigma"], r["lr"], r["B"], r["loss"], r["beta"])
        groups[key].append(r)
    agg = {}
    for key, rs in groups.items():
        final = [r["final_mae"] for r in rs if not np.isnan(r["final_mae"])]
        best = [r["best_mae"] for r in rs if not np.isnan(r["best_mae"])]
        vwn = [r["final_vwn"] for r in rs if not np.isnan(r["final_vwn"])]
        agg[key] = {
            "n": len(rs),
            "final_mae_mean": mean(final) if final else float("nan"),
            "final_mae_std": stdev(final) if len(final) > 1 else 0.0,
            "best_mae_mean": mean(best) if best else float("nan"),
            "best_mae_std": stdev(best) if len(best) > 1 else 0.0,
            "final_vwn_mean": mean(vwn) if vwn else float("nan"),
            "vwn_true": rs[0]["vwn_true"],
        }
    return agg


def plot_mae_vs_beta(agg: dict, sigma: float, out_path: Path,
                     metric: str = "best_mae_mean", title_extra: str = "") -> bool:
    """One figure per σ. Panels: lr × B. X = β, Y = MAE. TD shown as horizontal line."""
    keys = [k for k in agg if k[0] == sigma]
    if not keys:
        return False
    lrs = sorted({k[1] for k in keys})
    Bs = sorted({k[2] for k in keys})
    betas = sorted({k[4] for k in keys if k[3] != "td"})

    fig, axes = plt.subplots(len(lrs), len(Bs), figsize=(4.0 * len(Bs), 3.2 * len(lrs)),
                              squeeze=False, sharey="row")
    for i, lr in enumerate(lrs):
        for j, B in enumerate(Bs):
            ax = axes[i][j]
            td_key = (sigma, lr, B, "td", 0.0)
            bdtd_pts = []
            for beta in betas:
                k = (sigma, lr, B, "beta_dtd", beta)
                if k in agg:
                    bdtd_pts.append((beta, agg[k][metric], agg[k][metric.replace("_mean","_std")]))
            if bdtd_pts:
                xs, ys, errs = zip(*bdtd_pts)
                ax.errorbar(xs, ys, yerr=errs, fmt="o-", color="C0",
                            label="β-dTD", capsize=3, linewidth=1.5)
            if td_key in agg:
                td_mae = agg[td_key][metric]
                td_std = agg[td_key][metric.replace("_mean","_std")]
                ax.axhline(td_mae, color="C3", ls="--", lw=1.5, label=f"TD ({td_mae:.1f})")
                ax.axhspan(td_mae - td_std, td_mae + td_std, color="C3", alpha=0.1)
            ax.set_title(f"lr={lr:g}, B={B}", fontsize=10)
            ax.set_xlabel("β")
            if j == 0:
                ax.set_ylabel(metric.replace("_mean", ""))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    fig.suptitle(f"σ={sigma}  ·  MAE vs β across (lr, batch){title_extra}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def beats_td_table(agg: dict, sigma: float, out_path: Path) -> str:
    """For each (lr, B), find the best β-dTD config and compare to TD."""
    lines = [f"=== σ={sigma}  ·  best β-dTD vs TD (best-MAE metric, mean over seeds) ===",
             f"{'lr':>8s} {'B':>6s}  {'TD MAE':>9s}  {'best β':>7s}  {'β-dTD MAE':>10s}  {'Δ':>7s}  {'win?':>5s}"]
    keys = [k for k in agg if k[0] == sigma]
    lrs = sorted({k[1] for k in keys})
    Bs = sorted({k[2] for k in keys})
    win_count = 0; total = 0
    for lr in lrs:
        for B in Bs:
            td_key = (sigma, lr, B, "td", 0.0)
            if td_key not in agg:
                continue
            td_mae = agg[td_key]["best_mae_mean"]
            best_beta = None; best_bdtd = float("inf")
            for k in keys:
                if k[1] == lr and k[2] == B and k[3] == "beta_dtd":
                    v = agg[k]["best_mae_mean"]
                    if v < best_bdtd:
                        best_bdtd = v; best_beta = k[4]
            if best_beta is None:
                continue
            delta = best_bdtd - td_mae
            win = delta < -0.5  # require >0.5 MAE improvement to call a win
            if win: win_count += 1
            total += 1
            lines.append(f"{lr:>8g} {B:>6d}  {td_mae:>9.2f}  {best_beta:>7g}  "
                         f"{best_bdtd:>10.2f}  {delta:>+7.2f}  {'YES' if win else '-':>5s}")
    lines.append(f"\nβ-dTD beat TD in {win_count}/{total} (lr,B) cells.")
    text = "\n".join(lines)
    out_path.write_text(text)
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="results/hpo_sweep")
    args = p.parse_args()

    runs_dir = Path(args.out_dir) / "runs"
    plots_dir = Path(args.out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(runs_dir)
    print(f"Loaded {len(rows)} runs")
    agg = agg_over_seeds(rows)
    sigmas = sorted({k[0] for k in agg})

    for sigma in sigmas:
        n = sum(1 for k in agg if k[0] == sigma)
        print(f"σ={sigma}: {n} (lr, B, method, β) cells")
        for metric, tag in [("best_mae_mean", "best"), ("final_mae_mean", "final")]:
            out_path = plots_dir / f"sigma{sigma:g}_{tag}_mae_vs_beta.png"
            ok = plot_mae_vs_beta(agg, sigma, out_path, metric=metric,
                                   title_extra=f"  ·  metric: {tag} MAE")
            if ok:
                print(f"  wrote {out_path}")
        table = beats_td_table(agg, sigma, plots_dir / f"sigma{sigma:g}_summary.txt")
        print()
        print(table)
        print()


if __name__ == "__main__":
    main()
