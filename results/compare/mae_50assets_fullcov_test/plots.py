from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent


def _num_assets(summary: dict) -> int:
    pi = summary.get("meta", {}).get("policy", {}).get("pi", 1)
    return len(pi) if isinstance(pi, list) else 1


def _read_history(run_key: str) -> dict:
    history_path = BASE_DIR / run_key / "history.json"
    return json.loads(history_path.read_text())


def main() -> None:
    summary_path = BASE_DIR / "summary.json"
    summary = json.loads(summary_path.read_text())
    n_assets = _num_assets(summary)

    series = [
        ("td", "TD", "tab:blue"),
        ("beta_dtd_beta0.05", "beta-dTD (0.05)", "tab:orange"),
        ("beta_dtd_beta0.10", "beta-dTD (0.10)", "tab:green"),
        ("beta_dtd_beta0.20", "beta-dTD (0.20)", "tab:red"),
        ("beta_dtd_beta0.50", "beta-dTD (0.50)", "tab:purple"),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.0))

    for run_key, label, color in series:
        history = _read_history(run_key)
        ax.plot(history["step"], history["mae"], label=label, color=color, linewidth=2.0)

    ax.set_xlabel("Training step")
    ax.set_ylabel("MAE vs $V^\\pi$")
    #ax.set_yscale("log")
    ax.set_title(f"MAE over training ({n_assets} assets)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out_log = BASE_DIR / "mae_comparison_logy.png"
    out_default = BASE_DIR / "mae_comparison.png"
    fig.savefig(out_log, dpi=150)
    fig.savefig(out_default, dpi=150)

    print(f"yscale={ax.get_yscale()}")
    print(out_log)
    print(out_default)


if __name__ == "__main__":
    main()
