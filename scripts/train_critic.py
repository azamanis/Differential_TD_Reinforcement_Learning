from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from merton_dtd.config import HorizonConfig, MertonParams, PolicyParams, TrainConfig
from merton_dtd.plotting import plot_training_curves, plot_value_fit
from merton_dtd.rl_pinn import RLPinnConfig, train_fixed_policy_critic_rl_pinn
from merton_dtd.training import save_checkpoint, train_fixed_policy_critic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TD / dTD critic on a fixed Merton policy.")

    parser.add_argument("--r", type=float, default=0.02)
    parser.add_argument("--mu", type=float, default=0.08)
    parser.add_argument("--sigma", type=float, default=0.20)
    parser.add_argument("--gamma", type=float, default=2.0) # risk aversion param
    parser.add_argument("--rho", type=float, default=0.08) #TODO: continuos discount rate

    # Default policy is the infinite-horizon Merton-optimal policy for the given params.
    # Optimal Pi and Kappa are fully determined by the Merton parameters
    parser.add_argument("--pi", type=float, default=0.75) # fraction of wealth invested in risky asset
    parser.add_argument("--kappa", type=float, default=0.06125) # consumption = kappa * wealth

    parser.add_argument(
        "--loss",
        type=str,
        default="beta_dtd",
        choices=[
            "td", "dtd", "beta_dtd", "rl_pinn", "dtd_mean", "beta_dtd_mean",
            "naive_dtd", "beta_naive_dtd",
        ],
    )
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--num-replicas", type=int, default=1, help="K for dtd_mean / beta_dtd_mean")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-steps", type=int, default=12000)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--dt", type=float, default=1.0 / 252.0)
    parser.add_argument("--wealth-min", type=float, default=0.3)
    parser.add_argument("--wealth-max", type=float, default=3.0)
    parser.add_argument("--eval-points", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="results/train_critic")

    # RL-PINN dataset/optimization knobs (only used when --loss rl_pinn).
    # When --loss rl_pinn is selected, training uses a frozen trajectory
    # dataset (m1 initial states x m2 trajectories x N steps) and runs
    # --num-epochs SGD epochs over it instead of the streaming loop.
    parser.add_argument("--m1", type=int, default=256, help="rl_pinn: number of initial states")
    parser.add_argument("--m2", type=int, default=4, help="rl_pinn: trajectories per initial state")
    parser.add_argument("--N", type=int, default=64, help="rl_pinn: trajectory length")
    parser.add_argument("--num-epochs", type=int, default=40, help="rl_pinn: SGD epochs over dataset")
    parser.add_argument("--minibatch-size", type=int, default=4096, help="rl_pinn: minibatch size")

    # Finite-horizon options. If --horizon is passed, the critic becomes V(t, W)
    # and is evaluated against the finite-horizon closed form.
    parser.add_argument("--horizon", type=float, default=None, help="terminal time T (finite horizon)")
    parser.add_argument("--terminal-coef", type=float, default=0.0, help="finite horizon: bequest coefficient")
    parser.add_argument("--no-terminal-loss", action="store_true", help="finite horizon: disable terminal MSE")
    parser.add_argument("--terminal-weight", type=float, default=1.0, help="finite horizon: weight on terminal MSE")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    params = MertonParams(r=args.r, mu=args.mu, sigma=args.sigma, gamma=args.gamma, rho=args.rho)
    policy = PolicyParams(pi=args.pi, kappa=args.kappa)
    train_cfg = TrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        learning_rate=args.lr,
        dt=args.dt,
        wealth_min=args.wealth_min,
        wealth_max=args.wealth_max,
        eval_points=args.eval_points,
        beta=args.beta,
        device=args.device,
        log_every=args.log_every,
        num_replicas=args.num_replicas,
    )

    horizon = (
        HorizonConfig(T=args.horizon, terminal_coef=args.terminal_coef)
        if args.horizon is not None
        else None
    )

    if args.loss == "rl_pinn":
        pinn_cfg = RLPinnConfig(
            m1=args.m1,
            m2=args.m2,
            N=args.N,
            minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.lr,
            seed=args.seed,
            device=args.device,
            use_terminal_loss=not args.no_terminal_loss,
            terminal_weight=args.terminal_weight,
        )
        critic, result = train_fixed_policy_critic_rl_pinn(
            params=params,
            policy=policy,
            train_cfg=train_cfg,
            pinn_cfg=pinn_cfg,
            horizon=horizon,
        )
    else:
        critic, result = train_fixed_policy_critic(
            params=params,
            policy=policy,
            train_cfg=train_cfg,
            loss_name=args.loss,
            horizon=horizon,
            terminal_weight=0.0 if args.no_terminal_loss else args.terminal_weight,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(critic, result, out_dir)

    finite_horizon = args.horizon is not None
    plot_training_curves(result, out_dir / "training_curves.png")
    plot_value_fit(result, out_dir / "value_fit.png")

    # numpy arrays inside result["summary"] are not JSON serializable; keep only scalars in summary.json
    s = result["summary"]
    scalar_keys = ["mae", "rmse", "mape", "v_w_mae", "v_w_norm", "v_w_norm_true",
                   "hjb_rmse"]
    serializable = {
        "params": asdict(params),
        "policy": asdict(policy),
        "train_cfg": asdict(train_cfg),
        "loss": args.loss,
        "config": {
            "loss": args.loss,
            "beta": args.beta,
            "sigma": args.sigma,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "num_replicas": args.num_replicas,
        },
    }
    if finite_horizon:
        serializable["horizon"] = {"T": args.horizon, "terminal_coef": args.terminal_coef}
    for k in scalar_keys:
        if k in s:
            serializable[k] = float(s[k])
    (out_dir / "summary.json").write_text(json.dumps(serializable, indent=2))

    print("Training finished.")
    print(f"Final MAE  : {result['summary']['mae']:.6e}")
    print(f"Final RMSE : {result['summary']['rmse']:.6e}")
    print(f"Final MAPE : {result['summary']['mape']:.6e}")
    print(f"Saved artifacts to {out_dir}")


if __name__ == "__main__":
    main()
