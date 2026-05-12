from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import torch
from torch.optim import Adam
from tqdm import trange

from .actor import GaussianActor
from .config import HorizonConfig, MertonParams, PolicyParams, PPOConfig, TrainConfig
from .critic import VanillaMLPCritic
from .eval import evaluate_critic_on_grid
from .losses import compute_loss
from .merton import (
    exact_step_tensor,
    optimal_policy_closed_form,
    reward_rate_tensor,
    terminal_value_fn,
)
from .sampling import sample_log_uniform


_PPO_INCOMPATIBLE_LOSSES = {"dtd_mean", "beta_dtd_mean"}


def build_actor(
    params: MertonParams,
    ppo_cfg: PPOConfig,
    device: str = "cpu",
    horizon: HorizonConfig | None = None,
) -> GaussianActor:
    return GaussianActor(
        params=params,
        hidden_dim=ppo_cfg.actor_hidden_dim,
        depth=ppo_cfg.actor_depth,
        activation=ppo_cfg.actor_activation,
        time_horizon=horizon.T if horizon is not None else None,
        init_log_std=ppo_cfg.init_log_std,
        kappa_floor=ppo_cfg.kappa_floor,
        state_dependent_std=ppo_cfg.state_dependent_std,
    ).to(device)


def build_critic_for_ppo(
    params: MertonParams,
    device: str = "cpu",
    horizon: HorizonConfig | None = None,
) -> VanillaMLPCritic:
    time_horizon = horizon.T if horizon is not None else None
    return VanillaMLPCritic(params, time_horizon=time_horizon).to(device)


def _compute_gae(
    rewards: torch.Tensor,        # (T, N) — step reward = U(c) * dt
    values: torch.Tensor,         # (T, N)
    dones: torch.Tensor,          # (T, N) — 1 if step transitioned into a terminal
    last_value: torch.Tensor,     # (N,)   — V(W_{T+1}) bootstrap for non-terminal
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)
    for step in reversed(range(T)):
        if step == T - 1:
            next_value = last_value
        else:
            next_value = values[step + 1]
        # If step is terminal, the bootstrap value is replaced (already encoded
        # in `values_next_bootstrap` passed via `dones`): we mask out future flow.
        nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * nonterminal - values[step]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[step] = gae
    returns = advantages + values
    return advantages, returns


def train_ppo(
    params: MertonParams,
    train_cfg: TrainConfig,
    ppo_cfg: PPOConfig,
    loss_name: str = "beta_dtd",
    horizon: HorizonConfig | None = None,
    terminal_weight: float = 1.0,
) -> tuple[GaussianActor, VanillaMLPCritic, dict[str, Any]]:
    if loss_name in _PPO_INCOMPATIBLE_LOSSES:
        raise ValueError(
            f"loss_name={loss_name} requires multiple replicas from the same "
            "state under a fixed policy and is not supported in PPO mode."
        )
    torch.manual_seed(train_cfg.seed)

    device = train_cfg.device
    dt = train_cfg.dt
    gamma_disc = math.exp(-params.rho * dt)
    finite = horizon is not None

    actor = build_actor(params, ppo_cfg, device=device, horizon=horizon)
    critic = build_critic_for_ppo(params, device=device, horizon=horizon)
    opt_actor = Adam(actor.parameters(), lr=ppo_cfg.lr_actor)
    opt_critic = Adam(critic.parameters(), lr=train_cfg.learning_rate)

    g_fn = terminal_value_fn(params, horizon) if finite else None

    # State for rollout
    wealth = sample_log_uniform(
        batch_size=ppo_cfg.n_envs,
        low=train_cfg.wealth_min,
        high=train_cfg.wealth_max,
        device=device,
    )
    t_buf = torch.zeros_like(wealth) if finite else None

    # Reference policy for evaluation/diagnostics (closed-form optimum)
    try:
        ref_policy, _ = optimal_policy_closed_form(params)
    except ValueError:
        ref_policy = PolicyParams(pi=1.0, kappa=0.05)

    history: dict[str, list[float]] = {
        "iter": [],
        "actor_loss": [],
        "critic_loss": [],
        "entropy": [],
        "kl_approx": [],
        "mean_reward_rate": [],
        "mean_return": [],
        "mean_value": [],
        "mae": [],
        "rmse": [],
        "mape": [],
        "mean_pi": [],
        "mean_kappa": [],
        "std_pi": [],
        "std_kappa": [],
    }

    T = ppo_cfg.n_steps
    N = ppo_cfg.n_envs
    iterator = trange(ppo_cfg.n_iters, desc="ppo", leave=False)
    for iteration in iterator:
        # ---- Rollout ----
        b_wealth = torch.zeros((T, N), device=device)
        b_wealth_next = torch.zeros((T, N), device=device)
        b_reward_rate = torch.zeros((T, N), device=device)
        b_raw_action = torch.zeros((T, N, 2), device=device)
        b_log_prob = torch.zeros((T, N), device=device)
        b_value = torch.zeros((T, N), device=device)
        b_done = torch.zeros((T, N), device=device)
        b_t = torch.zeros((T, N), device=device) if finite else None
        b_t_next = torch.zeros((T, N), device=device) if finite else None
        b_terminal_value_next = (
            torch.zeros((T, N), device=device) if finite else None
        )

        actor.eval()
        critic.eval()
        with torch.no_grad():
            for step in range(T):
                t_in = t_buf if finite else None
                pi_a, kappa_a, raw, log_prob = actor.sample(wealth, t_in)
                noise = torch.randn_like(wealth)
                wealth_next = exact_step_tensor(
                    wealth, params, pi_a, kappa_a, dt, noise
                )
                reward_rate_step = reward_rate_tensor(wealth, params, kappa_a)
                value = critic.value(wealth, t_in)

                b_wealth[step] = wealth
                b_wealth_next[step] = wealth_next
                b_reward_rate[step] = reward_rate_step
                b_raw_action[step] = raw
                b_log_prob[step] = log_prob
                b_value[step] = value

                if finite:
                    assert t_buf is not None and b_t is not None
                    assert b_t_next is not None and b_terminal_value_next is not None
                    t_next = torch.clamp(t_buf + dt, max=horizon.T)
                    done_mask = t_next >= horizon.T
                    b_t[step] = t_buf
                    b_t_next[step] = t_next
                    # Terminal bootstrap value for V(W_{t+dt}) at the boundary
                    tv = critic.value(wealth_next, t_next)
                    if torch.any(done_mask):
                        assert g_fn is not None
                        tv = torch.where(done_mask, g_fn(wealth_next), tv)
                    b_terminal_value_next[step] = tv
                    b_done[step] = done_mask.float()

                    # Advance / reset
                    wealth = wealth_next.clone()
                    t_buf = t_next.clone()
                    if torch.any(done_mask):
                        num_resets = int(done_mask.sum().item())
                        wealth[done_mask] = sample_log_uniform(
                            batch_size=num_resets,
                            low=train_cfg.wealth_min,
                            high=train_cfg.wealth_max,
                            device=device,
                        )
                        t_buf[done_mask] = 0.0
                else:
                    wealth = wealth_next

            # Bootstrap value for the step after the rollout
            if finite:
                last_value = critic.value(wealth, t_buf)
            else:
                last_value = critic.value(wealth)

        # ---- GAE on reward = U(c) * dt with discount exp(-rho dt) ----
        step_reward = b_reward_rate * dt
        # For finite-horizon terminal steps, the "next value" used inside GAE
        # should be the terminal payoff g(W_next), not V(W_next).
        # We splice that in by replacing values[step+1] indirectly via masking:
        # treat done as terminating the trajectory and add g(W_next) as the
        # incremental reward at the terminal step.
        if finite:
            assert b_terminal_value_next is not None
            step_reward = step_reward + b_done * gamma_disc * b_terminal_value_next
        advantages, returns = _compute_gae(
            rewards=step_reward,
            values=b_value,
            dones=b_done if finite else torch.zeros_like(b_value),
            last_value=last_value,
            gamma=gamma_disc,
            lam=ppo_cfg.gae_lambda,
        )

        # Flatten for minibatching
        flat = lambda x: x.reshape(-1, *x.shape[2:])
        f_wealth = flat(b_wealth)
        f_wealth_next = flat(b_wealth_next)
        f_reward_rate = flat(b_reward_rate)
        f_raw_action = flat(b_raw_action)
        f_log_prob_old = flat(b_log_prob)
        f_adv = flat(advantages)
        f_ret = flat(returns)
        f_t = flat(b_t) if finite else None
        f_t_next = flat(b_t_next) if finite else None
        f_terminal_value_next = flat(b_terminal_value_next) if finite else None

        adv_mean = f_adv.mean()
        adv_std = f_adv.std().clamp_min(1e-8)
        f_adv_norm = (f_adv - adv_mean) / adv_std

        n_samples = f_wealth.shape[0]
        mb = ppo_cfg.minibatch_size

        # ---- PPO updates ----
        actor.train()
        critic.train()
        last_actor_loss = 0.0
        last_critic_loss = 0.0
        last_entropy = 0.0
        last_kl = 0.0
        n_updates = 0
        for _ in range(ppo_cfg.n_epochs):
            perm = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, mb):
                idx = perm[start:start + mb]
                w_b = f_wealth[idx]
                a_b = f_raw_action[idx]
                lp_old_b = f_log_prob_old[idx]
                adv_b = f_adv_norm[idx]
                t_b = f_t[idx] if finite else None

                log_prob_new, entropy = actor.log_prob_entropy(w_b, a_b, t_b)
                ratio = torch.exp(log_prob_new - lp_old_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(
                    ratio, 1.0 - ppo_cfg.clip_eps, 1.0 + ppo_cfg.clip_eps
                ) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                ent_term = entropy.mean()
                actor_loss = policy_loss - ppo_cfg.ent_coef * ent_term

                opt_actor.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), ppo_cfg.max_grad_norm
                )
                opt_actor.step()

                # Critic update: use the configured loss on the same minibatch.
                wn_b = f_wealth_next[idx]
                r_b = f_reward_rate[idx]
                tn_b = f_t_next[idx] if finite else None
                tv_b = f_terminal_value_next[idx] if finite else None

                critic_loss, _ = compute_loss(
                    critic=critic,
                    wealth=w_b,
                    wealth_next=wn_b,
                    reward=r_b,
                    params=params,
                    dt=dt,
                    loss_name=loss_name,
                    beta=train_cfg.beta,
                    t=t_b,
                    t_next=tn_b,
                    terminal_value_next=tv_b,
                    policy=None,
                    num_replicas=train_cfg.num_replicas,
                )
                if finite and terminal_weight > 0.0:
                    assert g_fn is not None
                    term_wealth = sample_log_uniform(
                        batch_size=w_b.shape[0],
                        low=train_cfg.wealth_min,
                        high=train_cfg.wealth_max,
                        device=device,
                    )
                    term_time = torch.full_like(term_wealth, horizon.T)
                    v_terminal = critic.value(term_wealth, term_time)
                    term_mse = torch.mean(
                        (v_terminal - g_fn(term_wealth)).square()
                    )
                    critic_loss = critic_loss + terminal_weight * term_mse

                opt_critic.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    critic.parameters(), ppo_cfg.max_grad_norm
                )
                opt_critic.step()

                with torch.no_grad():
                    last_actor_loss = float(actor_loss)
                    last_critic_loss = float(critic_loss)
                    last_entropy = float(ent_term)
                    last_kl = float((lp_old_b - log_prob_new).mean())
                    n_updates += 1

        # ---- Logging / evaluation ----
        with torch.no_grad():
            dist = actor.distribution(
                f_wealth, f_t if finite else None
            )
            mean = dist.mean
            std = dist.stddev
            pi_eff, kappa_eff = actor._split_action(mean)

        eval_metrics = evaluate_critic_on_grid(
            critic=critic,
            params=params,
            policy=ref_policy,
            low=train_cfg.wealth_min,
            high=train_cfg.wealth_max,
            num=train_cfg.eval_points,
            device=device,
            horizon=horizon,
        )

        history["iter"].append(iteration)
        history["actor_loss"].append(last_actor_loss)
        history["critic_loss"].append(last_critic_loss)
        history["entropy"].append(last_entropy)
        history["kl_approx"].append(last_kl)
        history["mean_reward_rate"].append(float(b_reward_rate.mean()))
        history["mean_return"].append(float(f_ret.mean()))
        history["mean_value"].append(float(b_value.mean()))
        history["mae"].append(float(eval_metrics["mae"]))
        history["rmse"].append(float(eval_metrics["rmse"]))
        history["mape"].append(float(eval_metrics["mape"]))
        history["mean_pi"].append(float(pi_eff.mean()))
        history["mean_kappa"].append(float(kappa_eff.mean()))
        history["std_pi"].append(float(std[..., 0].mean()))
        history["std_kappa"].append(float(std[..., 1].mean()))

        iterator.set_postfix(
            pi=f"{pi_eff.mean().item():.3f}",
            kappa=f"{kappa_eff.mean().item():.3f}",
            mae=f"{eval_metrics['mae']:.2e}",
        )

    summary = {
        "final_pi_mean": history["mean_pi"][-1],
        "final_kappa_mean": history["mean_kappa"][-1],
        "final_mae": history["mae"][-1],
        "ref_pi": float(ref_policy.pi),
        "ref_kappa": float(ref_policy.kappa),
    }
    meta = {
        "params": asdict(params),
        "train_cfg": asdict(train_cfg),
        "ppo_cfg": asdict(ppo_cfg),
        "loss_name": loss_name,
    }
    if horizon is not None:
        meta["horizon"] = asdict(horizon)
        meta["terminal_weight"] = terminal_weight
    result = {"history": history, "summary": summary, "meta": meta}
    return actor, critic, result
