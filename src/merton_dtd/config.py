from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MertonParams:
    """Parameters for the infinite-horizon CRRA Merton problem.

    Wealth dynamics under a constant policy (pi, kappa):
        dW_t / W_t = (r + pi (mu - r) - kappa) dt + pi sigma dB_t
    with consumption c_t = kappa W_t.

    Objective:
        discounted expected utility of consumption in continuous time

    Utility is CRRA with risk aversion gamma != 1:
        U(c) = c^(1-gamma) / (1-gamma)
    """

    r: float = 0.02
    mu: float = 0.08
    sigma: float = 0.20
    gamma: float = 2.0
    rho: float = 0.08

    def __post_init__(self) -> None:
        if self.sigma < 0.0:
            raise ValueError("sigma must be non-negative")
        if self.gamma <= 0.0 or abs(self.gamma - 1.0) < 1e-12:
            raise ValueError("gamma must be positive and different from 1")
        if self.rho <= 0.0:
            raise ValueError("rho must be strictly positive")


@dataclass(frozen=True)
class PolicyParams:
    """Constant policy parameters.

    pi: portfolio weight in the risky asset.
    kappa: consumption rate as a fraction of wealth, so c = kappa * W.
    """

    pi: float
    kappa: float

    def __post_init__(self) -> None:
        if self.kappa <= 0.0:
            raise ValueError("kappa must be strictly positive")


@dataclass(frozen=True)
class HorizonConfig:
    """Finite-horizon configuration.

    The CRRA value function in finite horizon takes the form
        V(t, W) = A(t) * W^{1-gamma} / (1-gamma),
    with A(t) determined by an ODE backward from the terminal condition
        V(T, W) = terminal_coef * W^{1-gamma} / (1-gamma).
    `terminal_coef = 0` is the standard "no bequest" case; setting it to the
    infinite-horizon coefficient kappa^{1-gamma}/D collapses A(t) to a constant
    and recovers the stationary problem.
    """

    T: float
    terminal_coef: float = 0.0

    def __post_init__(self) -> None:
        if self.T <= 0.0:
            raise ValueError("Horizon T must be strictly positive")


@dataclass(frozen=True)
class PPOConfig:
    """PPO hyperparameters.

    The actor outputs a 2-D Gaussian over the raw action a = (a_pi, a_kappa).
    Dynamics use pi = a_pi (unconstrained, leverage/shorts allowed) and
    kappa = softplus(a_kappa) + kappa_floor (strictly positive).
    """

    n_iters: int = 200
    n_steps: int = 64
    n_envs: int = 256
    n_epochs: int = 10
    minibatch_size: int = 1024
    clip_eps: float = 0.2
    gae_lambda: float = 0.95
    ent_coef: float = 0.0
    value_coef: float = 0.5
    lr_actor: float = 3e-4
    max_grad_norm: float = 0.5
    init_log_std: float = -1.0
    kappa_floor: float = 1e-3
    actor_hidden_dim: int = 64
    actor_depth: int = 2
    actor_activation: str = "tanh"
    state_dependent_std: bool = False


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    batch_size: int = 512
    num_steps: int = 4000
    learning_rate: float = 5e-3
    dt: float = 1.0 / 252.0
    wealth_min: float = 0.3
    wealth_max: float = 3.0
    eval_points: int = 200
    beta: float = 0.5
    device: str = "cpu"
    log_every: int = 100
    num_replicas: int = 1
