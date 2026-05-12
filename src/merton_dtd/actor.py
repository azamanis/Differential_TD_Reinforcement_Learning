from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

from .config import MertonParams


_ACT = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}


class GaussianActor(nn.Module):
    """Diagonal-Gaussian policy over the raw action a = (a_pi, a_kappa).

    The environment uses
        pi    = a_pi
        kappa = softplus(a_kappa) + kappa_floor
    so kappa stays strictly positive while pi is unconstrained. The PPO ratio
    is computed on the *raw* action, so no Jacobian correction is needed.

    Input features mirror the critic: log-wealth, plus t/T when finite-horizon.
    """

    def __init__(
        self,
        params: MertonParams,
        hidden_dim: int = 64,
        depth: int = 2,
        activation: str = "tanh",
        time_horizon: float | None = None,
        init_log_std: float = -1.0,
        kappa_floor: float = 1e-3,
        state_dependent_std: bool = False,
    ) -> None:
        super().__init__()
        self.params = params
        self.time_horizon = float(time_horizon) if time_horizon is not None else None
        self.kappa_floor = float(kappa_floor)
        self.state_dependent_std = bool(state_dependent_std)

        if activation not in _ACT:
            raise ValueError(f"Unknown activation: {activation}")
        act_factory = _ACT[activation]

        input_dim = 1 if self.time_horizon is None else 2
        last_dim = input_dim
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(act_factory())
            last_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        out_dim = 4 if state_dependent_std else 2
        self.head = nn.Linear(last_dim, out_dim)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=0.01)

        if not state_dependent_std:
            self.log_std = nn.Parameter(torch.full((2,), float(init_log_std)))
        else:
            with torch.no_grad():
                self.head.bias[2:].fill_(float(init_log_std))

    @property
    def is_time_aware(self) -> bool:
        return self.time_horizon is not None

    def _features(
        self, wealth: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        w = torch.clamp(wealth, min=1e-12)
        x_w = torch.log(w)
        if not self.is_time_aware:
            return x_w.unsqueeze(-1)
        if t is None:
            raise ValueError("Actor was built with time_horizon set; pass `t`.")
        return torch.stack([t / self.time_horizon, x_w], dim=-1)

    def distribution(
        self, wealth: torch.Tensor, t: torch.Tensor | None = None
    ) -> Normal:
        h = self.trunk(self._features(wealth, t))
        out = self.head(h)
        mean = out[..., :2]
        if self.state_dependent_std:
            log_std = out[..., 2:]
        else:
            log_std = self.log_std.expand_as(mean)
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)
        return Normal(mean, log_std.exp())

    def sample(
        self, wealth: torch.Tensor, t: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample raw action; return (pi, kappa, raw_action, log_prob).

        log_prob is summed over the 2 action dims.
        """
        dist = self.distribution(wealth, t)
        raw = dist.rsample()
        log_prob = dist.log_prob(raw).sum(dim=-1)
        pi, kappa = self._split_action(raw)
        return pi, kappa, raw, log_prob

    def log_prob_entropy(
        self,
        wealth: torch.Tensor,
        raw_action: torch.Tensor,
        t: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(wealth, t)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    def _split_action(
        self, raw_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pi = raw_action[..., 0]
        kappa = nn.functional.softplus(raw_action[..., 1]) + self.kappa_floor
        return pi, kappa

    def deterministic_action(
        self, wealth: torch.Tensor, t: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(self._features(wealth, t))
        mean = self.head(h)[..., :2]
        return self._split_action(mean)
