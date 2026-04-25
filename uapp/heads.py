"""Output heads that consume cached graph embeddings h_G.

Four heads:
    - MSEHead         : deterministic baseline, outputs only mu
    - TwoHeadNLL      : separate MLPs for mu and sigma
    - SingleHeadNLL   : one MLP outputs [mu, raw_sigma] jointly
    - FixedSigmaNLL   : predicts mu with a fixed global sigma

All heads take embedding dimension `d` and output tensors of shape (batch,)
for mu (and sigma, where applicable). Positivity of sigma is enforced with
softplus + a small epsilon floor to prevent variance collapse.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(d_in: int, d_hidden: int, d_out: int, dropout: float) -> nn.Sequential:
    """Small 2-layer MLP with ReLU + dropout."""
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(d_hidden, d_out),
    )


class MSEHead(nn.Module):
    """Deterministic baseline: predicts only a mean."""

    def __init__(self, d_in: int, d_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = _mlp(d_in, d_hidden, 1, dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns pred_mean of shape (batch,)."""
        return self.mlp(h).squeeze(-1)


class TwoHeadNLL(nn.Module):
    """Two separate MLPs: one for mean, one for scale.

    Pro: independent capacity per output, more flexible.
    Con: more parameters, no shared features.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int = 128,
        dropout: float = 0.1,
        sigma_floor: float = 1e-6,
        init_sigma_bias: float = 0.0,
    ):
        super().__init__()
        self.mu_mlp = _mlp(d_in, d_hidden, 1, dropout)
        self.sigma_mlp = _mlp(d_in, d_hidden, 1, dropout)
        self.sigma_floor = sigma_floor
        # Optional positive bias on the final sigma layer to start sigma
        # slightly above zero (helps avoid variance collapse early in training).
        with torch.no_grad():
            self.sigma_mlp[-1].bias.fill_(init_sigma_bias)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (pred_mean, pred_sigma), each of shape (batch,)."""
        mu = self.mu_mlp(h).squeeze(-1)
        raw = self.sigma_mlp(h).squeeze(-1)
        sigma = F.softplus(raw) + self.sigma_floor
        return mu, sigma




class FixedSigmaNLL(nn.Module):
    """Predicts mean with an MLP and returns a fixed sigma for all samples.

    Useful for mentor-style experiments where uncertainty is constrained
    and mean learning is isolated.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int = 128,
        dropout: float = 0.1,
        fixed_sigma: float = 1.5,
    ):
        super().__init__()
        if fixed_sigma <= 0:
            raise ValueError("fixed_sigma must be > 0")
        self.mu_mlp = _mlp(d_in, d_hidden, 1, dropout)
        self.register_buffer("fixed_sigma", torch.tensor(float(fixed_sigma)))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu_mlp(h).squeeze(-1)
        sigma = torch.ones_like(mu) * self.fixed_sigma
        return mu, sigma

class SingleHeadNLL(nn.Module):
    """One shared MLP outputs [mu, raw_sigma] jointly.

    Pro: shared features couple mean and variance, parameter-efficient.
    Con: less flexibility; can be harder to train if objectives conflict.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int = 128,
        dropout: float = 0.1,
        sigma_floor: float = 1e-6,
        init_sigma_bias: float = 0.0,
    ):
        super().__init__()
        self.mlp = _mlp(d_in, d_hidden, 2, dropout)
        self.sigma_floor = sigma_floor
        with torch.no_grad():
            # bias[1] corresponds to raw_sigma channel
            self.mlp[-1].bias[1] = init_sigma_bias

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (pred_mean, pred_sigma), each of shape (batch,)."""
        out = self.mlp(h)  # (batch, 2)
        mu = out[..., 0]
        raw = out[..., 1]
        sigma = F.softplus(raw) + self.sigma_floor
        return mu, sigma


def build_head(name: str, d_in: int, **kwargs) -> nn.Module:
    """Factory by name: 'mse', 'two_head_nll', 'single_head_nll', 'fixed_sigma_nll'."""
    name = name.lower()
    if name == "mse":
        return MSEHead(d_in, **kwargs)
    if name == "two_head_nll":
        return TwoHeadNLL(d_in, **kwargs)
    if name == "single_head_nll":
        return SingleHeadNLL(d_in, **kwargs)
    if name == "fixed_sigma_nll":
        return FixedSigmaNLL(d_in, **kwargs)
    raise ValueError(f"unknown head: {name!r}")


def is_probabilistic(head: nn.Module) -> bool:
    """True if the head outputs (mu, sigma); False if mu only."""
    return isinstance(head, (TwoHeadNLL, SingleHeadNLL, FixedSigmaNLL))
