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

import math

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

    When ``learn_nu=True`` the head also trains a global scalar degrees-of-
    freedom parameter (log_nu → nu via exp + clamp).  This is used by the
    training loop when ``loss_type="student_t"`` and ``student_t_nu <= 0``.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int = 128,
        dropout: float = 0.1,
        sigma_floor: float = 1e-6,
        init_sigma_bias: float = 0.0,
        learn_nu: bool = False,
        init_nu: float = 3.0,
    ):
        super().__init__()
        self.mu_mlp = _mlp(d_in, d_hidden, 1, dropout)
        self.sigma_mlp = _mlp(d_in, d_hidden, 1, dropout)
        self.sigma_floor = sigma_floor
        with torch.no_grad():
            self.sigma_mlp[-1].bias.fill_(init_sigma_bias)

        self.learn_nu = learn_nu
        if learn_nu:
            # Parameterize as log_nu so nu stays positive; clamped > 2 in the loss.
            self.log_nu = nn.Parameter(torch.tensor(math.log(max(init_nu, 2.01))))
        else:
            self.log_nu = None

    @property
    def nu(self) -> torch.Tensor | None:
        """Current (differentiable) nu scalar, or None if nu is not learned."""
        if self.log_nu is None:
            return None
        return torch.exp(self.log_nu)

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


class FeatureAugmentedHead(nn.Module):
    """μ branch sees only ``h_G``; σ branch additionally sees biophysical features.

    Designed for Track-D ablations: the mean pathway is held constant across
    feature-set variants, so any change in calibration / ranking metrics is
    attributable to the variance branch alone.

    The forward signature is ``forward(h, extra=None)``: when ``extra`` is
    provided it is concatenated onto ``h`` for the σ MLP only.

    Parameters
    ----------
    d_in        : embedding dimension
    d_extra     : dimensionality of the extra features (0 = baseline)
    d_hidden    : MLP hidden size
    dropout     : MLP dropout
    sigma_floor : lower bound added to softplus(raw_sigma)
    init_sigma_bias : positive bias on the final sigma layer to prevent collapse
    """

    def __init__(
        self,
        d_in: int,
        d_extra: int = 0,
        d_hidden: int = 128,
        dropout: float = 0.1,
        sigma_floor: float = 1e-6,
        init_sigma_bias: float = 0.5,
    ):
        super().__init__()
        if d_extra < 0:
            raise ValueError("d_extra must be >= 0")
        self.d_extra = d_extra
        self.sigma_floor = sigma_floor

        self.mu_mlp    = _mlp(d_in, d_hidden, 1, dropout)
        self.sigma_mlp = _mlp(d_in + d_extra, d_hidden, 1, dropout)
        with torch.no_grad():
            self.sigma_mlp[-1].bias.fill_(init_sigma_bias)

    def forward(
        self,
        h: torch.Tensor,
        extra: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (pred_mean, pred_sigma), each shape (batch,)."""
        mu = self.mu_mlp(h).squeeze(-1)

        if self.d_extra == 0:
            h_sigma = h
        else:
            if extra is None:
                raise ValueError(
                    f"FeatureAugmentedHead expects {self.d_extra} extra features, got None"
                )
            if extra.shape[-1] != self.d_extra:
                raise ValueError(
                    f"extra has {extra.shape[-1]} features, expected {self.d_extra}"
                )
            h_sigma = torch.cat([h, extra.float()], dim=-1)

        raw = self.sigma_mlp(h_sigma).squeeze(-1)
        sigma = F.softplus(raw) + self.sigma_floor
        return mu, sigma


def build_head(name: str, d_in: int, **kwargs) -> nn.Module:
    """Factory by name.

    Supported names:
        'mse', 'two_head_nll', 'single_head_nll', 'fixed_sigma_nll',
        'feature_augmented_nll'.

    ``two_head_nll`` accepts ``learn_nu=True`` to add a trainable global
    degrees-of-freedom parameter for the Student-t loss.

    ``feature_augmented_nll`` accepts ``d_extra=<int>`` for the size of the
    biophysical-feature vector concatenated to the σ branch only.
    """
    name = name.lower()
    if name == "mse":
        return MSEHead(d_in, **kwargs)
    if name == "two_head_nll":
        return TwoHeadNLL(d_in, **kwargs)
    if name == "single_head_nll":
        return SingleHeadNLL(d_in, **kwargs)
    if name == "fixed_sigma_nll":
        return FixedSigmaNLL(d_in, **kwargs)
    if name == "feature_augmented_nll":
        return FeatureAugmentedHead(d_in, **kwargs)
    raise ValueError(f"unknown head: {name!r}")


def is_probabilistic(head: nn.Module) -> bool:
    """True if the head outputs (mu, sigma); False if mu only."""
    return isinstance(head, (TwoHeadNLL, SingleHeadNLL, FixedSigmaNLL, FeatureAugmentedHead))
