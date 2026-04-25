"""Loss functions: MSE, Gaussian NLL, Student-t NLL, and sigma regularizers.

Key losses:
    mse_loss         — deterministic baseline
    gaussian_nll_loss — standard Gaussian NLL (original)
    student_t_nll_loss — Student-t NLL with learnable or fixed degrees of freedom
    regularized_nll_loss — Gaussian NLL + sigma penalty term
    uncertainty_ranking_loss — pairwise ranking loss for sigma vs residuals

The Gaussian NLL (Eq. 3 from proposal):
    L = (y - mu)^2 / (2*sigma^2) + 0.5 * log(sigma^2)

Student-t NLL (heavier tails, more robust to outliers):
    L = -log Gamma((nu+1)/2) + log Gamma(nu/2) + 0.5*log(nu*pi*sigma^2)
        + ((nu+1)/2) * log(1 + (y-mu)^2 / (nu*sigma^2))
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def mse_loss(pred_mean: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Plain mean squared error. Baseline."""
    return F.mse_loss(pred_mean, target)


def gaussian_nll_loss(
    pred_mean: torch.Tensor,
    pred_sigma: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Gaussian negative log-likelihood, averaged over the batch."""
    pred_var = pred_sigma.pow(2) + eps
    squared_error = (target - pred_mean).pow(2)
    nll = 0.5 * squared_error / pred_var + 0.5 * torch.log(pred_var)
    return nll.mean()


def student_t_nll_loss(
    pred_mean: torch.Tensor,
    pred_sigma: torch.Tensor,
    target: torch.Tensor,
    nu: float = 3.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Student-t negative log-likelihood."""
    pred_var = pred_sigma.pow(2) + eps
    z_sq = (target - pred_mean).pow(2) / (nu * pred_var)

    # Keep constants on same device/dtype as predictions.
    c1 = torch.tensor((nu + 1) / 2.0, device=pred_mean.device, dtype=pred_mean.dtype)
    c2 = torch.tensor(nu / 2.0, device=pred_mean.device, dtype=pred_mean.dtype)
    log_gamma_half_nu_plus_1 = torch.lgamma(c1)
    log_gamma_half_nu = torch.lgamma(c2)

    nll = (
        -log_gamma_half_nu_plus_1
        + log_gamma_half_nu
        + 0.5 * math.log(nu * math.pi)
        + 0.5 * torch.log(pred_var)
        + ((nu + 1) / 2.0) * torch.log1p(z_sq)
    )
    return nll.mean()


def regularized_nll_loss(
    pred_mean: torch.Tensor,
    pred_sigma: torch.Tensor,
    target: torch.Tensor,
    sigma_prior: float = 1.5,
    lambda_reg: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Gaussian NLL with a sigma regularization term."""
    nll = gaussian_nll_loss(pred_mean, pred_sigma, target, eps)
    sigma_penalty = lambda_reg * ((pred_sigma - sigma_prior) ** 2).mean()
    return nll + sigma_penalty


def uncertainty_ranking_loss(
    pred_mean: torch.Tensor,
    pred_sigma: torch.Tensor,
    target: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    """Pairwise hinge ranking loss between sigma and absolute residual.

    Encourages: if sample i has larger |error| than sample j, then
    sigma_i should be larger than sigma_j by at least `margin`.
    """
    abs_err = (target - pred_mean).abs()
    # Pairwise differences
    err_diff = abs_err[:, None] - abs_err[None, :]
    sig_diff = pred_sigma[:, None] - pred_sigma[None, :]

    # Only keep ordered pairs with strictly higher residual
    mask = err_diff > 0
    if not torch.any(mask):
        return torch.zeros((), device=pred_mean.device, dtype=pred_mean.dtype)

    # For those pairs we want sig_diff >= margin
    violations = F.relu(margin - sig_diff[mask])
    return violations.mean()
