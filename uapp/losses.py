"""Loss functions: MSE, Gaussian NLL, Student-t NLL, and sigma regularizers.

Key losses:
    mse_loss         — deterministic baseline
    gaussian_nll_loss — standard Gaussian NLL (original)
    student_t_nll_loss — Student-t NLL with learnable or fixed degrees of freedom
    regularized_nll_loss — Gaussian NLL + sigma penalty term

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
    """Student-t negative log-likelihood.

    Heavier tails than Gaussian — more robust to outlier ddG values.
    nu (degrees of freedom):
        nu = 1  -> Cauchy (very heavy tails)
        nu = 3  -> heavy tails (good default for protein data)
        nu = 30 -> approximately Gaussian
        nu -> inf -> exactly Gaussian

    Parameters
    ----------
    pred_mean : (batch,) predicted mu
    pred_sigma : (batch,) predicted scale (positive)
    target : (batch,) true ddG
    nu : degrees of freedom (fixed, not learned)
    eps : floor for numerical stability
    """
    pred_var = pred_sigma.pow(2) + eps
    z_sq = (target - pred_mean).pow(2) / (nu * pred_var)

    # log-gamma terms
    log_gamma_half_nu_plus_1 = torch.lgamma(torch.tensor((nu + 1) / 2.0))
    log_gamma_half_nu = torch.lgamma(torch.tensor(nu / 2.0))

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
    """Gaussian NLL with a sigma regularization term.

    loss = NLL + lambda * mean((sigma - sigma_prior)^2)

    This prevents sigma from inflating to "cover up" mean errors.
    sigma_prior should be roughly the std of the target distribution.
    """
    nll = gaussian_nll_loss(pred_mean, pred_sigma, target, eps)
    sigma_penalty = lambda_reg * ((pred_sigma - sigma_prior) ** 2).mean()
    return nll + sigma_penalty
