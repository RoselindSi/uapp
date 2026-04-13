"""Loss functions.

The key object here is the Gaussian negative log-likelihood, which is
the probabilistic analogue of MSE. For a target y and predicted Gaussian
N(mu, sigma^2):

    NLL = (y - mu)^2 / (2 * sigma^2) + 0.5 * log(sigma^2)

(dropping the constant 0.5 * log(2*pi) that does not affect optimization).

This loss is self-regulating: the first term penalizes squared error
inversely weighted by variance, and the second term penalizes inflated
variance. The model must either predict accurately or honestly admit
high uncertainty. MSE is the special case where sigma^2 is constant.
"""
from __future__ import annotations

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
    """Gaussian negative log-likelihood, averaged over the batch.

    Parameters
    ----------
    pred_mean : (batch,) predicted mean mu_theta(x)
    pred_sigma : (batch,) predicted std sigma_theta(x), must be positive
    target : (batch,) true scalar y
    eps : small floor added to sigma^2 to avoid division by zero if the
          head's softplus output ever gets very small

    Returns
    -------
    scalar tensor: mean NLL
    """
    pred_var = pred_sigma.pow(2) + eps
    squared_error = (target - pred_mean).pow(2)
    nll = 0.5 * squared_error / pred_var + 0.5 * torch.log(pred_var)
    return nll.mean()
