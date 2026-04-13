"""Evaluation metrics and reliability diagrams.

Four metrics:
    - RMSE : root mean squared error of pred_mean vs. target
    - MAE  : mean absolute error of pred_mean vs. target
    - NLL  : held-out Gaussian negative log-likelihood (probabilistic only)
    - ICE  : interval calibration error (probabilistic only)

ICE definition:
    For a grid of nominal confidence levels alpha in A,
    empirical coverage Cov_hat(1 - alpha) = fraction of test points y_i
    that fall inside the predicted interval [mu_i - z_{1-a/2} sigma_i,
                                             mu_i + z_{1-a/2} sigma_i].
    ICE = (1 / |A|) * sum_alpha | Cov_hat(1-alpha) - (1-alpha) |

Lower ICE = better calibration. A perfectly calibrated model has ICE near 0.

A reliability diagram plots Cov_hat(1-alpha) (y-axis) vs. (1-alpha) (x-axis).
The closer the curve is to the diagonal, the better calibrated.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from torch.utils.data import DataLoader

from .heads import is_probabilistic


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    head_name: str
    n_test: int
    rmse: float
    mae: float
    nll: float | None           # None for MSE head (no variance estimate)
    ice: float | None           # None for MSE head
    coverage: dict[str, float] | None  # {"0.50": 0.48, "0.90": 0.88, ...}

    def pretty(self) -> str:
        lines = [f"[{self.head_name}] n={self.n_test}"]
        lines.append(f"  RMSE : {self.rmse:.4f}")
        lines.append(f"  MAE  : {self.mae:.4f}")
        if self.nll is not None:
            lines.append(f"  NLL  : {self.nll:.4f}")
        if self.ice is not None:
            lines.append(f"  ICE  : {self.ice:.4f}")
        if self.coverage:
            cov_str = "  cov: " + ", ".join(
                f"{k}\u2192{v:.3f}" for k, v in sorted(self.coverage.items())
            )
            lines.append(cov_str)
        return "\n".join(lines)


def _gather_predictions(
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Run head over loader, return (mu, y, sigma_or_None) as numpy arrays."""
    head.eval()
    head.to(device)
    mus: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    prob = is_probabilistic(head)

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            if prob:
                mu, sigma = head(X_batch)
                sigmas.append(sigma.cpu().numpy())
            else:
                mu = head(X_batch)
            mus.append(mu.cpu().numpy())
            ys.append(y_batch.numpy())

    mu_arr = np.concatenate(mus)
    y_arr = np.concatenate(ys)
    sigma_arr = np.concatenate(sigmas) if sigmas else None
    return mu_arr, y_arr, sigma_arr


def compute_rmse(mu: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((mu - y) ** 2)))


def compute_mae(mu: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(mu - y)))


def compute_gaussian_nll(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    """Per-sample Gaussian NLL averaged over the test set (with the 2*pi term)."""
    var = np.maximum(sigma ** 2, 1e-12)
    nll = 0.5 * ((y - mu) ** 2 / var + np.log(2.0 * math.pi * var))
    return float(np.mean(nll))


def compute_coverage_curve(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical coverage for a grid of nominal levels.

    Returns
    -------
    nominal : array of (1 - alpha) values (x-axis of reliability diagram)
    empirical : array of empirical fractions of test points inside the
                corresponding prediction interval (y-axis)
    """
    if alphas is None:
        # nominal levels from 5% to 99%
        alphas = np.linspace(0.01, 0.99, 21)
    nominal = 1.0 - alphas
    empirical = np.zeros_like(nominal)
    for i, a in enumerate(alphas):
        z = norm.ppf(1.0 - a / 2.0)
        lo = mu - z * sigma
        hi = mu + z * sigma
        inside = (y >= lo) & (y <= hi)
        empirical[i] = float(np.mean(inside))
    return nominal, empirical


def compute_ice(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Interval calibration error + a coverage dict at standard levels."""
    nominal, empirical = compute_coverage_curve(mu, sigma, y, alphas)
    ice = float(np.mean(np.abs(empirical - nominal)))

    # Report coverage at standard levels for the results table
    standard_levels = [0.50, 0.80, 0.90, 0.95]
    coverage: dict[str, float] = {}
    for lvl in standard_levels:
        a = 1.0 - lvl
        z = norm.ppf(1.0 - a / 2.0)
        lo = mu - z * sigma
        hi = mu + z * sigma
        coverage[f"{lvl:.2f}"] = float(np.mean((y >= lo) & (y <= hi)))
    return ice, coverage


# ---------------------------------------------------------------------------
# Top-level evaluation entrypoint
# ---------------------------------------------------------------------------

def evaluate_head(
    head: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    head_name: str,
) -> EvalResult:
    """Run all metrics on the test set and return an EvalResult."""
    mu, y, sigma = _gather_predictions(head, test_loader, device)

    rmse = compute_rmse(mu, y)
    mae = compute_mae(mu, y)

    if sigma is None:
        return EvalResult(
            head_name=head_name,
            n_test=int(len(y)),
            rmse=rmse,
            mae=mae,
            nll=None,
            ice=None,
            coverage=None,
        )

    nll = compute_gaussian_nll(mu, sigma, y)
    ice, coverage = compute_ice(mu, sigma, y)
    return EvalResult(
        head_name=head_name,
        n_test=int(len(y)),
        rmse=rmse,
        mae=mae,
        nll=nll,
        ice=ice,
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_path: str | Path,
    title: str = "Reliability diagram (T2837 held-out)",
) -> None:
    """Save a reliability diagram comparing multiple probabilistic heads.

    Parameters
    ----------
    results : {head_name: (mu, sigma, y)} for each head to overlay
    out_path : .png path
    title : figure title
    """
    # Defer import so this module works in environments without matplotlib
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))

    # Diagonal = perfect calibration
    ax.plot([0, 1], [0, 1], color="#888888", lw=0.8, linestyle="--", label="perfect")

    colors = ["#0D7377", "#E8913A", "#2C4A6E", "#9B5FA7"]
    for i, (name, (mu, sigma, y)) in enumerate(results.items()):
        nominal, empirical = compute_coverage_curve(mu, sigma, y)
        ax.plot(
            nominal, empirical,
            marker="o", markersize=4,
            linewidth=1.5,
            color=colors[i % len(colors)],
            label=name,
        )

    ax.set_xlabel("Nominal coverage (1 \u2212 \u03b1)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_results_json(results: list[EvalResult], out_path: str | Path) -> None:
    """Dump a list of EvalResults to JSON."""
    payload = [asdict(r) for r in results]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)


def gather_probabilistic_predictions(
    head: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mu, sigma, y) numpy arrays for a probabilistic head."""
    mu, y, sigma = _gather_predictions(head, test_loader, device)
    assert sigma is not None, "head must be probabilistic"
    return mu, sigma, y
