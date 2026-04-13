"""Training loop for heads.

Operates on cached (X, y) tensors — the backbone is never touched here.
Supports both MSE (deterministic) and NLL (probabilistic) heads through
a small dispatch.

Includes early stopping on validation loss and records per-epoch metrics
to a history dict for later inspection/plotting.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .heads import is_probabilistic
from .losses import gaussian_nll_loss, mse_loss

log = logging.getLogger("uapp.train")


@dataclass
class TrainConfig:
    max_epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 20          # early-stopping patience (epochs)
    min_delta: float = 1e-4     # min val-loss improvement to reset patience
    log_every: int = 10         # log every N epochs


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    epoch: list[int] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = -1
    stopped_epoch: int = -1


def _compute_loss(
    head: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Dispatch to the right loss based on head type."""
    if is_probabilistic(head):
        mu, sigma = head(X)
        return gaussian_nll_loss(mu, sigma, y)
    else:
        mu = head(X)
        return mse_loss(mu, y)


def _epoch_pass(
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    """Run one pass over loader. If optimizer is None, eval mode."""
    is_train = optimizer is not None
    head.train(is_train)

    total_loss = 0.0
    total_n = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        n = X_batch.size(0)

        if is_train:
            optimizer.zero_grad()
            loss = _compute_loss(head, X_batch, y_batch)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss = _compute_loss(head, X_batch, y_batch)

        total_loss += float(loss.item()) * n
        total_n += n

    return total_loss / max(total_n, 1)


def train_head(
    head: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
) -> tuple[nn.Module, TrainHistory]:
    """Train a head with early stopping on validation loss.

    Returns
    -------
    best_head : the head at its best-validation checkpoint (deep-copied)
    history : TrainHistory with per-epoch losses and best-epoch info
    """
    head = head.to(device)
    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    history = TrainHistory()
    best_state: dict | None = None
    patience_counter = 0

    for epoch in range(1, config.max_epochs + 1):
        train_loss = _epoch_pass(head, train_loader, optimizer, device)
        val_loss = _epoch_pass(head, val_loader, None, device)

        history.epoch.append(epoch)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)

        improved = val_loss < history.best_val_loss - config.min_delta
        if improved:
            history.best_val_loss = val_loss
            history.best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % config.log_every == 0 or epoch == 1:
            log.info(
                "epoch %3d | train %.4f | val %.4f | best val %.4f @ ep %d",
                epoch, train_loss, val_loss,
                history.best_val_loss, history.best_epoch,
            )

        if patience_counter >= config.patience:
            history.stopped_epoch = epoch
            log.info(
                "early stopping at epoch %d (best val %.4f @ ep %d)",
                epoch, history.best_val_loss, history.best_epoch,
            )
            break

    if best_state is not None:
        head.load_state_dict(best_state)

    return head, history
