"""Training loop for heads.

Supports three training modes:
    1. Standard: train mu and sigma jointly (original)
    2. Two-stage: train mu first (MSE), then freeze mu and train sigma (NLL)
    3. Student-t: same as standard but with Student-t likelihood

Operates on cached (X, y) tensors — the backbone is never touched here.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .heads import is_probabilistic, MSEHead, TwoHeadNLL, SingleHeadNLL
from .losses import (
    gaussian_nll_loss,
    mse_loss,
    regularized_nll_loss,
    student_t_nll_loss,
)

log = logging.getLogger("uapp.train")


@dataclass
class TrainConfig:
    max_epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 30
    min_delta: float = 1e-4
    log_every: int = 10
    # Loss configuration
    loss_type: Literal["auto", "gaussian", "student_t", "regularized"] = "auto"
    # Student-t degrees of freedom
    student_t_nu: float = 3.0
    # Sigma regularization parameters
    sigma_prior: float = 1.5
    lambda_reg: float = 0.1
    # Sigma clamp range (None = no clamping)
    sigma_min: float | None = None
    sigma_max: float | None = None


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    epoch: list[int] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_epoch: int = -1
    stopped_epoch: int = -1
    stage: str = ""


def _clamp_sigma(head: nn.Module, sigma_min: float | None, sigma_max: float | None):
    """Post-forward hook to clamp sigma output if bounds are set."""
    if sigma_min is None and sigma_max is None:
        return
    # We'll apply clamping inside _compute_loss instead (cleaner)


def _compute_loss(
    head: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    config: TrainConfig,
) -> torch.Tensor:
    """Dispatch to the right loss based on head type and config."""
    if not is_probabilistic(head):
        mu = head(X)
        return mse_loss(mu, y)

    mu, sigma = head(X)

    # Apply sigma clamping if configured
    if config.sigma_min is not None or config.sigma_max is not None:
        sigma = torch.clamp(
            sigma,
            min=config.sigma_min if config.sigma_min is not None else 0.0,
            max=config.sigma_max if config.sigma_max is not None else 100.0,
        )

    # Select loss function
    loss_type = config.loss_type
    if loss_type == "auto":
        loss_type = "gaussian"

    if loss_type == "gaussian":
        return gaussian_nll_loss(mu, sigma, y)
    elif loss_type == "student_t":
        return student_t_nll_loss(mu, sigma, y, nu=config.student_t_nu)
    elif loss_type == "regularized":
        return regularized_nll_loss(
            mu, sigma, y,
            sigma_prior=config.sigma_prior,
            lambda_reg=config.lambda_reg,
        )
    else:
        raise ValueError(f"unknown loss_type: {loss_type}")


def _epoch_pass(
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    config: TrainConfig,
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
            loss = _compute_loss(head, X_batch, y_batch, config)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss = _compute_loss(head, X_batch, y_batch, config)

        total_loss += float(loss.item()) * n
        total_n += n

    return total_loss / max(total_n, 1)


def _run_training_loop(
    head: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    stage_label: str = "",
) -> tuple[nn.Module, TrainHistory]:
    """Core training loop with early stopping."""
    history = TrainHistory(stage=stage_label)
    best_state = None
    patience_counter = 0

    for epoch in range(1, config.max_epochs + 1):
        train_loss = _epoch_pass(head, train_loader, optimizer, device, config)
        val_loss = _epoch_pass(head, val_loader, None, device, config)

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
                "%s epoch %3d | train %.4f | val %.4f | best %.4f @ ep %d",
                stage_label, epoch, train_loss, val_loss,
                history.best_val_loss, history.best_epoch,
            )

        if patience_counter >= config.patience:
            history.stopped_epoch = epoch
            log.info(
                "%s early stop at epoch %d (best %.4f @ ep %d)",
                stage_label, epoch, history.best_val_loss, history.best_epoch,
            )
            break

    if best_state is not None:
        head.load_state_dict(best_state)

    return head, history


def train_head(
    head: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
) -> tuple[nn.Module, TrainHistory]:
    """Train a head with early stopping. Standard single-stage training."""
    head = head.to(device)
    optimizer = torch.optim.Adam(
        head.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    return _run_training_loop(
        head, train_loader, val_loader, config, device, optimizer, stage_label="[train]",
    )


def train_head_two_stage(
    head: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config_stage1: TrainConfig,
    config_stage2: TrainConfig,
    device: torch.device,
) -> tuple[nn.Module, TrainHistory, TrainHistory]:
    """Two-stage training: first mu (MSE), then sigma (NLL) with mu frozen.

    Stage 1: Train all parameters with MSE loss to learn a good mu.
    Stage 2: Freeze mu parameters, train only sigma parameters with NLL.

    This decouples mean and variance learning, preventing sigma from
    being used to "cover up" mean prediction errors.
    """
    head = head.to(device)

    if not is_probabilistic(head):
        raise ValueError("two-stage training requires a probabilistic head")

    # ── Stage 1: train mu with MSE ──────────────────────────────────
    log.info("=== Stage 1: training mu (MSE) ===")

    # For stage 1, we need a temporary MSE config
    mse_config = TrainConfig(
        max_epochs=config_stage1.max_epochs,
        lr=config_stage1.lr,
        weight_decay=config_stage1.weight_decay,
        patience=config_stage1.patience,
        min_delta=config_stage1.min_delta,
        log_every=config_stage1.log_every,
        loss_type="auto",  # will dispatch to MSE for non-probabilistic
    )

    # We need to train only the mean pathway. Strategy: use the full head
    # but compute MSE loss using only the mu output.
    # Temporarily override _compute_loss behavior by training with gaussian
    # NLL but with sigma fixed at a large constant (equivalent to MSE up to scale).

    # Simpler approach: just train all params with MSE-like loss
    # by setting sigma_min = sigma_max = constant
    stage1_config = TrainConfig(
        max_epochs=config_stage1.max_epochs,
        lr=config_stage1.lr,
        weight_decay=config_stage1.weight_decay,
        patience=config_stage1.patience,
        min_delta=config_stage1.min_delta,
        log_every=config_stage1.log_every,
        loss_type="gaussian",
        sigma_min=1.5,  # fix sigma to a constant
        sigma_max=1.5,
    )

    optimizer1 = torch.optim.Adam(
        head.parameters(), lr=config_stage1.lr, weight_decay=config_stage1.weight_decay,
    )
    head, history1 = _run_training_loop(
        head, train_loader, val_loader, stage1_config, device, optimizer1,
        stage_label="[stage1-mu]",
    )

    # ── Stage 2: freeze mu, train sigma with NLL ────────────────────
    log.info("=== Stage 2: training sigma (NLL, mu frozen) ===")

    # Identify which parameters belong to mu vs sigma
    if isinstance(head, TwoHeadNLL):
        mu_params = set(id(p) for p in head.mu_mlp.parameters())
        sigma_params = list(head.sigma_mlp.parameters())
    elif isinstance(head, SingleHeadNLL):
        # For single-head, we can't cleanly separate mu/sigma params
        # because they share layers. Instead, we train all params but
        # with a lower learning rate — the mu pathway is already good
        # from stage 1, so it will change slowly while sigma adapts.
        mu_params = set()
        sigma_params = list(head.parameters())
        log.info("  (single-head: training all params at lower lr)")
    else:
        raise ValueError(f"unsupported head type: {type(head)}")

    # Freeze mu params (two-head only)
    for p in head.parameters():
        if id(p) in mu_params:
            p.requires_grad = False

    # Only optimize sigma params
    trainable = [p for p in head.parameters() if p.requires_grad]
    stage2_lr = config_stage2.lr
    if isinstance(head, SingleHeadNLL):
        stage2_lr = config_stage2.lr * 0.1  # lower lr for shared params

    optimizer2 = torch.optim.Adam(
        trainable, lr=stage2_lr, weight_decay=config_stage2.weight_decay,
    )

    head, history2 = _run_training_loop(
        head, train_loader, val_loader, config_stage2, device, optimizer2,
        stage_label="[stage2-sigma]",
    )

    # Unfreeze everything for future use
    for p in head.parameters():
        p.requires_grad = True

    return head, history1, history2
