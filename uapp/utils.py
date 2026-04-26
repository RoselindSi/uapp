"""Small utilities: seeding, device selection, logging."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer_cuda: bool = True, device_str: str | None = None) -> torch.device:
    """Resolve a torch device.

    Parameters
    ----------
    prefer_cuda : if True, prefer CUDA over MPS over CPU on auto-resolve.
    device_str  : optional explicit override.  Accepts ``"auto"``, ``"cpu"``,
                  ``"cuda"``, ``"mps"``.  If unset or ``"auto"``, picks the
                  best available accelerator.

    Resolution order on auto: CUDA → MPS (Apple Silicon) → CPU.
    """
    if device_str and device_str.lower() != "auto":
        return torch.device(device_str.lower())

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure a plain logger that prints to stdout."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("uapp")


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing and return a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
