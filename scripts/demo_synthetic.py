"""Run the full pipeline on synthetic data and save a reliability diagram.

This is a standalone end-to-end demo that does NOT need the Stability
Oracle backbone. It uses synthetic heteroscedastic data to exercise the
entire pipeline and produces a real reliability_diagram.png + results.json
you can look at.

Use this as a sanity check that everything works before wiring up the
real backbone.

Usage
-----
    python scripts/demo_synthetic.py --out outputs/synthetic_demo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uapp.data import make_loader, save_cached_embeddings
from uapp.evaluate import (
    evaluate_head,
    gather_probabilistic_predictions,
    plot_reliability_diagram,
    save_results_json,
)
from uapp.heads import build_head, is_probabilistic
from uapp.train import TrainConfig, train_head
from uapp.utils import ensure_dir, get_device, set_seed, setup_logging


def make_synthetic(n: int, d: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=(d,)).astype(np.float32) * 0.3
    signal = X @ w
    noise_std = 0.1 + 0.8 * np.abs(X[:, 0])
    noise = rng.normal(size=(n,)).astype(np.float32) * noise_std
    y = signal + noise
    return torch.from_numpy(X), torch.from_numpy(y)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--d", type=int, default=128, help="embedding dim")
    parser.add_argument("--n-train", type=int, default=1800)
    parser.add_argument("--n-val", type=int, default=400)
    parser.add_argument("--n-test", type=int, default=600)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log = setup_logging("INFO")
    set_seed(args.seed)
    device = get_device()
    out_dir = ensure_dir(args.out)
    log.info("device: %s", device)

    # Build synthetic splits --------------------------------------------------
    X_tr, y_tr = make_synthetic(args.n_train, args.d, seed=1)
    X_va, y_va = make_synthetic(args.n_val, args.d, seed=2)
    X_te, y_te = make_synthetic(args.n_test, args.d, seed=3)
    log.info(
        "synthetic splits: train=%d val=%d test=%d d=%d",
        args.n_train, args.n_val, args.n_test, args.d,
    )

    # Also save them as a cached .pt so you can test scripts/04_run_all.py
    save_cached_embeddings(
        out_dir / "synthetic_cache.pt",
        {"train": (X_tr, y_tr), "val": (X_va, y_va), "test": (X_te, y_te)},
        meta={"source": "synthetic", "seed": args.seed},
    )

    train_loader = make_loader(X_tr, y_tr, 128, shuffle=True)
    val_loader = make_loader(X_va, y_va, 128, shuffle=False)
    test_loader = make_loader(X_te, y_te, 128, shuffle=False)

    # Train and evaluate each head -------------------------------------------
    results = []
    reliability = {}
    cfg = TrainConfig(max_epochs=args.max_epochs, patience=args.patience, log_every=20)

    for name in ("mse", "two_head_nll", "single_head_nll"):
        log.info("=== %s ===", name)
        kwargs: dict = {"d_hidden": 64, "dropout": 0.1}
        if name != "mse":
            kwargs["init_sigma_bias"] = 0.5
        head = build_head(name, d_in=args.d, **kwargs)
        head, _ = train_head(head, train_loader, val_loader, cfg, device)

        result = evaluate_head(head, test_loader, device, name)
        results.append(result)
        log.info("\n%s", result.pretty())

        if is_probabilistic(head):
            reliability[name] = gather_probabilistic_predictions(head, test_loader, device)

    save_results_json(results, out_dir / "results.json")
    plot_reliability_diagram(
        reliability,
        out_dir / "reliability_diagram.png",
        title="Reliability diagram \u2014 synthetic data",
    )

    print("\n" + "=" * 60)
    print("RESULTS (synthetic demo)")
    print("=" * 60)
    for r in results:
        print(r.pretty())
        print()
    print(f"artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
