"""Train a single head on cached embeddings.

Usage
-----
    python scripts/02_train_head.py \\
        --embeddings cache/t2837_embeddings.pt \\
        --head two_head_nll \\
        --out outputs/real/two_head_nll \\
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uapp.data import load_cached_embeddings, make_loader
from uapp.heads import build_head
from uapp.train import TrainConfig, train_head
from uapp.utils import ensure_dir, get_device, set_seed, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument(
        "--head",
        required=True,
        choices=["mse", "two_head_nll", "single_head_nll", "fixed_sigma_nll"],
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--init-sigma-bias", type=float, default=0.5)
    parser.add_argument("--fixed-sigma", type=float, default=1.5)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss-type", choices=["auto", "gaussian", "student_t", "regularized"], default="auto")
    parser.add_argument("--student-t-nu", type=float, default=3.0)
    parser.add_argument("--sigma-prior", type=float, default=1.5)
    parser.add_argument("--lambda-reg", type=float, default=0.1)
    parser.add_argument("--ranking-lambda", type=float, default=0.0)
    parser.add_argument("--ranking-margin", type=float, default=0.05)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    log = setup_logging(args.log_level)
    set_seed(args.seed)
    device = get_device()
    out_dir = ensure_dir(args.out)

    # Load cached embeddings -------------------------------------------------
    log.info("loading cached embeddings from %s", args.embeddings)
    splits, meta = load_cached_embeddings(args.embeddings)
    if "train" not in splits or "val" not in splits:
        raise RuntimeError("cache must contain 'train' and 'val' splits")
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    d_in = int(X_train.shape[-1])
    log.info("d_in=%d | n_train=%d n_val=%d", d_in, len(y_train), len(y_val))

    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, args.batch_size, shuffle=False)

    # Build head -------------------------------------------------------------
    head_kwargs: dict = {
        "d_hidden": args.d_hidden,
        "dropout": args.dropout,
    }
    if args.head in {"two_head_nll", "single_head_nll"}:
        head_kwargs["init_sigma_bias"] = args.init_sigma_bias
    if args.head == "fixed_sigma_nll":
        head_kwargs["fixed_sigma"] = args.fixed_sigma
    head = build_head(args.head, d_in, **head_kwargs)
    log.info(
        "built %s (%d params)",
        args.head,
        sum(p.numel() for p in head.parameters()),
    )

    # Train ------------------------------------------------------------------
    cfg = TrainConfig(
        max_epochs=args.max_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        loss_type=args.loss_type,
        student_t_nu=args.student_t_nu,
        sigma_prior=args.sigma_prior,
        lambda_reg=args.lambda_reg,
        ranking_lambda=args.ranking_lambda,
        ranking_margin=args.ranking_margin,
    )
    head, history = train_head(head, train_loader, val_loader, cfg, device)

    # Save head weights + history --------------------------------------------
    torch.save(head.state_dict(), out_dir / "head.pt")
    with (out_dir / "history.json").open("w") as f:
        json.dump(asdict(history), f, indent=2)
    with (out_dir / "args.json").open("w") as f:
        json.dump(vars(args) | {"d_in": d_in}, f, indent=2, default=str)
    log.info("saved to %s", out_dir)


if __name__ == "__main__":
    main()
