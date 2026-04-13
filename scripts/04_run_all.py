"""End-to-end: train and evaluate all three heads on cached embeddings,
then produce a results table + reliability diagram comparing them.

This is the script you run most often once the cache exists.

Usage
-----
    python scripts/04_run_all.py \\
        --embeddings cache/t2837_embeddings.pt \\
        --out outputs/real \\
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
from uapp.evaluate import (
    evaluate_head,
    gather_probabilistic_predictions,
    plot_reliability_diagram,
    save_results_json,
)
from uapp.heads import build_head, is_probabilistic
from uapp.train import TrainConfig, train_head
from uapp.utils import ensure_dir, get_device, set_seed, setup_logging


HEAD_NAMES = ("mse", "two_head_nll", "single_head_nll")


def train_and_eval_one(
    head_name: str,
    splits: dict,
    d_in: int,
    head_kwargs: dict,
    cfg: TrainConfig,
    device,
    out_dir: Path,
    batch_size: int,
):
    """Train one head and evaluate it on the test split."""
    log = setup_logging()
    log.info("=== %s ===", head_name)

    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["val"]
    X_te, y_te = splits["test"]
    train_loader = make_loader(X_tr, y_tr, batch_size, shuffle=True)
    val_loader = make_loader(X_va, y_va, batch_size, shuffle=False)
    test_loader = make_loader(X_te, y_te, batch_size, shuffle=False)

    kwargs = dict(head_kwargs)
    if head_name == "mse":
        kwargs.pop("init_sigma_bias", None)
    head = build_head(head_name, d_in, **kwargs)
    log.info("  built %s (%d params)", head_name, sum(p.numel() for p in head.parameters()))

    head, history = train_head(head, train_loader, val_loader, cfg, device)

    head_out = ensure_dir(out_dir / head_name)
    torch.save(head.state_dict(), head_out / "head.pt")
    with (head_out / "history.json").open("w") as f:
        json.dump(asdict(history), f, indent=2)

    result = evaluate_head(head, test_loader, device, head_name=head_name)
    log.info("\n%s", result.pretty())
    with (head_out / "eval.json").open("w") as f:
        json.dump(asdict(result), f, indent=2)

    reliability_data = None
    if is_probabilistic(head):
        mu, sigma, y = gather_probabilistic_predictions(head, test_loader, device)
        reliability_data = (mu, sigma, y)

    return result, reliability_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--init-sigma-bias", type=float, default=0.5)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    log = setup_logging(args.log_level)
    set_seed(args.seed)
    device = get_device()
    out_dir = ensure_dir(args.out)
    log.info("device: %s", device)

    splits, _meta = load_cached_embeddings(args.embeddings)
    for name in ("train", "val", "test"):
        if name not in splits:
            raise RuntimeError(f"cache missing '{name}' split")
    d_in = int(splits["train"][0].shape[-1])
    log.info("d_in=%d", d_in)

    head_kwargs = {
        "d_hidden": args.d_hidden,
        "dropout": args.dropout,
        "init_sigma_bias": args.init_sigma_bias,
    }
    cfg = TrainConfig(
        max_epochs=args.max_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    results = []
    reliability_inputs: dict = {}
    for head_name in HEAD_NAMES:
        result, rel = train_and_eval_one(
            head_name, splits, d_in, head_kwargs, cfg, device, out_dir, args.batch_size
        )
        results.append(result)
        if rel is not None:
            reliability_inputs[head_name] = rel

    # Save combined results JSON + reliability diagram
    save_results_json(results, out_dir / "results.json")
    log.info("saved combined results to %s", out_dir / "results.json")

    if reliability_inputs:
        plot_reliability_diagram(
            reliability_inputs,
            out_dir / "reliability_diagram.png",
            title="Reliability diagram — T2837 test set",
        )
        log.info("saved reliability diagram to %s", out_dir / "reliability_diagram.png")

    # Print results table to stdout
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for r in results:
        print(r.pretty())
        print()


if __name__ == "__main__":
    main()
