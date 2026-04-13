"""Evaluate a trained head on the test split.

Usage
-----
    python scripts/03_evaluate.py \\
        --embeddings cache/t2837_embeddings.pt \\
        --head two_head_nll \\
        --checkpoint outputs/real/two_head_nll/head.pt \\
        --out outputs/real/two_head_nll \\
        --d-hidden 128
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
from uapp.evaluate import evaluate_head
from uapp.heads import build_head
from uapp.utils import ensure_dir, get_device, set_seed, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument(
        "--head",
        required=True,
        choices=["mse", "two_head_nll", "single_head_nll"],
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    log = setup_logging(args.log_level)
    set_seed(args.seed)
    device = get_device()
    out_dir = ensure_dir(args.out)

    splits, _meta = load_cached_embeddings(args.embeddings)
    if "test" not in splits:
        raise RuntimeError("cache must contain a 'test' split")
    X_test, y_test = splits["test"]
    d_in = int(X_test.shape[-1])
    test_loader = make_loader(X_test, y_test, args.batch_size, shuffle=False)

    head_kwargs: dict = {"d_hidden": args.d_hidden, "dropout": args.dropout}
    head = build_head(args.head, d_in, **head_kwargs)
    head.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    result = evaluate_head(head, test_loader, device, head_name=args.head)
    log.info("\n%s", result.pretty())

    with (out_dir / "eval.json").open("w") as f:
        json.dump(asdict(result), f, indent=2)


if __name__ == "__main__":
    main()
