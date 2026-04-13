"""Run the frozen Stability Oracle backbone over T2837 and cache the
graph-level embeddings h_G to disk.

This is the only script that touches the real backbone. Everything
downstream reads from the cached .pt file.

Usage
-----
    python scripts/01_cache_embeddings.py \\
        --t2837-path /path/to/T2837.csv \\
        --backbone-checkpoint /path/to/stability_oracle.pt \\
        --out cache/t2837_embeddings.pt \\
        --val-fraction 0.15 \\
        --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make `uapp` importable when running this file as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uapp.backbone import BackboneAdapter, load_stability_oracle
from uapp.data import Mutation, load_t2837_mutations, save_cached_embeddings
from uapp.utils import get_device, set_seed, setup_logging


def split_train_val(
    train_mutations: list[Mutation],
    val_fraction: float,
    seed: int,
) -> tuple[list[Mutation], list[Mutation]]:
    """Random split of training mutations into train/val."""
    import random
    rng = random.Random(seed)
    mutations = list(train_mutations)
    rng.shuffle(mutations)
    n_val = int(len(mutations) * val_fraction)
    return mutations[n_val:], mutations[:n_val]


def build_protein_inputs(mutations: list[Mutation]) -> list:
    """Convert Mutation objects into whatever format Stability Oracle expects.

    TODO(you): this is the second spot where you need to match the
    Stability Oracle API. Their inference script constructs a graph
    from a PDB structure + a residue position. You can either:
      (a) preprocess PDBs yourself, or
      (b) reuse their existing graph-construction helpers.

    Check scripts/run_stability_oracle.py in their repo.
    """
    raise NotImplementedError(
        "build_protein_inputs is not implemented yet. See docstring."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t2837-path", required=True, type=Path)
    parser.add_argument("--backbone-checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    log = setup_logging(args.log_level)
    set_seed(args.seed)
    device = get_device()
    log.info("device: %s", device)

    # 1. Load mutations ------------------------------------------------------
    log.info("loading T2837 mutations from %s", args.t2837_path)
    all_mutations = load_t2837_mutations(args.t2837_path)
    log.info("loaded %d mutations", len(all_mutations))

    by_split: dict[str, list[Mutation]] = {"train": [], "val": [], "test": []}
    for m in all_mutations:
        by_split.setdefault(m.split, []).append(m)

    # If the CSV already has a 'val' split, respect it. Otherwise, carve one
    # out of train.
    if not by_split["val"]:
        log.info(
            "no val split in CSV, splitting %.0f%% of train into val",
            args.val_fraction * 100,
        )
        by_split["train"], by_split["val"] = split_train_val(
            by_split["train"], args.val_fraction, args.seed
        )
    for name in ("train", "val", "test"):
        log.info("  %s: %d", name, len(by_split[name]))

    # 2. Load backbone -------------------------------------------------------
    log.info("loading Stability Oracle from %s", args.backbone_checkpoint)
    model = load_stability_oracle(args.backbone_checkpoint)
    model = model.to(device)
    backbone = BackboneAdapter(model)

    # 3. Embed each split ----------------------------------------------------
    splits_out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, mutations in by_split.items():
        if not mutations:
            continue
        log.info("embedding %s split (%d mutations)", name, len(mutations))
        protein_inputs = build_protein_inputs(mutations)
        X = backbone.embed_many(protein_inputs)
        y = torch.tensor([m.ddg for m in mutations], dtype=torch.float32)
        log.info("  -> X: %s, y: %s", tuple(X.shape), tuple(y.shape))
        splits_out[name] = (X, y)

    # 4. Save ----------------------------------------------------------------
    log.info("saving cached embeddings to %s", args.out)
    save_cached_embeddings(
        args.out,
        splits_out,
        meta={
            "source": "T2837",
            "backbone": "Stability Oracle",
            "seed": args.seed,
            "val_fraction": args.val_fraction,
        },
    )
    log.info("done.")


if __name__ == "__main__":
    main()
