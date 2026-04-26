"""Residual-signal diagnostic — is bio-feature signal already exhausted, or
is more available than the current σ-branch architecture extracts?

Procedure
---------
1. Train a baseline μ predictor (MSEHead) on the *train* split only.
2. Predict on val and test → compute per-sample absolute residuals
   |y - μ_pred|.  These are the things the σ-branch is trying to estimate.
3. Train a tiny MLP that takes **only the 7-d biophysical features** (no
   embedding!) and regresses against the val |residuals|.
4. Score the residual-predictor on the test split.

Decision rule
-------------
| Spearman on test residuals | What it means                             | Next step                               |
|----------------------------|--------------------------------------------|-----------------------------------------|
| ≈ 0.15 (similar to D2/D3)  | σ-branch already extracts the bio signal   | Change backbone or add data, not arch   |
| > 0.30                     | σ-branch is wasting bio signal             | Switch σ branch to FiLM / cross-attn    |
| < 0.10                     | Bio features themselves carry little       | Need richer features (SS, pLDDT, contact)|

It also reports per-feature univariate Spearman against the test |residuals|,
which tells you *which* of the 7 bio features carry the signal.

Usage
-----
    python scripts/08_residual_signal_diagnostic.py \\
        --embeddings cache/t2837_embeddings_v2.pt \\
        --bio-feats  cache/t2837_bio_features.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uapp.data import load_cached_embeddings, make_loader
from uapp.heads import MSEHead
from uapp.train import TrainConfig, train_head
from uapp.utils import get_device, set_seed, setup_logging


def predict_mse(head: MSEHead, loader, device) -> np.ndarray:
    """Run an MSEHead over a loader; return mu as a numpy array."""
    head.eval().to(device)
    mus = []
    with torch.no_grad():
        for xb, _ in loader:
            mus.append(head(xb.to(device)).cpu().numpy())
    return np.concatenate(mus)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings", required=True, type=Path)
    p.add_argument("--bio-feats",  required=True, type=Path)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience",   type=int, default=25)
    p.add_argument("--device",     type=str, default="auto",
                   help="auto | cpu | cuda | mps   (default: auto)")
    args = p.parse_args()

    log = setup_logging("INFO")
    set_seed(args.seed)
    device = get_device(device_str=args.device)
    log.info("Device: %s", device)

    # ── Load data ────────────────────────────────────────────────────────────
    splits, _ = load_cached_embeddings(args.embeddings)
    bio = torch.load(args.bio_feats, map_location="cpu", weights_only=False)
    feat_names = bio["meta"]["feature_names"]

    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["val"]
    X_te, y_te = splits["test"]

    # ── Step 1: train baseline μ predictor on TRAIN only ─────────────────────
    log.info("Step 1/4  Training baseline MSE μ-predictor on train...")
    mu_head = MSEHead(d_in=int(X_tr.shape[-1]), d_hidden=128, dropout=0.1)
    mu_head, _ = train_head(
        mu_head,
        make_loader(X_tr, y_tr, 128, shuffle=True),
        make_loader(X_va, y_va, 128, shuffle=False),
        TrainConfig(max_epochs=args.max_epochs, patience=args.patience, log_every=999),
        device,
    )

    # ── Step 2: residuals on val and test ────────────────────────────────────
    log.info("Step 2/4  Computing |residuals| on val and test...")
    mu_va = predict_mse(mu_head, make_loader(X_va, y_va, 128, shuffle=False), device)
    mu_te = predict_mse(mu_head, make_loader(X_te, y_te, 128, shuffle=False), device)
    abs_res_va = np.abs(y_va.numpy() - mu_va).astype(np.float32)
    abs_res_te = np.abs(y_te.numpy() - mu_te).astype(np.float32)
    log.info("  baseline test RMSE = %.4f", np.sqrt(np.mean((y_te.numpy() - mu_te) ** 2)))
    log.info("  mean |residual|: val=%.4f, test=%.4f", abs_res_va.mean(), abs_res_te.mean())

    # ── Step 3: tiny MLP that regresses |residual| from BIO FEATURES ONLY ───
    log.info("Step 3/4  Training |residual| predictor on bio features only (val split)...")
    bio_va = bio["val"]["feats"]                                # (n_val, 7)
    bio_te = bio["test"]["feats"]                               # (n_te,  7)

    # Re-seed so this is independent of the baseline training noise.
    set_seed(args.seed + 1)
    res_head = MSEHead(d_in=int(bio_va.shape[-1]), d_hidden=32, dropout=0.1)
    res_head, _ = train_head(
        res_head,
        make_loader(bio_va, torch.from_numpy(abs_res_va), 64, shuffle=True),
        # Tiny holdout from val itself for early stopping (the test set is sacred)
        make_loader(bio_va[-100:], torch.from_numpy(abs_res_va[-100:]), 64, shuffle=False),
        TrainConfig(max_epochs=args.max_epochs, patience=args.patience, log_every=999),
        device,
    )

    # ── Step 4: evaluate on test residuals ───────────────────────────────────
    log.info("Step 4/4  Scoring on test residuals...")
    pred_res_te = predict_mse(
        res_head, make_loader(bio_te, torch.from_numpy(abs_res_te), 64, shuffle=False), device,
    )
    sp_full, _ = spearmanr(pred_res_te, abs_res_te)

    # Univariate per-feature Spearman with test |residuals|
    univariate = []
    for i, name in enumerate(feat_names):
        s, _ = spearmanr(bio_te[:, i].numpy(), abs_res_te)
        univariate.append((name, float(s)))

    # Reference: random ranker
    rng = np.random.default_rng(args.seed)
    sp_random, _ = spearmanr(rng.permutation(pred_res_te), abs_res_te)

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Residual-signal diagnostic")
    print("=" * 72)
    print(f"Spearman(predicted |res| from bio-features-only, true |res|):  {sp_full:+.4f}")
    print(f"Spearman(random shuffle, true |res|):                          {sp_random:+.4f}")
    print()
    print("Univariate Spearman per bio feature (vs test |residual|):")
    for name, s in sorted(univariate, key=lambda t: -abs(t[1])):
        bar = "█" * int(abs(s) * 50)
        print(f"  {name:<24} {s:+.4f}  {bar}")
    print()
    print("Decision:")
    if sp_full > 0.30:
        print("  → bio features carry MORE signal than your σ-branch is extracting.")
        print("    Switch σ-branch architecture: FiLM modulation or cross-attention.")
    elif sp_full > 0.15:
        print("  → bio-feature signal is comparable to your D2/D3 σ-branch (~0.15).")
        print("    Architecture is roughly extracting available signal. Next bottleneck:")
        print("    backbone (ESM2-650M / SaProt) or more data (Megascale pretrain).")
    elif sp_full > 0.05:
        print("  → bio-feature signal is weak.  Add structural features (secondary")
        print("    structure, pLDDT, contact density) before architecture changes.")
    else:
        print("  → bio features carry essentially no residual signal.  σ ranking gains")
        print("    from D1-D3 may be statistical noise on n=170.  Re-run with seeds.")
    print("=" * 72)


if __name__ == "__main__":
    main()
