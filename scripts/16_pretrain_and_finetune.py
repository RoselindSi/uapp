"""Pretrain D5 head on Megascale → fine-tune on T2837.

Pipeline
========
1. Pretrain a `FeatureAugmentedHead` (D5 = RSA + chemistry + sequence-struct,
   k=13) on Megascale using train/val splits already encoded in the bio
   feature file.  Loss = Student-t NLL (ν=3).  Save best-val checkpoint.
2. Init a fresh `FeatureAugmentedHead` (same architecture) on T2837 and
   load the Megascale-pretrained weights as the starting point.
3. Fine-tune on T2837 train with a smaller learning rate (default 5e-5,
   10× lower than from-scratch) to avoid catastrophic forgetting.
4. Evaluate on T2837 test.  Compare to the from-scratch D5 baseline numbers
   and print the deliverable-table metrics + Δ vs baseline.

Architecture notes
==================
- Pretrain and fine-tune use the same head architecture (D5, d_extra=13).
- Megascale doesn't have RSA (filler 0.5 column) — the σ branch's RSA
  pathway gets little signal during pretraining.  T2837 fine-tune adds the
  real signal back in.
- Bio-feature standardisation is per-dataset (train stats of each).  The
  σ branch sees z-scored inputs in both phases, so the input distribution
  shift is reduced.
- LR sweep is exposed via --finetune-lr; default 5e-5 is a conservative
  starting point.

Outputs (under --out)
=====================
- pretrain/head_state.pt           best-val Megascale checkpoint
- pretrain/training_log.json       per-epoch losses
- finetune/head_state.pt           fine-tuned T2837 checkpoint
- finetune/training_log.json       per-epoch losses
- t2837_test_metrics.json          deliverable-table metrics on T2837 test
- comparison.json                  Δ vs from-scratch D5 baseline (if provided)
- t2837_test_predictions.npz       mu, sigma, y for offline analysis

Usage
-----
    python scripts/16_pretrain_and_finetune.py \\
        --megascale-emb cache/megascale_embeddings_650m.pt \\
        --megascale-bio cache/megascale_bio_features_650m_extended.pt \\
        --t2837-emb     cache/t2837_embeddings_v2_650m.pt \\
        --t2837-bio     cache/t2837_bio_features_650m_extended.pt \\
        --out           outputs/megascale_pretrain \\
        --device        cuda
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uapp.data import load_cached_embeddings
from uapp.evaluate import (
    compute_gaussian_nll, compute_ice, compute_mae, compute_rmse,
    compute_spearman_sigma_error, compute_top_k_risk_capture,
)
from uapp.heads import FeatureAugmentedHead
from uapp.losses import student_t_nll_loss, uncertainty_ranking_loss
from uapp.utils import ensure_dir, get_device, set_seed, setup_logging


# ─────────────────────────────────────────────────────────────────────────────
# Bio-feature slicing (D5 = RSA + chemistry + sequence-struct, k=13)
# Identical to scripts/07/10/11.
# ─────────────────────────────────────────────────────────────────────────────
RSA_IDX = 0
BIO_IDX = list(range(1, 7))      # chemistry
EXT_IDX = list(range(7, 13))     # sequence-derived structural


def select_d5(feats: torch.Tensor) -> torch.Tensor:
    return feats[:, [RSA_IDX] + BIO_IDX + EXT_IDX]


# ─────────────────────────────────────────────────────────────────────────────
# Loaders / training / prediction
# ─────────────────────────────────────────────────────────────────────────────
def make_loader(X, y, extra, batch_size, shuffle):
    ds = TensorDataset(X.float(), y.float(), extra.float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_head(
    head, train_loader, val_loader, device,
    *, max_epochs, lr, weight_decay, patience, nu,
    ranking_lambda, ranking_margin, log, label,
) -> tuple[FeatureAugmentedHead, dict]:
    head = head.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, ctr = float("inf"), None, 0
    best_epoch = -1
    history = []
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        ep_t0 = time.time()
        head.train()
        for xb, yb, eb in train_loader:
            xb, yb, eb = xb.to(device), yb.to(device), eb.to(device)
            opt.zero_grad()
            mu, sig = head(xb, eb)
            loss = student_t_nll_loss(mu, sig, yb, nu=nu)
            if ranking_lambda > 0.0:
                loss = loss + ranking_lambda * uncertainty_ranking_loss(
                    pred_mean=mu, pred_sigma=sig, target=yb, margin=ranking_margin,
                )
            loss.backward(); opt.step()

        head.eval()
        v_loss, v_n = 0.0, 0
        with torch.no_grad():
            for xb, yb, eb in val_loader:
                xb, yb, eb = xb.to(device), yb.to(device), eb.to(device)
                mu, sig = head(xb, eb)
                v_loss += float(student_t_nll_loss(mu, sig, yb, nu=nu).item()) * xb.size(0)
                v_n += xb.size(0)
        v_loss /= max(v_n, 1)
        ep_dur = time.time() - ep_t0
        history.append({"epoch": epoch, "val_loss": v_loss, "duration_s": ep_dur})

        if v_loss < best_val - 1e-4:
            best_val, best_state, best_epoch, ctr = (
                v_loss, copy.deepcopy(head.state_dict()), epoch, 0
            )
        else:
            ctr += 1

        if epoch == 1 or epoch % 5 == 0 or ctr >= patience:
            log.info("[%s] epoch %3d/%3d  val %.4f  best %.4f @ ep %d  (%.1fs)",
                     label, epoch, max_epochs, v_loss, best_val, best_epoch, ep_dur)

        if ctr >= patience:
            log.info("[%s] early stop at epoch %d", label, epoch)
            break

    log.info("[%s] training done in %.1f min", label, (time.time() - t0) / 60)
    if best_state is not None:
        head.load_state_dict(best_state)
    return head, {"history": history, "best_val": best_val, "best_epoch": best_epoch}


def predict(head, loader, device):
    head.eval().to(device)
    mus, sigs, ys = [], [], []
    with torch.no_grad():
        for xb, yb, eb in loader:
            xb, eb = xb.to(device), eb.to(device)
            mu, sig = head(xb, eb)
            mus.append(mu.cpu().numpy())
            sigs.append(sig.cpu().numpy())
            ys.append(yb.numpy())
    return np.concatenate(mus), np.concatenate(sigs), np.concatenate(ys)


def metrics_dict(name: str, mu, sigma, y) -> dict:
    rmse = compute_rmse(mu, y); mae = compute_mae(mu, y)
    nll  = compute_gaussian_nll(mu, sigma, y)
    ice, cov = compute_ice(mu, sigma, y)
    sp = compute_spearman_sigma_error(mu, sigma, y)
    tk = compute_top_k_risk_capture(mu, sigma, y, k_fracs=[0.10, 0.20, 0.30])
    return {
        "name": name, "n": int(len(y)),
        "rmse": rmse, "mae": mae, "nll": nll, "ice": ice,
        "cov@0.50": cov.get("0.50"), "cov@0.80": cov.get("0.80"),
        "cov@0.90": cov.get("0.90"), "cov@0.95": cov.get("0.95"),
        "spearman": sp,
        "top0.10": tk["0.10"], "top0.20": tk["0.20"], "top0.30": tk["0.30"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--megascale-emb", required=True, type=Path)
    p.add_argument("--megascale-bio", required=True, type=Path)
    p.add_argument("--t2837-emb",     required=True, type=Path)
    p.add_argument("--t2837-bio",     required=True, type=Path)
    p.add_argument("--out",           required=True, type=Path)

    # Pretrain hyperparameters
    p.add_argument("--pretrain-batch-size",   type=int,   default=256)
    p.add_argument("--pretrain-max-epochs",   type=int,   default=10)
    p.add_argument("--pretrain-lr",           type=float, default=5e-4)
    p.add_argument("--pretrain-patience",     type=int,   default=3)

    # Fine-tune hyperparameters
    p.add_argument("--finetune-batch-size",   type=int,   default=128)
    p.add_argument("--finetune-max-epochs",   type=int,   default=200)
    p.add_argument("--finetune-lr",           type=float, default=5e-5,
                   help="10× smaller than from-scratch (1e-3) to avoid catastrophic forgetting")
    p.add_argument("--finetune-patience",     type=int,   default=25)

    # Shared
    p.add_argument("--d-hidden",       type=int,   default=128)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--weight-decay",   type=float, default=1e-5)
    p.add_argument("--nu",             type=float, default=3.0)
    p.add_argument("--ranking-lambda", type=float, default=0.0)
    p.add_argument("--ranking-margin", type=float, default=0.05)

    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         type=str,   default="auto")
    p.add_argument("--log-level",      type=str,   default="INFO")
    args = p.parse_args()

    log = setup_logging(args.log_level)
    set_seed(args.seed)
    device = get_device(device_str=args.device)
    out_dir = ensure_dir(args.out)
    log.info("Device: %s", device)

    # ─────────────────────────────────────────────────────────────────────────
    # 1.  PRETRAIN on Megascale
    # ─────────────────────────────────────────────────────────────────────────
    log.info("\n══════════════ PRETRAIN on Megascale ══════════════")

    mega_splits, _ = load_cached_embeddings(args.megascale_emb)
    mega_bio = torch.load(args.megascale_bio, map_location="cpu", weights_only=False)
    log.info("Megascale split sizes: %s",
             {k: int(v[0].shape[0]) for k, v in mega_splits.items()})

    if "test" in mega_splits and "train" in mega_splits and "val" not in mega_splits:
        log.warning("Megascale has no 'val' split; using 'test' for early stopping.")
        mega_splits["val"] = mega_splits["test"]
        mega_bio["val"] = mega_bio["test"]

    Xm_tr, ym_tr = mega_splits["train"]
    Xm_va, ym_va = mega_splits["val"]
    em_tr = select_d5(mega_bio["train"]["feats"])
    em_va = select_d5(mega_bio["val"]["feats"])

    d_in    = int(Xm_tr.shape[-1])
    d_extra = int(em_tr.shape[-1])
    log.info("d_in=%d   d_extra=%d (D5)", d_in, d_extra)

    pretrain_head = FeatureAugmentedHead(
        d_in, d_extra=d_extra,
        d_hidden=args.d_hidden, dropout=args.dropout,
        init_sigma_bias=0.5,
    )
    pretrain_head, pretrain_log = train_head(
        pretrain_head,
        make_loader(Xm_tr, ym_tr, em_tr, args.pretrain_batch_size, shuffle=True),
        make_loader(Xm_va, ym_va, em_va, args.pretrain_batch_size, shuffle=False),
        device,
        max_epochs=args.pretrain_max_epochs, lr=args.pretrain_lr,
        weight_decay=args.weight_decay, patience=args.pretrain_patience,
        nu=args.nu,
        ranking_lambda=args.ranking_lambda, ranking_margin=args.ranking_margin,
        log=log, label="pretrain",
    )

    pretrain_dir = ensure_dir(out_dir / "pretrain")
    torch.save(pretrain_head.state_dict(), pretrain_dir / "head_state.pt")
    (pretrain_dir / "training_log.json").write_text(json.dumps(pretrain_log, indent=2))
    log.info("Saved Megascale-pretrained head to %s", pretrain_dir / "head_state.pt")

    # ─────────────────────────────────────────────────────────────────────────
    # 2.  FINE-TUNE on T2837 from the pretrained checkpoint
    # ─────────────────────────────────────────────────────────────────────────
    log.info("\n══════════════ FINE-TUNE on T2837 ══════════════")

    t_splits, _ = load_cached_embeddings(args.t2837_emb)
    t_bio = torch.load(args.t2837_bio, map_location="cpu", weights_only=False)
    log.info("T2837 split sizes: %s",
             {k: int(v[0].shape[0]) for k, v in t_splits.items()})

    Xt_tr, yt_tr = t_splits["train"]
    Xt_va, yt_va = t_splits["val"]
    Xt_te, yt_te = t_splits["test"]
    et_tr = select_d5(t_bio["train"]["feats"])
    et_va = select_d5(t_bio["val"]["feats"])
    et_te = select_d5(t_bio["test"]["feats"])

    if int(Xt_tr.shape[-1]) != d_in:
        raise SystemExit(
            f"Embedding dim mismatch: Megascale d_in={d_in}, T2837 d_in={Xt_tr.shape[-1]}.  "
            "Both caches must use the same backbone (ESM2-650M)."
        )

    finetune_head = FeatureAugmentedHead(
        d_in, d_extra=d_extra,
        d_hidden=args.d_hidden, dropout=args.dropout,
        init_sigma_bias=0.5,
    )
    finetune_head.load_state_dict(pretrain_head.state_dict())
    log.info("Loaded pretrained weights → ready to fine-tune")

    finetune_head, finetune_log = train_head(
        finetune_head,
        make_loader(Xt_tr, yt_tr, et_tr, args.finetune_batch_size, shuffle=True),
        make_loader(Xt_va, yt_va, et_va, args.finetune_batch_size, shuffle=False),
        device,
        max_epochs=args.finetune_max_epochs, lr=args.finetune_lr,
        weight_decay=args.weight_decay, patience=args.finetune_patience,
        nu=args.nu,
        ranking_lambda=args.ranking_lambda, ranking_margin=args.ranking_margin,
        log=log, label="finetune",
    )

    finetune_dir = ensure_dir(out_dir / "finetune")
    torch.save(finetune_head.state_dict(), finetune_dir / "head_state.pt")
    (finetune_dir / "training_log.json").write_text(json.dumps(finetune_log, indent=2))
    log.info("Saved fine-tuned head to %s", finetune_dir / "head_state.pt")

    # ─────────────────────────────────────────────────────────────────────────
    # 3.  EVALUATE on T2837 test
    # ─────────────────────────────────────────────────────────────────────────
    log.info("\n══════════════ EVALUATE on T2837 test ══════════════")

    test_loader = make_loader(Xt_te, yt_te, et_te, args.finetune_batch_size, shuffle=False)
    mu_te, sig_te, y_te = predict(finetune_head, test_loader, device)
    metrics = metrics_dict("megascale_pretrain_d5_finetune", mu_te, sig_te, y_te)

    np.savez(out_dir / "t2837_test_predictions.npz", mu=mu_te, sigma=sig_te, y=y_te)
    (out_dir / "t2837_test_metrics.json").write_text(json.dumps(metrics, indent=2))

    # ── Pretty print ─────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("Megascale-pretrained D5 head, fine-tuned on T2837")
    print("=" * 90)
    print(f"\nPretrain best val_loss = {pretrain_log['best_val']:.4f} @ epoch {pretrain_log['best_epoch']}")
    print(f"Finetune best val_loss = {finetune_log['best_val']:.4f} @ epoch {finetune_log['best_epoch']}")
    print(f"\nT2837 test metrics  (n = {metrics['n']}):")
    for k_, v in metrics.items():
        if k_ in ("name", "n"): continue
        if isinstance(v, float):
            print(f"  {k_:<14} = {v:+.4f}")
    print()
    print("Compare against from-scratch D5 baseline (campaign Day-2):")
    print("  Frozen D5 ensemble:   RMSE 1.50   NLL 1.85   ICE 0.05   Spearman 0.348")
    print("  Frozen D5 single seed: RMSE 1.50   NLL 1.92   ICE 0.05   Spearman 0.30 ± 0.07")
    print()
    print("Decision rule:")
    print("  Megascale-pretrain helps if Spearman > 0.40 OR NLL < 1.80 (strict).")
    print("  Marginal lift if Spearman ≈ 0.34-0.40 with much faster convergence (lower epochs).")
    print("=" * 90)


if __name__ == "__main__":
    main()
