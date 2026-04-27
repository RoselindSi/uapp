"""Post-hoc temperature scaling for D3 σ predictions.

Why
===
Frozen 650M D3 baseline has ICE = 0.069 — well above NEXT_STEPS' target
≤ 0.02.  The σ branch produces predictions with the right *ranking* but the
wrong *absolute magnitude*.  Fitting one tiny scalar T on the validation set
and rescaling σ → T·σ on test typically pushes ICE below 0.02 with **zero
change** to RMSE / Spearman / top-k risk capture (those metrics depend only
on the rank-order of σ, which a monotone rescale preserves).

What this script does
=====================
1. Train a fresh FeatureAugmentedHead (D3 ablation by default, ranking-loss
   optional) on the train split.  Same recipe as scripts/07.
2. Predict (μ, σ) on val and test.
3. Fit two scaling laws on val Gaussian-NLL:
        σ' = T · σ
        σ' = T · σ + b
4. Apply each to test, evaluate the deliverable-table metrics.
5. Print before / after table.  Save scaled test predictions + summary CSV.

For a saved set of predictions (e.g., from scripts/10), use --predictions-npz
instead of training fresh.  The npz must contain keys:
``val_mu, val_sigma, val_y, test_mu, test_sigma, test_y``.

Outputs (under --out)
=====================
- ``temperature_summary.json``   raw vs T-scaled vs (T,b)-scaled metrics
- ``temperature_summary.csv``    same content, flat
- ``test_predictions_scaled.npz`` test predictions for all three variants

Usage
-----
    # Self-contained: train D3 fresh, then scale.
    python scripts/13_temperature_scaling.py \\
        --embeddings cache/t2837_embeddings_v2_650m.pt \\
        --bio-feats  cache/t2837_bio_features_650m.pt \\
        --out        outputs/temperature_d3_650m \\
        --device     mps

    # Optional: also include ranking loss during training
    python scripts/13_temperature_scaling.py \\
        --embeddings cache/t2837_embeddings_v2_650m.pt \\
        --bio-feats  cache/t2837_bio_features_650m.pt \\
        --out        outputs/temperature_d3_rank_650m \\
        --ranking-lambda 0.05 \\
        --device     mps
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
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
# Bio-feature slicing — same convention as scripts/07, 10, 11
# ─────────────────────────────────────────────────────────────────────────────
RSA_IDX = 0
BIO_IDX = list(range(1, 7))


def select_features(feats: torch.Tensor, ablation: str) -> torch.Tensor | None:
    if ablation == "D0":
        return None
    if ablation == "D1":
        return feats[:, [RSA_IDX]]
    if ablation == "D2":
        return feats[:, BIO_IDX]
    if ablation == "D3":
        return feats[:, [RSA_IDX] + BIO_IDX]
    raise ValueError(f"unknown ablation: {ablation}")


# ─────────────────────────────────────────────────────────────────────────────
# Loaders / training / prediction (compact, no log spam)
# ─────────────────────────────────────────────────────────────────────────────
def make_loader(X, y, extra, batch_size, shuffle):
    if extra is None:
        extra = torch.zeros(X.shape[0], 0)
    ds = TensorDataset(X.float(), y.float(), extra.float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one(head, train_loader, val_loader, device,
              *, max_epochs, lr, weight_decay, patience, nu,
              ranking_lambda, ranking_margin, log) -> FeatureAugmentedHead:
    head = head.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, ctr = float("inf"), None, 0
    best_epoch = -1

    for epoch in range(1, max_epochs + 1):
        head.train()
        for xb, yb, eb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            eb = eb.to(device) if eb.numel() > 0 else None
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
                xb = xb.to(device); yb = yb.to(device)
                eb = eb.to(device) if eb.numel() > 0 else None
                mu, sig = head(xb, eb)
                v_loss += float(student_t_nll_loss(mu, sig, yb, nu=nu).item()) * xb.size(0)
                v_n += xb.size(0)
        v_loss /= max(v_n, 1)

        if v_loss < best_val - 1e-4:
            best_val, best_state, best_epoch, ctr = (
                v_loss, copy.deepcopy(head.state_dict()), epoch, 0
            )
        else:
            ctr += 1
        if ctr >= patience:
            break

    if best_state is not None:
        head.load_state_dict(best_state)
    log.info("Trained D3: best val_loss=%.4f @ ep %d", best_val, best_epoch)
    return head


def predict(head, loader, device):
    head.eval().to(device)
    mus, sigs, ys = [], [], []
    with torch.no_grad():
        for xb, yb, eb in loader:
            xb = xb.to(device)
            eb = eb.to(device) if eb.numel() > 0 else None
            mu, sig = head(xb, eb)
            mus.append(mu.cpu().numpy())
            sigs.append(sig.cpu().numpy())
            ys.append(yb.numpy())
    return np.concatenate(mus), np.concatenate(sigs), np.concatenate(ys)


# ─────────────────────────────────────────────────────────────────────────────
# Temperature scaling
# ─────────────────────────────────────────────────────────────────────────────
def fit_temperature_one(mu_va, sig_va, y_va) -> float:
    """Fit σ' = T·σ minimising val Gaussian NLL.  Returns scalar T."""
    def obj(x):
        T = float(x[0])
        s = np.maximum(T * sig_va, 1e-6)
        var = s ** 2
        return float(np.mean(0.5 * ((y_va - mu_va) ** 2 / var + np.log(2 * math.pi * var))))
    res = minimize(obj, x0=[1.0], bounds=[(1e-3, 50.0)])
    return float(res.x[0]) if res.success else 1.0


def fit_temperature_two(mu_va, sig_va, y_va) -> tuple[float, float]:
    """Fit σ' = T·σ + b minimising val Gaussian NLL."""
    def obj(x):
        T, b = float(x[0]), float(x[1])
        s = np.maximum(T * sig_va + b, 1e-6)
        var = s ** 2
        return float(np.mean(0.5 * ((y_va - mu_va) ** 2 / var + np.log(2 * math.pi * var))))
    res = minimize(obj, x0=[1.0, 0.0], bounds=[(1e-3, 50.0), (0.0, 5.0)])
    return (float(res.x[0]), float(res.x[1])) if res.success else (1.0, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def metrics_dict(name: str, mu, sigma, y, scaling: str) -> dict:
    rmse = compute_rmse(mu, y); mae = compute_mae(mu, y)
    nll  = compute_gaussian_nll(mu, sigma, y)
    ice, cov = compute_ice(mu, sigma, y)
    sp = compute_spearman_sigma_error(mu, sigma, y)
    tk = compute_top_k_risk_capture(mu, sigma, y, k_fracs=[0.10, 0.20, 0.30])
    return {
        "name": name, "scaling": scaling, "n": int(len(y)),
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
    p.add_argument("--embeddings", type=Path, required=False)
    p.add_argument("--bio-feats",  type=Path, required=False)
    p.add_argument("--predictions-npz", type=Path, default=None,
                   help="If supplied, skip training and read val/test predictions from this npz "
                        "(keys: val_mu, val_sigma, val_y, test_mu, test_sigma, test_y).")
    p.add_argument("--out",        required=True, type=Path)
    p.add_argument("--ablation",   choices=["D0", "D1", "D2", "D3"], default="D3")

    # Training hyperparameters (used when --predictions-npz is not given)
    p.add_argument("--batch-size",     type=int,   default=128)
    p.add_argument("--d-hidden",       type=int,   default=128)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--max-epochs",     type=int,   default=200)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--weight-decay",   type=float, default=1e-5)
    p.add_argument("--patience",       type=int,   default=25)
    p.add_argument("--nu",             type=float, default=3.0)
    p.add_argument("--ranking-lambda", type=float, default=0.0)
    p.add_argument("--ranking-margin", type=float, default=0.05)

    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     type=str, default="auto")
    p.add_argument("--log-level",  type=str, default="INFO")
    args = p.parse_args()

    log = setup_logging(args.log_level)
    set_seed(args.seed)
    device = get_device(device_str=args.device)
    out_dir = ensure_dir(args.out)
    log.info("Device: %s", device)

    # ── Get predictions: either from a saved npz, or by training fresh ──────
    if args.predictions_npz is not None:
        log.info("Loading predictions from %s", args.predictions_npz)
        npz = np.load(args.predictions_npz)
        for k in ("val_mu", "val_sigma", "val_y",
                  "test_mu", "test_sigma", "test_y"):
            if k not in npz.files:
                raise KeyError(f"--predictions-npz must contain {k!r}; "
                               f"got {list(npz.files)}")
        mu_va, sig_va, y_va = npz["val_mu"], npz["val_sigma"], npz["val_y"]
        mu_te, sig_te, y_te = npz["test_mu"], npz["test_sigma"], npz["test_y"]

    else:
        if args.embeddings is None or args.bio_feats is None:
            raise SystemExit(
                "Provide either --predictions-npz, or both --embeddings and --bio-feats"
            )
        log.info("Training fresh %s head on train …", args.ablation)
        splits, _ = load_cached_embeddings(args.embeddings)
        bio = torch.load(args.bio_feats, map_location="cpu", weights_only=False)
        X_tr, y_tr = splits["train"]; X_va, y_va_t = splits["val"]; X_te, y_te_t = splits["test"]
        d_in = int(X_tr.shape[-1])

        e_tr = select_features(bio["train"]["feats"], args.ablation)
        e_va = select_features(bio["val"]["feats"],   args.ablation)
        e_te = select_features(bio["test"]["feats"],  args.ablation)
        d_extra = 0 if e_tr is None else int(e_tr.shape[-1])

        head = FeatureAugmentedHead(
            d_in, d_extra=d_extra,
            d_hidden=args.d_hidden, dropout=args.dropout,
            init_sigma_bias=0.5,
        )
        head = train_one(
            head,
            make_loader(X_tr, y_tr, e_tr, args.batch_size, shuffle=True),
            make_loader(X_va, y_va_t, e_va, args.batch_size, shuffle=False),
            device,
            max_epochs=args.max_epochs, lr=args.lr,
            weight_decay=args.weight_decay, patience=args.patience,
            nu=args.nu, ranking_lambda=args.ranking_lambda,
            ranking_margin=args.ranking_margin, log=log,
        )

        mu_va, sig_va, y_va = predict(
            head, make_loader(X_va, y_va_t, e_va, args.batch_size, shuffle=False), device,
        )
        mu_te, sig_te, y_te = predict(
            head, make_loader(X_te, y_te_t, e_te, args.batch_size, shuffle=False), device,
        )

    # ── Fit scalings on val ──────────────────────────────────────────────────
    T = fit_temperature_one(mu_va, sig_va, y_va)
    Tb = fit_temperature_two(mu_va, sig_va, y_va)
    log.info("Fitted on val:  T = %.4f       T,b = (%.4f, %.4f)", T, Tb[0], Tb[1])

    # ── Apply to test, evaluate ──────────────────────────────────────────────
    sig_te_T   = np.maximum(T * sig_te, 1e-6)
    sig_te_Tb  = np.maximum(Tb[0] * sig_te + Tb[1], 1e-6)

    rows = [
        metrics_dict("raw",         mu_te, sig_te,    y_te, scaling="none"),
        metrics_dict("T·σ",         mu_te, sig_te_T,  y_te, scaling=f"T={T:.4f}"),
        metrics_dict("T·σ + b",     mu_te, sig_te_Tb, y_te, scaling=f"T={Tb[0]:.4f}, b={Tb[1]:.4f}"),
    ]

    # ── Persist ──────────────────────────────────────────────────────────────
    np.savez(out_dir / "test_predictions_scaled.npz",
             mu=mu_te, sigma_raw=sig_te, sigma_T=sig_te_T, sigma_Tb=sig_te_Tb, y=y_te,
             T=T, Tb_T=Tb[0], Tb_b=Tb[1])
    (out_dir / "temperature_summary.json").write_text(json.dumps(rows, indent=2))
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with (out_dir / "temperature_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in rows: w.writerow(r)

    # ── Pretty print ─────────────────────────────────────────────────────────
    print()
    print("=" * 92)
    print("Post-hoc temperature scaling")
    print("=" * 92)
    print(f"\n{'variant':<14} {'RMSE':>9} {'NLL':>9} {'ICE':>9} {'cov@90':>9} {'Spearman':>10} {'top20':>9}")
    print("-" * 92)
    for r in rows:
        print(f"{r['name']:<14} {r['rmse']:9.4f} {r['nll']:9.4f} {r['ice']:9.4f} "
              f"{r['cov@0.90']:9.4f} {r['spearman']:10.4f} {r['top0.20']:9.4f}")
    print()
    print("Notes:")
    print("  - RMSE / Spearman / top-k are unchanged by σ rescaling (rank-order preserved)")
    print(f"  - ICE target ≤ 0.02:  raw {rows[0]['ice']:.3f}  →  T·σ {rows[1]['ice']:.3f}  "
          f"→  T·σ+b {rows[2]['ice']:.3f}")
    print(f"  - NLL  target < 1.85: raw {rows[0]['nll']:.3f}  →  T·σ {rows[1]['nll']:.3f}  "
          f"→  T·σ+b {rows[2]['nll']:.3f}")
    print("=" * 92)

    log.info("Saved %s", out_dir / "temperature_summary.csv")


if __name__ == "__main__":
    main()
