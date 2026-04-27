"""Experiment D — does the σ branch need biophysical features?

This script implements Experiments 1 and 2 from the focused research plan:

    1. Track-D ablation on real T2837 data:
        D0   ESM only                         (baseline)
        D1   ESM + RSA                        (1-dim extra)
        D2   ESM + mutation-type features     (BLOSUM, Grantham, Δcharge, Δpol, Δhy, Δvol)
        D3   ESM + RSA + mutation-type        (full)

    2. Post-hoc variance scaling on the best raw model:
           σ' = a·σ
           σ' = a·σ + b
       fitted on validation, evaluated on test.

All four heads share the same μ branch architecture and likelihood
(``Student-t`` with ν = 3) — only the σ-branch input changes.  This isolates
the effect of biophysical features on uncertainty estimates.

Inputs
------
- ``--embeddings``  cached ESM2 embedding tensor (from scripts/01_*)
- ``--bio-feats``   aligned bio-feature tensor (from scripts/06_build_bio_features.py)

Outputs (under ``--out``)
-------------------------
- ``experiment_d_summary.csv``  one row per model with all metrics
- ``experiment_d_summary.json`` same content, machine-readable
- ``best_model.json``           summary of the model that maximises Spearman

Usage
-----
    python scripts/07_experiment_d_real.py \\
        --embeddings cache/t2837_embeddings_v2.pt \\
        --bio-feats  cache/t2837_bio_features.pt \\
        --out        outputs/experiment_d \\
        --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uapp.data import load_cached_embeddings
from uapp.evaluate import (
    compute_coverage_curve,
    compute_gaussian_nll,
    compute_ice,
    compute_mae,
    compute_rmse,
    compute_spearman_sigma_error,
    compute_top_k_risk_capture,
)
from uapp.heads import FeatureAugmentedHead
from uapp.losses import student_t_nll_loss, uncertainty_ranking_loss
from uapp.utils import ensure_dir, get_device, set_seed, setup_logging


# ─────────────────────────────────────────────────────────────────────────────
# Bio-feature loading + ablation slicing
# ─────────────────────────────────────────────────────────────────────────────
# Indices in the standardised feature vector produced by mutation_features.
RSA_IDX = 0
BIO_IDX = list(range(1, 7))  # blosum, grantham, dCharge, dPolarity, dHydro, dVolume


def load_bio_feats(path: Path) -> tuple[dict, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload.pop("meta", {})
    feats = {k: v["feats"] for k, v in payload.items()}
    return feats, meta


def select_features(feats: torch.Tensor, ablation: str) -> torch.Tensor | None:
    """Slice columns of the bio-feature tensor for one of D0/D1/D2/D3."""
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
# Loaders that yield (X, y, extra)
# ─────────────────────────────────────────────────────────────────────────────
def make_triple_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    extra: torch.Tensor | None,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    if extra is None:
        # Use a zero-dim placeholder so the collate signature stays uniform.
        extra = torch.zeros(X.shape[0], 0)
    ds = TensorDataset(X.float(), y.float(), extra.float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train_one(
    head: FeatureAugmentedHead,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    max_epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    nu: float,
    log,
    label: str,
    ranking_lambda: float = 0.0,
    ranking_margin: float = 0.05,
) -> FeatureAugmentedHead:
    """Single-stage training under Student-t NLL."""
    import copy

    head = head.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = None
    patience_ctr = 0
    best_epoch = -1

    for epoch in range(1, max_epochs + 1):
        # ── train ───────────────────────────────────────────────────────────
        head.train()
        for xb, yb, eb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            eb = eb.to(device) if eb.numel() > 0 else None
            opt.zero_grad()
            mu, sigma = head(xb, eb)
            loss = student_t_nll_loss(mu, sigma, yb, nu=nu)
            if ranking_lambda > 0.0:
                loss = loss + ranking_lambda * uncertainty_ranking_loss(
                    pred_mean=mu, pred_sigma=sigma, target=yb, margin=ranking_margin,
                )
            loss.backward()
            opt.step()

        # ── val ─────────────────────────────────────────────────────────────
        head.eval()
        total_loss, total_n = 0.0, 0
        with torch.no_grad():
            for xb, yb, eb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                eb = eb.to(device) if eb.numel() > 0 else None
                mu, sigma = head(xb, eb)
                loss = student_t_nll_loss(mu, sigma, yb, nu=nu)
                total_loss += float(loss.item()) * xb.size(0)
                total_n += xb.size(0)
        val_loss = total_loss / max(total_n, 1)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = copy.deepcopy(head.state_dict())
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch == 1 or epoch % 10 == 0:
            log.info("[%s] epoch %3d  val_loss %.4f  (best %.4f @ ep %d)",
                     label, epoch, val_loss, best_val, best_epoch)

        if patience_ctr >= patience:
            log.info("[%s] early stop at epoch %d", label, epoch)
            break

    if best_state is not None:
        head.load_state_dict(best_state)
    return head


# ─────────────────────────────────────────────────────────────────────────────
# Prediction + post-hoc scaling
# ─────────────────────────────────────────────────────────────────────────────
def predict(head: FeatureAugmentedHead, loader: DataLoader, device: torch.device):
    head.eval().to(device)
    mus, sigs, ys = [], [], []
    with torch.no_grad():
        for xb, yb, eb in loader:
            xb = xb.to(device)
            eb = eb.to(device) if eb.numel() > 0 else None
            mu, sigma = head(xb, eb)
            mus.append(mu.cpu().numpy())
            sigs.append(sigma.cpu().numpy())
            ys.append(yb.numpy())
    return np.concatenate(mus), np.concatenate(sigs), np.concatenate(ys)


def fit_scale_one(mu, sigma, y) -> float:
    def obj(x):
        a = float(x[0])
        s = np.maximum(a * sigma, 1e-6)
        var = s ** 2
        return float(np.mean(0.5 * ((y - mu) ** 2 / var + np.log(2 * math.pi * var))))
    res = minimize(obj, x0=[1.0], bounds=[(1e-4, 10.0)])
    return float(res.x[0]) if res.success else 1.0


def fit_scale_two(mu, sigma, y) -> tuple[float, float]:
    def obj(x):
        a, b = float(x[0]), float(x[1])
        s = np.maximum(a * sigma + b, 1e-6)
        var = s ** 2
        return float(np.mean(0.5 * ((y - mu) ** 2 / var + np.log(2 * math.pi * var))))
    res = minimize(obj, x0=[1.0, 0.0], bounds=[(1e-4, 10.0), (0.0, 5.0)])
    return (float(res.x[0]), float(res.x[1])) if res.success else (1.0, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics dict
# ─────────────────────────────────────────────────────────────────────────────
def compute_all_metrics(name: str, mu, sigma, y, *, ablation: str, scaling: str) -> dict:
    rmse = compute_rmse(mu, y)
    mae  = compute_mae(mu, y)
    nll  = compute_gaussian_nll(mu, sigma, y)
    ice, cov = compute_ice(mu, sigma, y)
    spear = compute_spearman_sigma_error(mu, sigma, y)
    topk  = compute_top_k_risk_capture(mu, sigma, y, k_fracs=[0.10, 0.20, 0.30])
    row = {
        "model": name, "ablation": ablation, "scaling": scaling,
        "n_test": int(len(y)),
        "rmse": rmse, "mae": mae, "nll": nll, "ice": ice,
        "cov@0.50": cov.get("0.50"), "cov@0.80": cov.get("0.80"),
        "cov@0.90": cov.get("0.90"), "cov@0.95": cov.get("0.95"),
        "spearman": spear,
        "top0.10": topk["0.10"], "top0.20": topk["0.20"], "top0.30": topk["0.30"],
    }
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--embeddings", required=True, type=Path)
    p.add_argument("--bio-feats",  required=True, type=Path)
    p.add_argument("--out",        required=True, type=Path)
    p.add_argument("--batch-size", type=int,   default=128)
    p.add_argument("--d-hidden",   type=int,   default=128)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--max-epochs", type=int,   default=200)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience",   type=int,   default=25)
    p.add_argument("--nu",         type=float, default=3.0,
                   help="Student-t degrees of freedom (fixed for clean attribution)")
    p.add_argument("--ranking-lambda", type=float, default=0.0,
                   help="Pairwise ranking loss weight (0 = NLL only).  "
                        "Recommended 0.05 to push σ-branch ranking on frozen backbone.")
    p.add_argument("--ranking-margin", type=float, default=0.05)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--device",     type=str,   default="auto",
                   help="auto | cpu | cuda | mps   (default: auto)")
    p.add_argument("--log-level",  type=str,   default="INFO")
    p.add_argument("--ablations",  nargs="+",  default=["D0", "D1", "D2", "D3"],
                   choices=["D0", "D1", "D2", "D3"])
    args = p.parse_args()

    log = setup_logging(args.log_level)
    set_seed(args.seed)
    device = get_device(device_str=args.device)
    out_dir = ensure_dir(args.out)
    log.info("Device: %s", device)

    # ── Load embeddings + bio features ───────────────────────────────────────
    splits, _ = load_cached_embeddings(args.embeddings)
    bio_feats, bio_meta = load_bio_feats(args.bio_feats)
    log.info("Bio-feature names: %s", bio_meta.get("feature_names"))

    for split in ["train", "val", "test"]:
        n_emb = int(splits[split][0].shape[0])
        n_bio = int(bio_feats[split].shape[0])
        if n_emb != n_bio:
            raise ValueError(
                f"alignment failure on split {split!r}: embeddings={n_emb} "
                f"bio={n_bio}.  Re-run scripts/06_build_bio_features.py."
            )

    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["val"]
    X_te, y_te = splits["test"]
    d_in = int(X_tr.shape[-1])

    # ── Run ablations ────────────────────────────────────────────────────────
    rows: list[dict] = []
    raw_preds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    for ablation in args.ablations:
        log.info("\n══════════════════════════════════════════════════════════════")
        log.info("Ablation %s", ablation)
        log.info("══════════════════════════════════════════════════════════════")

        extra_tr = select_features(bio_feats["train"], ablation)
        extra_va = select_features(bio_feats["val"],   ablation)
        extra_te = select_features(bio_feats["test"],  ablation)
        d_extra = 0 if extra_tr is None else int(extra_tr.shape[-1])
        log.info("d_extra = %d  (μ branch input dim = %d)", d_extra, d_in)

        train_loader = make_triple_loader(X_tr, y_tr, extra_tr, args.batch_size, shuffle=True)
        val_loader   = make_triple_loader(X_va, y_va, extra_va, args.batch_size, shuffle=False)
        test_loader  = make_triple_loader(X_te, y_te, extra_te, args.batch_size, shuffle=False)

        # Re-seed before each training run so the only difference between
        # ablations is the σ-branch input.
        set_seed(args.seed)
        head = FeatureAugmentedHead(
            d_in, d_extra=d_extra,
            d_hidden=args.d_hidden, dropout=args.dropout,
            init_sigma_bias=0.5,
        )
        head = train_one(
            head, train_loader, val_loader, device,
            max_epochs=args.max_epochs, lr=args.lr,
            weight_decay=args.weight_decay, patience=args.patience,
            nu=args.nu, log=log, label=ablation,
            ranking_lambda=args.ranking_lambda, ranking_margin=args.ranking_margin,
        )

        # Test-set predictions
        mu_te, sig_te, y_te_np = predict(head, test_loader, device)
        # Validation predictions for fitting the post-hoc scalers
        mu_va, sig_va, y_va_np = predict(head, val_loader, device)
        raw_preds[ablation] = (mu_te, sig_te, y_te_np, mu_va, sig_va, y_va_np)

        rows.append(compute_all_metrics(
            f"{ablation}_raw", mu_te, sig_te, y_te_np,
            ablation=ablation, scaling="raw",
        ))

    # ── Pick best raw model by Spearman, then apply post-hoc scaling ────────
    raw_rows = [r for r in rows if r["scaling"] == "raw"]
    best_raw = max(raw_rows, key=lambda r: r["spearman"])
    best_ablation = best_raw["ablation"]
    log.info("\nBest raw model (by Spearman): %s  (Spearman=%.4f)", best_ablation, best_raw["spearman"])

    mu_te, sig_te, y_te_np, mu_va, sig_va, y_va_np = raw_preds[best_ablation]

    a1 = fit_scale_one(mu_va, sig_va, y_va_np)
    sig_1 = np.maximum(a1 * sig_te, 1e-6)
    rows.append(compute_all_metrics(
        f"{best_ablation}_scale_a", mu_te, sig_1, y_te_np,
        ablation=best_ablation, scaling=f"a={a1:.4f}",
    ))

    a2, b2 = fit_scale_two(mu_va, sig_va, y_va_np)
    sig_2 = np.maximum(a2 * sig_te + b2, 1e-6)
    rows.append(compute_all_metrics(
        f"{best_ablation}_scale_a+b", mu_te, sig_2, y_te_np,
        ablation=best_ablation, scaling=f"a={a2:.4f},b={b2:.4f}",
    ))

    log.info("Post-hoc scaling: a=%.4f  |  a=%.4f, b=%.4f", a1, a2, b2)

    # ── Persist results ──────────────────────────────────────────────────────
    csv_path  = out_dir / "experiment_d_summary.csv"
    json_path = out_dir / "experiment_d_summary.json"

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)

    best_overall = max(rows, key=lambda r: r["spearman"])
    with (out_dir / "best_model.json").open("w") as f:
        json.dump(best_overall, f, indent=2)

    # ── Print mentor-style summary table ─────────────────────────────────────
    log.info("\n%s", "=" * 92)
    log.info("Experiment D — final table")
    log.info("%s", "=" * 92)
    hdr = f"{'model':<20} {'RMSE':>7} {'NLL':>7} {'ICE':>7} {'cov@90':>7} {'Spearman':>9} {'top20':>7}"
    log.info(hdr)
    log.info("-" * 92)
    for r in rows:
        log.info(
            f"{r['model']:<20} {r['rmse']:7.4f} {r['nll']:7.4f} {r['ice']:7.4f} "
            f"{r.get('cov@0.90', float('nan')):7.4f} {r['spearman']:9.4f} {r['top0.20']:7.4f}"
        )
    log.info("=" * 92)
    log.info("Best by Spearman: %s  (Spearman=%.4f)", best_overall["model"], best_overall["spearman"])
    log.info("Saved: %s, %s", csv_path, json_path)


if __name__ == "__main__":
    main()
