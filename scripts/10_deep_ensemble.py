"""Deep ensemble for the chosen Track-D variant (default: D1, RSA-only).

Why D1
======
Multi-seed Track-D on ESM2-650M (script 09 output) showed::

    D0:  Spearman 0.296 ± 0.042
    D1:  Spearman 0.347 ± 0.068    ★ best (only RSA in the σ branch)
    D2:  Spearman 0.242 ± 0.057    ← *worse* than D0
    D3:  Spearman 0.333 ± 0.062

ESM2-650M's per-residue embedding already encodes substitution-type
chemistry (BLOSUM/Grantham/charge/polarity/hydrophobicity/volume).
Re-injecting hand-crafted versions into the σ branch becomes redundant
and even hurts (D2 < D0).  Only RSA — structural context that is *not*
in ESM2's training objective — keeps adding signal.

What this script does
=====================
1. Train K independent D1 (or any chosen ablation) models, one per seed.
2. Save per-member μ and σ on the test set.
3. Build the ensemble predictive distribution::

       μ_ens   = (1/K) Σ μ_k
       σ²_ens  = (1/K) Σ σ_k² + Var_k(μ_k)
                 \─── mean aleatoric ──/   \─ epistemic from disagreement ─/

4. Report the deliverable-table metrics on the ensemble plus a per-member
   mean ± std baseline (apples-to-apples vs script 09 output).

Outputs under ``--out``
----------------------
- ``per_member_predictions.npz``  arrays (K, N_test) for mu and sigma
- ``ensemble_summary.json``       per-member rows + ensemble row + diff
- ``summary.csv``                 same content in flat CSV form

Recommended runs (in order)
---------------------------
::

    # Step 1: D1 ensemble on 650M backbone — the headline deliverable
    python scripts/10_deep_ensemble.py \\
        --embeddings cache/t2837_embeddings_v2_650m.pt \\
        --bio-feats  cache/t2837_bio_features_650m.pt \\
        --out        outputs/ensemble_d1_650m \\
        --ablation   D1 --members 5 --device mps

    # Step 2: D0 ensemble on the same backbone — apples-to-apples baseline
    python scripts/10_deep_ensemble.py \\
        --embeddings cache/t2837_embeddings_v2_650m.pt \\
        --bio-feats  cache/t2837_bio_features_650m.pt \\
        --out        outputs/ensemble_d0_650m \\
        --ablation   D0 --members 5 --device mps
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uapp.data import load_cached_embeddings
from uapp.evaluate import (
    compute_gaussian_nll,
    compute_ice,
    compute_mae,
    compute_rmse,
    compute_spearman_sigma_error,
    compute_top_k_risk_capture,
)
from uapp.heads import FeatureAugmentedHead
from uapp.losses import student_t_nll_loss
from uapp.utils import ensure_dir, get_device, set_seed, setup_logging


# ─────────────────────────────────────────────────────────────────────────────
# Bio-feature slicing — must match scripts/07_experiment_d_real.py exactly
# ─────────────────────────────────────────────────────────────────────────────
RSA_IDX = 0
BIO_IDX = list(range(1, 7))  # blosum, grantham, dCharge, dPolarity, dHydro, dVolume


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
# Dataloader yielding (X, y, extra)
# ─────────────────────────────────────────────────────────────────────────────
def make_loader(X, y, extra, batch_size, shuffle):
    if extra is None:
        extra = torch.zeros(X.shape[0], 0)
    ds = TensorDataset(X.float(), y.float(), extra.float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ─────────────────────────────────────────────────────────────────────────────
# Train one ensemble member
# ─────────────────────────────────────────────────────────────────────────────
def train_member(
    head: FeatureAugmentedHead,
    train_loader, val_loader, device,
    *, max_epochs, lr, weight_decay, patience, nu, log, label,
) -> FeatureAugmentedHead:
    import copy
    head = head.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, ctr = float("inf"), None, 0

    for epoch in range(1, max_epochs + 1):
        head.train()
        for xb, yb, eb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            eb = eb.to(device) if eb.numel() > 0 else None
            opt.zero_grad()
            mu, sig = head(xb, eb)
            loss = student_t_nll_loss(mu, sig, yb, nu=nu)
            loss.backward()
            opt.step()

        head.eval()
        v_loss, v_n = 0.0, 0
        with torch.no_grad():
            for xb, yb, eb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                eb = eb.to(device) if eb.numel() > 0 else None
                mu, sig = head(xb, eb)
                v_loss += float(student_t_nll_loss(mu, sig, yb, nu=nu).item()) * xb.size(0)
                v_n += xb.size(0)
        v_loss /= max(v_n, 1)

        if v_loss < best_val - 1e-4:
            best_val = v_loss; best_state = copy.deepcopy(head.state_dict()); ctr = 0
        else:
            ctr += 1
        if ctr >= patience:
            break
    if best_state is not None:
        head.load_state_dict(best_state)
    log.info("[%s] best val_loss=%.4f", label, best_val)
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
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def metrics_dict(name: str, mu, sigma, y) -> dict:
    rmse = compute_rmse(mu, y)
    mae  = compute_mae(mu, y)
    nll  = compute_gaussian_nll(mu, sigma, y)
    ice, cov = compute_ice(mu, sigma, y)
    sp = compute_spearman_sigma_error(mu, sigma, y)
    tk = compute_top_k_risk_capture(mu, sigma, y, k_fracs=[0.10, 0.20, 0.30])
    return {
        "name": name, "n_test": int(len(y)),
        "rmse": rmse, "mae": mae, "nll": nll, "ice": ice,
        "cov@0.50": cov.get("0.50"), "cov@0.80": cov.get("0.80"),
        "cov@0.90": cov.get("0.90"), "cov@0.95": cov.get("0.95"),
        "spearman": sp,
        "top0.10": tk["0.10"], "top0.20": tk["0.20"], "top0.30": tk["0.30"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings", required=True, type=Path)
    p.add_argument("--bio-feats",  required=True, type=Path)
    p.add_argument("--out",        required=True, type=Path)
    p.add_argument("--ablation",   choices=["D0", "D1", "D2", "D3"], default="D1")
    p.add_argument("--members",    type=int, default=5)
    p.add_argument("--seed",       type=int, default=42, help="seed for member 0; member k uses seed+k")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--d-hidden",   type=int, default=128)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience",   type=int, default=25)
    p.add_argument("--nu",         type=float, default=3.0)
    p.add_argument("--device",     type=str, default="auto")
    p.add_argument("--log-level",  type=str, default="INFO")
    args = p.parse_args()

    log = setup_logging(args.log_level)
    device = get_device(device_str=args.device)
    out_dir = ensure_dir(args.out)
    log.info("Device: %s", device)
    log.info("Ablation: %s   Members: %d", args.ablation, args.members)

    # ── Load ────────────────────────────────────────────────────────────────
    splits, _ = load_cached_embeddings(args.embeddings)
    bio = torch.load(args.bio_feats, map_location="cpu", weights_only=False)
    log.info("Bio-feature names: %s", bio["meta"].get("feature_names"))

    X_tr, y_tr = splits["train"]; X_va, y_va = splits["val"]; X_te, y_te = splits["test"]
    d_in = int(X_tr.shape[-1])

    extra_tr = select_features(bio["train"]["feats"], args.ablation)
    extra_va = select_features(bio["val"]["feats"],   args.ablation)
    extra_te = select_features(bio["test"]["feats"],  args.ablation)
    d_extra = 0 if extra_tr is None else int(extra_tr.shape[-1])
    log.info("d_in=%d   d_extra=%d", d_in, d_extra)

    train_loader = make_loader(X_tr, y_tr, extra_tr, args.batch_size, shuffle=True)
    val_loader   = make_loader(X_va, y_va, extra_va, args.batch_size, shuffle=False)
    test_loader  = make_loader(X_te, y_te, extra_te, args.batch_size, shuffle=False)

    # ── Train K members ─────────────────────────────────────────────────────
    member_mu, member_sig = [], []
    per_member_rows = []

    for k in range(args.members):
        seed = args.seed + k
        log.info("\n──── Member %d/%d (seed=%d) ────", k + 1, args.members, seed)
        set_seed(seed)
        head = FeatureAugmentedHead(
            d_in, d_extra=d_extra,
            d_hidden=args.d_hidden, dropout=args.dropout,
            init_sigma_bias=0.5,
        )
        head = train_member(
            head, train_loader, val_loader, device,
            max_epochs=args.max_epochs, lr=args.lr,
            weight_decay=args.weight_decay, patience=args.patience,
            nu=args.nu, log=log, label=f"{args.ablation}_m{k}",
        )
        mu, sig, y = predict(head, test_loader, device)
        member_mu.append(mu); member_sig.append(sig)
        row = metrics_dict(f"{args.ablation}_m{k}", mu, sig, y)
        row["seed"] = seed
        per_member_rows.append(row)

    # Stack into (K, N) arrays
    mu_arr  = np.stack(member_mu,  axis=0)
    sig_arr = np.stack(member_sig, axis=0)
    y_np    = y  # last predict() call returned the same y

    # ── Build ensemble distribution ─────────────────────────────────────────
    mu_ens  = mu_arr.mean(axis=0)
    var_aleatoric = (sig_arr ** 2).mean(axis=0)
    var_epistemic = mu_arr.var(axis=0, ddof=0)
    sig_ens = np.sqrt(var_aleatoric + var_epistemic)

    ens_row = metrics_dict(f"{args.ablation}_ensemble", mu_ens, sig_ens, y_np)

    # Per-member aggregate for comparison
    metric_keys = [
        "rmse", "mae", "nll", "ice",
        "cov@0.50", "cov@0.80", "cov@0.90", "cov@0.95",
        "spearman", "top0.10", "top0.20", "top0.30",
    ]
    member_agg = {f"{m}_mean": float(np.mean([r[m] for r in per_member_rows]))
                  for m in metric_keys}
    member_agg.update({f"{m}_std": float(np.std([r[m] for r in per_member_rows], ddof=1))
                       for m in metric_keys})

    # ── Save ────────────────────────────────────────────────────────────────
    np.savez(out_dir / "per_member_predictions.npz",
             mu=mu_arr, sigma=sig_arr, y=y_np,
             mu_ens=mu_ens, sigma_ens=sig_ens)

    summary = {
        "ablation": args.ablation,
        "members": args.members,
        "per_member": per_member_rows,
        "per_member_aggregate": member_agg,
        "ensemble": ens_row,
    }
    (out_dir / "ensemble_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    flat_rows = list(per_member_rows) + [ens_row]
    fieldnames = sorted({k for r in flat_rows for k in r.keys()})
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in flat_rows: w.writerow(r)

    # ── Pretty print comparison ─────────────────────────────────────────────
    print()
    print("=" * 90)
    print(f"Deep ensemble — {args.ablation}, K = {args.members} members")
    print("=" * 90)
    print(f"\n{'model':<28} {'RMSE':>9} {'NLL':>9} {'ICE':>9} {'Spearman':>10} {'top20':>9}")
    print("-" * 90)

    fmt = lambda r: (f"{r['rmse']:9.4f} {r['nll']:9.4f} {r['ice']:9.4f} "
                     f"{r['spearman']:10.4f} {r['top0.20']:9.4f}")

    for r in per_member_rows:
        print(f"{r['name']:<28} {fmt(r)}")

    print("-" * 90)
    avg_name = f"{args.ablation}_member_mean"
    avg_row = {
        "rmse":     member_agg["rmse_mean"],
        "nll":      member_agg["nll_mean"],
        "ice":      member_agg["ice_mean"],
        "spearman": member_agg["spearman_mean"],
        "top0.20":  member_agg["top0.20_mean"],
    }
    print(f"{avg_name:<28} {fmt(avg_row)}")
    print(f"{'ENSEMBLE':<28} {fmt(ens_row)}    ← combined predictive distribution")
    print()

    # ── Decision summary ────────────────────────────────────────────────────
    sp_gain = ens_row["spearman"] - member_agg["spearman_mean"]
    nll_gain = member_agg["nll_mean"] - ens_row["nll"]   # positive = ensemble better
    ice_gain = member_agg["ice_mean"] - ens_row["ice"]   # positive = ensemble better

    print("Ensemble vs. mean-of-members:")
    print(f"  Δ Spearman = {sp_gain:+.4f}   (positive: ensemble ranks better)")
    print(f"  Δ NLL      = {nll_gain:+.4f}  (positive: ensemble more probable)")
    print(f"  Δ ICE      = {ice_gain:+.4f}  (positive: ensemble better calibrated)")
    print("=" * 90)
    log.info("Saved %s", out_dir / "summary.csv")
    log.info("Saved %s", out_dir / "ensemble_summary.json")
    log.info("Saved %s", out_dir / "per_member_predictions.npz")


if __name__ == "__main__":
    main()
