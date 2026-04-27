"""K-fold cross-validation for Track-D ablation (D0 / D1 / D2 / D3).

Why
===
The fixed T2837 split has n_test = 170, giving Spearman σ_seed ≈ 0.06.
That is roughly the same magnitude as the D0-vs-D1 effect we are trying
to measure — so single-test-set comparisons cannot distinguish them.

K-fold CV pools all 2584 samples and rotates them through K test folds
(split by ``pdb_code`` so no protein leaks between train and test of any
fold).  Two complementary aggregations:

1. **Per-fold-per-seed metrics**: K × S paired observations.  Paired
   t-test + Wilcoxon signed-rank give us power to detect a true effect
   on the order of Δ_Spearman ≈ 0.04 at α = 0.05.

2. **Pooled out-of-sample (OOS) predictions**: every sample appears
   exactly once in some fold's test set.  Concatenate predictions →
   one Spearman / NLL / ICE estimate over n = 2584.

Setup
=====
- Load embeddings + bio features + metadata CSV; pool train+val+test.
- Split unique pdb_codes into K equal-sized fold buckets (seed-controlled).
- Inside each fold, hold out a fraction of train proteins as the
  early-stopping val set.
- For each (fold, seed, ablation): train a FeatureAugmentedHead under
  Student-t (ν=3) NLL, predict on the fold's test samples, store
  (mu, sigma, y, fold_idx) tuples.
- Aggregate.

Outputs (under --out)
=====================
- ``per_fold_seed.csv``       one row per (ablation, fold, seed)
- ``pooled_oos.csv``          one row per (ablation, seed) using pooled OOS preds
- ``aggregate.csv``           mean ± std per ablation across all (fold, seed)
- ``significance.json``       paired tests, all challengers vs D0
- ``pooled_predictions.npz``  mu, sigma, y, fold_idx for offline analysis

Usage
-----
    python scripts/11_kfold_cv_track_d.py \\
        --embeddings    cache/t2837_embeddings_v2_650m.pt \\
        --bio-feats     cache/t2837_bio_features_650m.pt \\
        --out           outputs/cv_650m \\
        --folds 5 --seeds 0 1 2 --device mps
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
import pandas as pd
import torch
from scipy.stats import ttest_rel, wilcoxon
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uapp.data import load_cached_embeddings
from uapp.evaluate import (
    compute_gaussian_nll, compute_ice, compute_mae, compute_rmse,
    compute_spearman_sigma_error, compute_top_k_risk_capture,
)
from uapp.heads import FeatureAugmentedHead
from uapp.losses import student_t_nll_loss
from uapp.utils import ensure_dir, get_device, set_seed, setup_logging


# ─────────────────────────────────────────────────────────────────────────────
# Bio-feature slicing — must match scripts/07_experiment_d_real.py
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
# Data pooling — concatenate train/val/test from cache + bio + metadata
# ─────────────────────────────────────────────────────────────────────────────
def auto_discover_metadata(embeddings_path: Path) -> Path:
    candidate = embeddings_path.parent / "t2837_metadata.csv"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Auto-discovery failed; expected {candidate}.  "
            "Pass --metadata-csv explicitly."
        )
    return candidate


def load_pool(
    embeddings_path: Path,
    bio_path: Path,
    metadata_csv: Path,
    log,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Pool train + val + test into one big set.

    Returns (X, y, bio_feats, pdb_codes), each aligned row-wise.
    """
    splits, _ = load_cached_embeddings(embeddings_path)
    bio = torch.load(bio_path, map_location="cpu", weights_only=False)
    md = pd.read_csv(metadata_csv)

    # Find pdb_code column with case-insensitive lookup
    cols_lower = {c.lower(): c for c in md.columns}
    if "pdb_code" not in cols_lower:
        raise KeyError(f"metadata CSV missing pdb_code column (have {list(md.columns)})")
    pdb_col = cols_lower["pdb_code"]
    spl_col = cols_lower.get("split", "split")
    md["__split"] = md[spl_col].astype(str).str.strip().str.lower()
    grouped = {s: g for s, g in md.groupby("__split")}

    Xs, ys, bs, ps = [], [], [], []
    for split in ("train", "val", "test"):
        X, y = splits[split]
        b = bio[split]["feats"]
        pdbs = grouped[split][pdb_col].astype(str).values
        if not (X.shape[0] == b.shape[0] == len(pdbs)):
            raise RuntimeError(
                f"Alignment failure on {split!r}: X={X.shape[0]}, "
                f"bio={b.shape[0]}, metadata={len(pdbs)}"
            )
        Xs.append(X); ys.append(y); bs.append(b); ps.append(pdbs)

    X_all = torch.cat(Xs, dim=0)
    y_all = torch.cat(ys, dim=0)
    b_all = torch.cat(bs, dim=0)
    p_all = np.concatenate(ps, axis=0)
    log.info("Pooled  X=%s  y=%s  bio=%s  unique pdbs=%d",
             tuple(X_all.shape), tuple(y_all.shape), tuple(b_all.shape),
             len(np.unique(p_all)))
    return X_all, y_all, b_all, p_all


# ─────────────────────────────────────────────────────────────────────────────
# Protein-level fold + within-fold val split
# ─────────────────────────────────────────────────────────────────────────────
def kfold_protein_split(
    pdb_codes: np.ndarray, K: int, seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """K folds where each unique pdb_code lands in exactly one test fold."""
    rng = np.random.default_rng(seed)
    unique_pdbs = np.unique(pdb_codes)
    rng.shuffle(unique_pdbs)
    fold_pdbs = np.array_split(unique_pdbs, K)
    folds = []
    for k in range(K):
        test_pdbs = set(fold_pdbs[k].tolist())
        test_mask = np.array([p in test_pdbs for p in pdb_codes])
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        folds.append((train_idx, test_idx))
    return folds


def within_fold_val_split(
    train_idx: np.ndarray, pdb_codes: np.ndarray, val_frac: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve a protein-level val split out of the train pool for early stopping."""
    rng = np.random.default_rng(seed)
    train_pdbs = np.unique(pdb_codes[train_idx])
    rng.shuffle(train_pdbs)
    n_val = max(1, int(len(train_pdbs) * val_frac))
    val_pdbs = set(train_pdbs[:n_val].tolist())
    val_mask  = np.array([pdb_codes[i] in val_pdbs for i in train_idx])
    return train_idx[~val_mask], train_idx[val_mask]


# ─────────────────────────────────────────────────────────────────────────────
# Loaders + training + prediction
# ─────────────────────────────────────────────────────────────────────────────
def make_loader(X, y, extra, batch_size, shuffle):
    if extra is None:
        extra = torch.zeros(X.shape[0], 0)
    ds = TensorDataset(X.float(), y.float(), extra.float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one(head, train_loader, val_loader, device,
              max_epochs, lr, weight_decay, patience, nu) -> FeatureAugmentedHead:
    head = head.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    best_val, best_state, ctr = float("inf"), None, 0
    for _epoch in range(1, max_epochs + 1):
        head.train()
        for xb, yb, eb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            eb = eb.to(device) if eb.numel() > 0 else None
            opt.zero_grad()
            mu, sig = head(xb, eb)
            loss = student_t_nll_loss(mu, sig, yb, nu=nu)
            loss.backward(); opt.step()
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
            best_val, best_state, ctr = v_loss, copy.deepcopy(head.state_dict()), 0
        else:
            ctr += 1
        if ctr >= patience:
            break
    if best_state is not None:
        head.load_state_dict(best_state)
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
def metrics_dict(mu, sigma, y) -> dict:
    rmse = compute_rmse(mu, y); mae = compute_mae(mu, y)
    nll  = compute_gaussian_nll(mu, sigma, y)
    ice, cov = compute_ice(mu, sigma, y)
    sp = compute_spearman_sigma_error(mu, sigma, y)
    tk = compute_top_k_risk_capture(mu, sigma, y, k_fracs=[0.10, 0.20, 0.30])
    return {
        "n":        int(len(y)),
        "rmse":     rmse, "mae": mae, "nll": nll, "ice": ice,
        "cov@0.50": cov.get("0.50"), "cov@0.80": cov.get("0.80"),
        "cov@0.90": cov.get("0.90"), "cov@0.95": cov.get("0.95"),
        "spearman": sp,
        "top0.10":  tk["0.10"], "top0.20": tk["0.20"], "top0.30": tk["0.30"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Statistical tests
# ─────────────────────────────────────────────────────────────────────────────
def paired_test(per: pd.DataFrame, baseline: str, challenger: str, metric: str) -> dict:
    """Pair per (fold, seed); return paired t-test + Wilcoxon."""
    base = per[per.ablation == baseline].sort_values(["fold", "seed"])[metric].to_numpy()
    chal = per[per.ablation == challenger].sort_values(["fold", "seed"])[metric].to_numpy()
    if len(base) != len(chal) or len(base) < 2:
        return {"n": int(len(base)), "skipped": "not enough paired obs"}
    diff = chal - base
    t_stat, t_p = ttest_rel(chal, base)
    try:
        w_stat, w_p = wilcoxon(chal, base)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    return {
        "n":             int(len(base)),
        "mean_diff":     float(diff.mean()),
        "std_diff":      float(diff.std(ddof=1)),
        "t_stat":        float(t_stat),
        "t_p_two_sided": float(t_p),
        "wilcoxon_stat":      float(w_stat),
        "wilcoxon_p_two_sided": float(w_p),
        "wins_for_challenger": int((diff > 0).sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings",  required=True, type=Path)
    p.add_argument("--bio-feats",   required=True, type=Path)
    p.add_argument("--metadata-csv", type=Path, default=None,
                   help="Default: auto-discover next to --embeddings")
    p.add_argument("--out",         required=True, type=Path)
    p.add_argument("--ablations",   nargs="+", default=["D0", "D1", "D2", "D3"],
                   choices=["D0", "D1", "D2", "D3"])
    p.add_argument("--folds",       type=int, default=5)
    p.add_argument("--seeds",       type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--fold-seed",   type=int, default=42,
                   help="Seed controlling the fold assignment (kept fixed across runs)")
    p.add_argument("--val-frac",    type=float, default=0.10,
                   help="Fraction of train proteins held out for early stopping")
    p.add_argument("--batch-size",  type=int, default=128)
    p.add_argument("--d-hidden",    type=int, default=128)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--max-epochs",  type=int, default=200)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience",    type=int, default=25)
    p.add_argument("--nu",          type=float, default=3.0)
    p.add_argument("--device",      type=str, default="auto")
    p.add_argument("--log-level",   type=str, default="INFO")
    args = p.parse_args()

    log = setup_logging(args.log_level)
    device = get_device(device_str=args.device)
    out_dir = ensure_dir(args.out)
    log.info("Device: %s", device)

    metadata_csv = args.metadata_csv or auto_discover_metadata(args.embeddings)
    log.info("Metadata CSV: %s", metadata_csv)

    # ── Load + pool ──────────────────────────────────────────────────────────
    X, y, bio, pdb_codes = load_pool(args.embeddings, args.bio_feats, metadata_csv, log)
    d_in = int(X.shape[-1])
    n_total = X.shape[0]

    # ── K folds ──────────────────────────────────────────────────────────────
    folds = kfold_protein_split(pdb_codes, args.folds, args.fold_seed)
    log.info("Fold sizes (test): %s", [len(te) for _, te in folds])

    # ── Run all (ablation × fold × seed) ─────────────────────────────────────
    rows: list[dict] = []
    # For pooled OOS predictions: dict[(ablation, seed)] -> {"mu": np.array(n_total), "sigma": ..., "fold_idx": ...}
    pooled: dict[tuple[str, int], dict] = {}
    for abl in args.ablations:
        for s in args.seeds:
            pooled[(abl, s)] = {
                "mu":       np.full(n_total, np.nan, dtype=np.float64),
                "sigma":    np.full(n_total, np.nan, dtype=np.float64),
                "fold_idx": np.full(n_total, -1,    dtype=np.int32),
            }

    total_runs = len(args.ablations) * args.folds * len(args.seeds)
    run_i = 0
    for k, (train_idx, test_idx) in enumerate(folds):
        for s in args.seeds:
            # Within-fold val split (depends on seed so val rotates too)
            tr2_idx, va_idx = within_fold_val_split(
                train_idx, pdb_codes, args.val_frac, seed=args.fold_seed * 1000 + s,
            )

            for abl in args.ablations:
                run_i += 1
                set_seed(s + 100 * k)  # per-(fold, seed) reproducibility
                extra_full = select_features(bio, abl)
                d_extra = 0 if extra_full is None else int(extra_full.shape[-1])

                X_tr = X[tr2_idx]; y_tr = y[tr2_idx]
                X_va = X[va_idx];  y_va = y[va_idx]
                X_te = X[test_idx]; y_te = y[test_idx]
                e_tr = e_va = e_te = None
                if extra_full is not None:
                    e_tr = extra_full[tr2_idx]
                    e_va = extra_full[va_idx]
                    e_te = extra_full[test_idx]

                tl = make_loader(X_tr, y_tr, e_tr, args.batch_size, shuffle=True)
                vl = make_loader(X_va, y_va, e_va, args.batch_size, shuffle=False)
                el = make_loader(X_te, y_te, e_te, args.batch_size, shuffle=False)

                head = FeatureAugmentedHead(
                    d_in, d_extra=d_extra,
                    d_hidden=args.d_hidden, dropout=args.dropout,
                    init_sigma_bias=0.5,
                )
                head = train_one(
                    head, tl, vl, device,
                    max_epochs=args.max_epochs, lr=args.lr,
                    weight_decay=args.weight_decay, patience=args.patience,
                    nu=args.nu,
                )
                mu_te, sig_te, y_te_np = predict(head, el, device)

                # Per-fold per-seed metrics row
                m = metrics_dict(mu_te, sig_te, y_te_np)
                m.update({"ablation": abl, "fold": int(k), "seed": int(s),
                          "n_train": int(len(tr2_idx)), "n_val": int(len(va_idx)),
                          "n_test": int(len(test_idx))})
                rows.append(m)
                log.info("[%2d/%2d] fold %d seed %d  %s   "
                         "RMSE %.3f  NLL %.3f  ICE %.3f  Sp %+.3f",
                         run_i, total_runs, k, s, abl,
                         m["rmse"], m["nll"], m["ice"], m["spearman"])

                # Stash for pooled OOS predictions
                pooled[(abl, s)]["mu"][test_idx]       = mu_te
                pooled[(abl, s)]["sigma"][test_idx]    = sig_te
                pooled[(abl, s)]["fold_idx"][test_idx] = k

    per = pd.DataFrame(rows)
    per.to_csv(out_dir / "per_fold_seed.csv", index=False)
    log.info("Saved %s", out_dir / "per_fold_seed.csv")

    # ── Pooled OOS metrics: one Spearman / NLL / ICE per (ablation, seed) ────
    pooled_rows = []
    y_full = y.numpy().astype(np.float64)
    for (abl, s), p in pooled.items():
        if np.isnan(p["mu"]).any():
            log.warning("Pooled predictions for (%s, %d) have NaNs (incomplete CV)", abl, s)
            continue
        m = metrics_dict(p["mu"], p["sigma"], y_full)
        m.update({"ablation": abl, "seed": int(s), "n_pooled": int(n_total)})
        pooled_rows.append(m)
    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(out_dir / "pooled_oos.csv", index=False)
    log.info("Saved %s", out_dir / "pooled_oos.csv")

    # Save pooled prediction arrays for offline analysis
    np.savez(
        out_dir / "pooled_predictions.npz",
        y=y_full,
        **{
            f"{abl}_seed{s}_mu":       pooled[(abl, s)]["mu"]
            for (abl, s) in pooled
        },
        **{
            f"{abl}_seed{s}_sigma":    pooled[(abl, s)]["sigma"]
            for (abl, s) in pooled
        },
        **{
            f"{abl}_seed{s}_fold_idx": pooled[(abl, s)]["fold_idx"]
            for (abl, s) in pooled
        },
    )

    # ── Aggregate per ablation: mean ± std across all (fold, seed) ──────────
    METRIC_COLS = ["rmse", "mae", "nll", "ice",
                   "cov@0.50", "cov@0.80", "cov@0.90", "cov@0.95",
                   "spearman", "top0.10", "top0.20", "top0.30"]
    agg_rows = []
    for abl, sub in per.groupby("ablation"):
        row = {"ablation": abl, "n_obs": int(len(sub))}
        for m in METRIC_COLS:
            row[f"{m}_mean"] = float(sub[m].mean())
            row[f"{m}_std"]  = float(sub[m].std(ddof=1)) if len(sub) > 1 else 0.0
        agg_rows.append(row)
    pd.DataFrame(agg_rows).to_csv(out_dir / "aggregate.csv", index=False)
    log.info("Saved %s", out_dir / "aggregate.csv")

    # ── Paired stats: D1/D2/D3 vs D0 over per-(fold,seed) Spearman/ICE/NLL ──
    sig = {}
    for metric in ["spearman", "ice", "nll", "top0.20", "rmse"]:
        sig[metric] = {
            f"{c}_vs_D0": paired_test(per, "D0", c, metric)
            for c in ["D1", "D2", "D3"] if c in args.ablations
        }
    (out_dir / "significance.json").write_text(json.dumps(sig, indent=2))
    log.info("Saved %s", out_dir / "significance.json")

    # ── Pretty print ─────────────────────────────────────────────────────────
    print()
    print("=" * 92)
    print(f"K-fold CV Track-D — K = {args.folds} folds × {len(args.seeds)} seeds = "
          f"{args.folds * len(args.seeds)} paired obs per ablation")
    print("=" * 92)

    print("\nPer-(fold,seed) metrics  (mean ± std):\n")
    print(f"{'ablation':<6} {'RMSE':>16} {'NLL':>16} {'ICE':>16} {'Spearman':>16} {'top20':>16}")
    for r in sorted(agg_rows, key=lambda r: r["ablation"]):
        def fmt(m): return f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_std']:.4f}"
        print(f"{r['ablation']:<6} {fmt('rmse'):>16} {fmt('nll'):>16} "
              f"{fmt('ice'):>16} {fmt('spearman'):>16} {fmt('top0.20'):>16}")

    print("\nPooled OOS metrics  (each row uses n_pooled predictions):\n")
    if not pooled_df.empty:
        print(f"{'ablation':<6} {'seed':>4} {'RMSE':>9} {'NLL':>9} {'ICE':>9} {'Spearman':>10} {'top20':>9}")
        for _, r in pooled_df.sort_values(["ablation", "seed"]).iterrows():
            print(f"{r['ablation']:<6} {int(r['seed']):>4} {r['rmse']:9.4f} "
                  f"{r['nll']:9.4f} {r['ice']:9.4f} {r['spearman']:10.4f} {r['top0.20']:9.4f}")

    print("\nPaired tests on Spearman  (challenger − D0, paired by (fold,seed)):\n")
    print(f"{'comparison':<14} {'mean Δ':>10} {'p (t-test)':>12} {'p (Wilcoxon)':>14} {'wins':>10}")
    for k_, v in sig["spearman"].items():
        if "skipped" in v:
            print(f"{k_:<14} {v['skipped']}"); continue
        wins = f"{v['wins_for_challenger']}/{v['n']}"
        star = " *" if v["t_p_two_sided"] < 0.05 else ""
        print(f"{k_:<14} {v['mean_diff']:+10.4f} {v['t_p_two_sided']:12.4f} "
              f"{v['wilcoxon_p_two_sided']:14.4f} {wins:>10}{star}")

    print("\nDecision (Spearman, α = 0.05):")
    for c in ["D1", "D2", "D3"]:
        v = sig["spearman"].get(f"{c}_vs_D0")
        if v is None or "skipped" in v: continue
        verdict = ("BEATS D0" if v["t_p_two_sided"] < 0.05 and v["mean_diff"] > 0
                   else "no significant difference")
        print(f"  {c}: {verdict}  (Δ={v['mean_diff']:+.4f}, p={v['t_p_two_sided']:.3f})")
    print("=" * 92)


if __name__ == "__main__":
    main()
