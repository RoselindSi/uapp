"""External validation: train a Track-D ensemble on T2837, evaluate on S669.

Why
===
T2837 has only 170 test mutations across a small protein-level holdout, so
single-dataset numbers (RMSE 1.49, NLL 1.82, Spearman 0.331 for the D6
ensemble) sit well within ± 0.07 noise.  S669 (Pancotti et al. 2022) is the
field-standard cross-dataset benchmark with low train-overlap by design —
running our production model on it is the cleanest available test of
generalisation.

What this script does
=====================
1. Loads cached T2837 (train + val + test) embeddings and bio features.
2. Loads cached S669 (test-only) embeddings and bio features.
3. Trains K independent Track-D heads on T2837 train, early-stops on T2837 val.
4. Predicts on:
       • T2837 test  — sanity check, should reproduce the campaign numbers.
       • S669 test   — the headline external-validation result.
5. Reports the deliverable-table metrics for both datasets, plus a
   per-member mean ± std baseline and the ensemble row.

Outputs under ``--out``
----------------------
- ``per_member_predictions_t2837.npz``   arrays (K, N) for T2837 test
- ``per_member_predictions_s669.npz``    arrays (K, N) for S669
- ``ensemble_summary.json``              per-member rows + ensemble row,
                                          for both T2837 test and S669
- ``summary.csv``                        flat CSV form

Prerequisites
-------------
S669 must be cached in the same shape as T2837.  Run scripts 17 → 01 → 06
(→ 14 if --ablation D6) first; see the `Next steps` block printed by
``scripts/17_s669_to_t2837_format.py``.

Usage
-----
    # D6 ensemble (production model on T2837):
    python scripts/18_evaluate_on_s669.py \\
        --t2837-emb cache/t2837_embeddings_v2_650m.pt \\
        --t2837-bio cache/t2837_bio_features_650m_dssp.pt \\
        --s669-emb  cache/s669_embeddings_650m.pt \\
        --s669-bio  cache/s669_bio_features_650m_dssp.pt \\
        --out       outputs/s669_eval_d6 \\
        --ablation D6 --members 5 --device cuda

    # Or D5 (matches the headline Spearman number):
    python scripts/18_evaluate_on_s669.py \\
        --t2837-emb cache/t2837_embeddings_v2_650m.pt \\
        --t2837-bio cache/t2837_bio_features_650m_extended.pt \\
        --s669-emb  cache/s669_embeddings_650m.pt \\
        --s669-bio  cache/s669_bio_features_650m_extended.pt \\
        --out       outputs/s669_eval_d5 \\
        --ablation D5 --members 5 --device cuda
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
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


# Bio-feature slicing — must match scripts/07 and scripts/10 exactly.
RSA_IDX  = 0
BIO_IDX  = list(range(1, 7))     # chemistry
EXT_IDX  = list(range(7, 13))    # sequence-derived structural
DSSP_IDX = list(range(13, 21))   # DSSP + pLDDT


def select_features(feats: torch.Tensor, ablation: str) -> torch.Tensor | None:
    if ablation == "D0":
        return None
    if ablation == "D1":
        return feats[:, [RSA_IDX]]
    if ablation == "D2":
        return feats[:, BIO_IDX]
    if ablation == "D3":
        return feats[:, [RSA_IDX] + BIO_IDX]
    if ablation in ("D4", "D5"):
        if feats.shape[-1] < 13:
            raise ValueError(
                f"ablation {ablation!r} requires extended bio features (k≥13); "
                f"got file with k={feats.shape[-1]}.  Re-run "
                "scripts/06_build_bio_features.py with --include-extended."
            )
        if ablation == "D4":
            return feats[:, EXT_IDX]
        return feats[:, [RSA_IDX] + BIO_IDX + EXT_IDX]
    if ablation == "D6":
        if feats.shape[-1] < 21:
            raise ValueError(
                f"ablation 'D6' requires DSSP+pLDDT bio features (k=21); "
                f"got file with k={feats.shape[-1]}.  Re-run "
                "scripts/14_compute_structural_features.py."
            )
        return feats[:, [RSA_IDX] + BIO_IDX + EXT_IDX + DSSP_IDX]
    raise ValueError(f"unknown ablation: {ablation}")


def make_loader(X, y, extra, batch_size, shuffle):
    if extra is None:
        extra = torch.zeros(X.shape[0], 0)
    ds = TensorDataset(X.float(), y.float(), extra.float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_member(
    head, train_loader, val_loader, device,
    *, max_epochs, lr, weight_decay, patience, nu, log, label,
):
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
            best_val, best_state, ctr = v_loss, copy.deepcopy(head.state_dict()), 0
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


def metrics_dict(name, mu, sigma, y):
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


def aggregate_members(rows):
    keys = [
        "rmse", "mae", "nll", "ice",
        "cov@0.50", "cov@0.80", "cov@0.90", "cov@0.95",
        "spearman", "top0.10", "top0.20", "top0.30",
    ]
    agg = {f"{k}_mean": float(np.mean([r[k] for r in rows])) for k in keys}
    agg.update({f"{k}_std": float(np.std([r[k] for r in rows], ddof=1)) for k in keys})
    return agg


def s669_test_split(splits: dict, log) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Pick the S669 cache split that contains the data.

    `scripts/01 --test-fraction 1.0 --val-fraction 0` may yield slightly
    different layouts depending on rounding (e.g. 1 row in 'val').  We pick
    whichever split has the most rows and warn if the others are non-empty.
    """
    candidates = [(name, X, y) for name, (X, y) in splits.items() if X.shape[0] > 0]
    if not candidates:
        raise SystemExit("S669 cache contains no rows.")
    candidates.sort(key=lambda t: t[1].shape[0], reverse=True)
    chosen_name, chosen_X, chosen_y = candidates[0]
    if len(candidates) > 1:
        leftover = sum(t[1].shape[0] for t in candidates[1:])
        log.warning(
            "S669 cache has rows in multiple splits; using %r (%d rows), "
            "ignoring %d rows in other splits.  Re-cache with "
            "--val-fraction 0 --test-fraction 1.0 to put everything in 'test'.",
            chosen_name, chosen_X.shape[0], leftover,
        )
    return chosen_X, chosen_y, chosen_name


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--t2837-emb", required=True, type=Path)
    p.add_argument("--t2837-bio", required=True, type=Path)
    p.add_argument("--s669-emb",  required=True, type=Path)
    p.add_argument("--s669-bio",  required=True, type=Path)
    p.add_argument("--out",       type=Path, default=Path("outputs/s669_eval"))
    p.add_argument("--ablation",  choices=["D0", "D1", "D2", "D3", "D4", "D5", "D6"],
                   default="D6")
    p.add_argument("--members",   type=int, default=5)
    p.add_argument("--seed",      type=int, default=42,
                   help="Member k uses seed = base_seed + k.")
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

    # ── Load T2837 ──────────────────────────────────────────────────────────
    t_splits, _ = load_cached_embeddings(args.t2837_emb)
    t_bio = torch.load(args.t2837_bio, map_location="cpu", weights_only=False)
    log.info("T2837 bio features: %s", t_bio["meta"].get("feature_names"))
    X_tr, y_tr = t_splits["train"]
    X_va, y_va = t_splits["val"]
    X_te, y_te = t_splits["test"]
    extra_tr = select_features(t_bio["train"]["feats"], args.ablation)
    extra_va = select_features(t_bio["val"]["feats"],   args.ablation)
    extra_te = select_features(t_bio["test"]["feats"],  args.ablation)
    d_in    = int(X_tr.shape[-1])
    d_extra = 0 if extra_tr is None else int(extra_tr.shape[-1])
    log.info("T2837 split sizes: train=%d  val=%d  test=%d   d_in=%d  d_extra=%d",
             X_tr.shape[0], X_va.shape[0], X_te.shape[0], d_in, d_extra)

    # ── Load S669 ───────────────────────────────────────────────────────────
    s_splits, _ = load_cached_embeddings(args.s669_emb)
    s_bio = torch.load(args.s669_bio, map_location="cpu", weights_only=False)
    log.info("S669 bio features: %s", s_bio["meta"].get("feature_names"))
    X_s, y_s, s_split_name = s669_test_split(s_splits, log)
    extra_s = select_features(s_bio[s_split_name]["feats"], args.ablation)
    log.info("S669: n=%d  (from split %r)", X_s.shape[0], s_split_name)
    if int(X_s.shape[-1]) != d_in:
        raise SystemExit(
            f"Embedding-dim mismatch: T2837 d_in={d_in} vs S669 d_in={X_s.shape[-1]}.  "
            "Make sure both caches were built with the same --esm-model."
        )
    if d_extra and (extra_s is None or int(extra_s.shape[-1]) != d_extra):
        raise SystemExit(
            f"Bio-feature dim mismatch for ablation {args.ablation}: "
            f"T2837 d_extra={d_extra}, S669 d_extra="
            f"{0 if extra_s is None else int(extra_s.shape[-1])}.  "
            "Re-build S669 bio features with the matching script "
            "(scripts/06 with --include-extended for D5; scripts/14 for D6)."
        )

    # ── Build loaders ───────────────────────────────────────────────────────
    train_loader = make_loader(X_tr, y_tr, extra_tr, args.batch_size, shuffle=True)
    val_loader   = make_loader(X_va, y_va, extra_va, args.batch_size, shuffle=False)
    t_test_loader = make_loader(X_te, y_te, extra_te, args.batch_size, shuffle=False)
    s_test_loader = make_loader(X_s,  y_s,  extra_s,  args.batch_size, shuffle=False)

    # ── Train K members, predict on both test sets ──────────────────────────
    t_member_mu, t_member_sig = [], []
    s_member_mu, s_member_sig = [], []
    t_per_member_rows, s_per_member_rows = [], []

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

        mu_t, sig_t, _ = predict(head, t_test_loader, device)
        mu_s, sig_s, _ = predict(head, s_test_loader, device)

        t_member_mu.append(mu_t); t_member_sig.append(sig_t)
        s_member_mu.append(mu_s); s_member_sig.append(sig_s)

        rt = metrics_dict(f"{args.ablation}_m{k}_t2837", mu_t, sig_t, y_te.numpy())
        rs = metrics_dict(f"{args.ablation}_m{k}_s669",  mu_s, sig_s, y_s.numpy())
        rt["seed"] = rs["seed"] = seed
        t_per_member_rows.append(rt); s_per_member_rows.append(rs)

    def ensemble(mu_list, sig_list):
        mu_arr  = np.stack(mu_list,  axis=0)
        sig_arr = np.stack(sig_list, axis=0)
        mu_e   = mu_arr.mean(axis=0)
        var_a  = (sig_arr ** 2).mean(axis=0)
        var_e  = mu_arr.var(axis=0, ddof=0)
        sig_e  = np.sqrt(var_a + var_e)
        return mu_arr, sig_arr, mu_e, sig_e

    t_mu_arr, t_sig_arr, t_mu_e, t_sig_e = ensemble(t_member_mu, t_member_sig)
    s_mu_arr, s_sig_arr, s_mu_e, s_sig_e = ensemble(s_member_mu, s_member_sig)

    t_ens = metrics_dict(f"{args.ablation}_ensemble_t2837", t_mu_e, t_sig_e, y_te.numpy())
    s_ens = metrics_dict(f"{args.ablation}_ensemble_s669",  s_mu_e, s_sig_e, y_s.numpy())

    t_agg = aggregate_members(t_per_member_rows)
    s_agg = aggregate_members(s_per_member_rows)

    # ── Save ────────────────────────────────────────────────────────────────
    np.savez(out_dir / "per_member_predictions_t2837.npz",
             mu=t_mu_arr, sigma=t_sig_arr, y=y_te.numpy(),
             mu_ens=t_mu_e, sigma_ens=t_sig_e)
    np.savez(out_dir / "per_member_predictions_s669.npz",
             mu=s_mu_arr, sigma=s_sig_arr, y=y_s.numpy(),
             mu_ens=s_mu_e, sigma_ens=s_sig_e)

    summary = {
        "ablation": args.ablation,
        "members": args.members,
        "t2837_test": {
            "per_member": t_per_member_rows,
            "per_member_aggregate": t_agg,
            "ensemble": t_ens,
        },
        "s669": {
            "per_member": s_per_member_rows,
            "per_member_aggregate": s_agg,
            "ensemble": s_ens,
        },
    }
    (out_dir / "ensemble_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    flat_rows = (list(t_per_member_rows) + [t_ens]
                 + list(s_per_member_rows) + [s_ens])
    fieldnames = sorted({k for r in flat_rows for k in r.keys()})
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in flat_rows: w.writerow(r)

    # ── Pretty print ────────────────────────────────────────────────────────
    fmt = lambda r: (f"{r['rmse']:9.4f} {r['nll']:9.4f} {r['ice']:9.4f} "
                     f"{r['spearman']:10.4f} {r['top0.20']:9.4f}")
    header = f"\n{'model':<32} {'RMSE':>9} {'NLL':>9} {'ICE':>9} {'Spearman':>10} {'top20':>9}"

    print()
    print("=" * 92)
    print(f"External validation — {args.ablation}, K = {args.members} members")
    print("=" * 92)

    print(f"\n── T2837 test  (n = {t_ens['n_test']})  — sanity check vs campaign numbers")
    print(header)
    print("-" * 92)
    for r in t_per_member_rows:
        print(f"{r['name']:<32} {fmt(r)}")
    print("-" * 92)
    avg = {
        "rmse": t_agg["rmse_mean"], "nll": t_agg["nll_mean"], "ice": t_agg["ice_mean"],
        "spearman": t_agg["spearman_mean"], "top0.20": t_agg["top0.20_mean"],
    }
    print(f"{args.ablation+'_member_mean_t2837':<32} {fmt(avg)}")
    print(f"{'ENSEMBLE_t2837':<32} {fmt(t_ens)}")

    print(f"\n── S669  (n = {s_ens['n_test']})  — external validation")
    print(header)
    print("-" * 92)
    for r in s_per_member_rows:
        print(f"{r['name']:<32} {fmt(r)}")
    print("-" * 92)
    avg_s = {
        "rmse": s_agg["rmse_mean"], "nll": s_agg["nll_mean"], "ice": s_agg["ice_mean"],
        "spearman": s_agg["spearman_mean"], "top0.20": s_agg["top0.20_mean"],
    }
    print(f"{args.ablation+'_member_mean_s669':<32} {fmt(avg_s)}")
    print(f"{'ENSEMBLE_s669':<32} {fmt(s_ens)}")

    # ── Headline comparison ─────────────────────────────────────────────────
    print()
    print("Cross-dataset Δ (S669 − T2837 ensemble):")
    print(f"  Δ RMSE     = {s_ens['rmse']     - t_ens['rmse']:+.4f}  (positive: worse on S669)")
    print(f"  Δ NLL      = {s_ens['nll']      - t_ens['nll']:+.4f}")
    print(f"  Δ ICE      = {s_ens['ice']      - t_ens['ice']:+.4f}")
    print(f"  Δ Spearman = {s_ens['spearman'] - t_ens['spearman']:+.4f}  (positive: better on S669)")
    print()
    print("Reference T2837 ensemble numbers from the campaign (Day 3, REPORT.md):")
    print("  D6 ensemble:  RMSE 1.49   NLL 1.82   ICE 0.04   Spearman 0.331")
    print("  D5 ensemble:  RMSE 1.50   NLL 1.85   ICE 0.05   Spearman 0.348")
    print("=" * 92)
    log.info("Saved %s", out_dir / "summary.csv")
    log.info("Saved %s", out_dir / "ensemble_summary.json")


if __name__ == "__main__":
    main()
