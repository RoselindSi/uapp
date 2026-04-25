"""Run the implemented research tracks from NEXT_STEPS.

Tracks implemented:
1) Student-t nu sweep + post-hoc variance scaling
2) Fixed-variance (mentor direction)
3) Ranking-aware uncertainty auxiliary loss

Usage
-----
python scripts/05_run_research_tracks.py \
  --embeddings cache/t2837_embeddings_v2.pt \
  --out outputs/research_tracks \
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uapp.data import load_cached_embeddings, make_loader
from uapp.evaluate import EvalResult, evaluate_head
from uapp.heads import build_head
from uapp.train import TrainConfig, train_head
from uapp.utils import ensure_dir, get_device, set_seed, setup_logging


def _predict_probabilistic(head, loader, device):
    head.eval()
    mus, sigs, ys = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            mu, sigma = head(xb)
            mus.append(mu.cpu().numpy())
            sigs.append(sigma.cpu().numpy())
            ys.append(yb.numpy())
    return np.concatenate(mus), np.concatenate(sigs), np.concatenate(ys)


def _gaussian_nll(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    var = np.maximum(sigma**2, 1e-12)
    return float(np.mean(0.5 * ((y - mu) ** 2 / var + np.log(2.0 * math.pi * var))))


def fit_variance_scaling(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit sigma' = a*sigma + b on validation by minimizing Gaussian NLL."""

    def obj(x):
        a, b = float(x[0]), float(x[1])
        scaled = np.maximum(a * sigma + b, 1e-6)
        return _gaussian_nll(mu, scaled, y)

    res = minimize(obj, x0=np.array([1.0, 0.0]), bounds=[(1e-4, 10.0), (0.0, 5.0)])
    if not res.success:
        return 1.0, 0.0
    return float(res.x[0]), float(res.x[1])


def evaluate_with_scaled_sigma(head, loader, device, name: str, a: float, b: float) -> EvalResult:
    mu, sigma, y = _predict_probabilistic(head, loader, device)
    scaled = np.maximum(a * sigma + b, 1e-6)

    from uapp.evaluate import compute_rmse, compute_mae, compute_gaussian_nll, compute_ice

    rmse = compute_rmse(mu, y)
    mae = compute_mae(mu, y)
    nll = compute_gaussian_nll(mu, scaled, y)
    ice, cov = compute_ice(mu, scaled, y)
    return EvalResult(head_name=name, n_test=int(len(y)), rmse=rmse, mae=mae, nll=nll, ice=ice, coverage=cov)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-epochs", type=int, default=160)
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

    splits, _ = load_cached_embeddings(args.embeddings)
    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["val"]
    X_te, y_te = splits["test"]
    d_in = int(X_tr.shape[-1])

    train_loader = make_loader(X_tr, y_tr, args.batch_size, shuffle=True)
    val_loader = make_loader(X_va, y_va, args.batch_size, shuffle=False)
    test_loader = make_loader(X_te, y_te, args.batch_size, shuffle=False)

    base_cfg = dict(
        max_epochs=args.max_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    rows: list[dict] = []

    # Track A: Student-t sweep + post-hoc scaling
    for nu in (3.0, 5.0, 8.0):
        head = build_head("two_head_nll", d_in, d_hidden=args.d_hidden, dropout=args.dropout, init_sigma_bias=0.5)
        cfg = TrainConfig(**base_cfg, loss_type="student_t", student_t_nu=nu)
        head, _ = train_head(head, train_loader, val_loader, cfg, device)

        raw_res = evaluate_head(head, test_loader, device, head_name=f"student_t_nu{int(nu)}")
        rows.append({"track": "A_student_t", "variant": f"nu={nu}", "scaled": "no", **asdict(raw_res)})

        mu_val, sigma_val, y_val = _predict_probabilistic(head, val_loader, device)
        a, b = fit_variance_scaling(mu_val, sigma_val, y_val)
        scaled_res = evaluate_with_scaled_sigma(
            head, test_loader, device,
            name=f"student_t_nu{int(nu)}_scaled",
            a=a,
            b=b,
        )
        rows.append({
            "track": "A_student_t",
            "variant": f"nu={nu}",
            "scaled": "yes",
            "a": a,
            "b": b,
            **asdict(scaled_res),
        })

    # Track B: Mentor direction fixed sigma
    for fixed_sigma in (1.0, 1.5, 2.0):
        head = build_head(
            "fixed_sigma_nll",
            d_in,
            d_hidden=args.d_hidden,
            dropout=args.dropout,
            fixed_sigma=fixed_sigma,
        )
        cfg = TrainConfig(**base_cfg, loss_type="gaussian")
        head, _ = train_head(head, train_loader, val_loader, cfg, device)
        res = evaluate_head(head, test_loader, device, head_name=f"fixed_sigma_{fixed_sigma:.1f}")
        rows.append({"track": "B_fixed_sigma", "variant": f"sigma={fixed_sigma}", "scaled": "no", **asdict(res)})

    # Track C: Ranking-aware
    for lam in (0.01, 0.05):
        head = build_head("two_head_nll", d_in, d_hidden=args.d_hidden, dropout=args.dropout, init_sigma_bias=0.5)
        cfg = TrainConfig(
            **base_cfg,
            loss_type="student_t",
            student_t_nu=3.0,
            ranking_lambda=lam,
            ranking_margin=0.05,
        )
        head, _ = train_head(head, train_loader, val_loader, cfg, device)
        res = evaluate_head(head, test_loader, device, head_name=f"rank_lambda_{lam}")
        rows.append({"track": "C_ranking", "variant": f"lambda={lam}", "scaled": "no", **asdict(res)})

    # save outputs
    out_csv = out_dir / "research_track_summary.csv"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            # Flatten coverage dict for csv compatibility
            rr = dict(r)
            cov = rr.get("coverage")
            if isinstance(cov, dict):
                for k, v in cov.items():
                    rr[f"cov@{k}"] = v
                rr["coverage"] = json.dumps(cov)
            w.writerow(rr)

    # best by ICE and NLL among rows with probabilistic metrics
    valid = [r for r in rows if r.get("ice") is not None and r.get("nll") is not None]
    best_ice = min(valid, key=lambda r: r["ice"])
    best_nll = min(valid, key=lambda r: r["nll"])
    with (out_dir / "best_models.json").open("w") as f:
        json.dump({"best_ice": best_ice, "best_nll": best_nll}, f, indent=2, default=str)

    log.info("Saved %s", out_csv)
    log.info("Saved %s", out_dir / "best_models.json")


if __name__ == "__main__":
    main()
