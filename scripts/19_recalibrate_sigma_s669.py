"""Post-hoc σ recalibration on S669 (temperature scaling + isotonic regression).

Why
===
S669 results from `scripts/18_evaluate_on_s669.py`:

    Spearman(σ, |error|) = 0.434   ★ better than T2837 (0.348)
    ICE                  = 0.308   ✗ severely under-covered
    cov@90               = 0.374   (target 0.90)
    cov@95               = 0.447   (target 0.95)
    RMSE                 = 2.83    ← μ accuracy degraded vs T2837 (1.50)

The σ branch's *ranking* of |error| transferred from T2837 (where it was
trained) to S669 — and even improved with the 3.6× larger sample.  But
the *absolute scale* of σ is wrong: the mean prediction error is far
larger on S669, so σ is way too small relative to actual residuals.

This is a textbook case for post-hoc σ recalibration.  We run two methods
on a held-out calibration split of S669 and evaluate on the other split:

1. **Temperature scaling**:  σ' = T · σ, with T ≥ 0 chosen to minimise
   Gaussian NLL on the calibration set.  Closed-form when μ is fixed:
       T² = (1/N) Σ (y - μ)² / σ²
   Single global multiplier — preserves σ-ranking, only fixes the scale.

2. **Isotonic regression**:  fits a monotone σ_iso = f(σ) on the
   calibration set such that E[|y - μ| | σ_iso] is consistent with a
   half-normal at predicted σ_iso.  We fit `iso : σ → |residual|` and
   convert back to a Gaussian σ via σ' = (π/2)^(1/2) · iso(σ).
   Allows non-uniform stretch — fixes both scale *and* shape mismatch.

Both methods are monotone in σ, so Spearman(σ, |error|) is preserved.
Only ICE / NLL / coverage change.

What this script does
=====================
1. Loads the saved per-member predictions from
   `outputs/s669_eval_d5/per_member_predictions_s669.npz` (written by
   script 18).
2. Splits S669 randomly into calibration / evaluation halves (50/50 by
   default; protein-level split if --split-by-protein and the original
   metadata CSV is supplied).
3. Fits temperature T and an isotonic mapper on the calibration half.
4. Re-evaluates the eval half under three settings:
       • baseline σ
       • σ' = T·σ           (temperature scaling)
       • σ' = iso-mapped σ  (isotonic regression)
5. Saves a comparison JSON + CSV; pretty-prints the headline table.

Outputs under ``--out``
----------------------
- ``recalibration_summary.json``  per-method metrics on the eval split
- ``recalibration_summary.csv``   same in flat CSV form

Usage
-----
    python scripts/19_recalibrate_sigma_s669.py \\
        --predictions outputs/s669_eval_d5/per_member_predictions_s669.npz \\
        --out         outputs/s669_recalibration

    # Optional: protein-level split (more honest, requires the metadata csv)
    python scripts/19_recalibrate_sigma_s669.py \\
        --predictions outputs/s669_eval_d5/per_member_predictions_s669.npz \\
        --metadata    cache/s669_metadata_processed.csv \\
        --split-by-protein \\
        --out         outputs/s669_recalibration_protein
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uapp.evaluate import (
    compute_gaussian_nll,
    compute_ice,
    compute_mae,
    compute_rmse,
    compute_spearman_sigma_error,
    compute_top_k_risk_capture,
)
from uapp.utils import ensure_dir, setup_logging


# ─────────────────────────────────────────────────────────────────────────────
# Calibrators
# ─────────────────────────────────────────────────────────────────────────────
def fit_temperature(mu_cal: np.ndarray, sig_cal: np.ndarray, y_cal: np.ndarray) -> float:
    """Closed-form minimiser of Gaussian NLL with σ' = T·σ (μ fixed).

    NLL ∝ Σ ((y-μ)² / (T·σ)² + 2·log(T·σ))
    d/dT = 0  →  T² = (1/N) · Σ (y-μ)² / σ²
    """
    var_min = 1e-12
    z2 = (y_cal - mu_cal) ** 2 / np.maximum(sig_cal ** 2, var_min)
    return float(np.sqrt(np.mean(z2)))


class IsotonicCalibrator:
    """Wraps sklearn IsotonicRegression for σ → |residual| → Gaussian σ.

    fit:  iso.fit(σ_cal, |y_cal - μ_cal|)
    apply: σ_iso = sqrt(π/2) · clip(iso.predict(σ), σ_min)

    The √(π/2) factor converts E[|Z|] for Z ~ N(0, σ²) (= σ·√(2/π)) back
    into σ.  σ_min keeps the predictive density well-defined where the
    isotonic mapper might emit zero.
    """

    HALF_NORMAL_TO_SIGMA = math.sqrt(math.pi / 2.0)

    def __init__(self, sigma_floor: float = 1e-3):
        from sklearn.isotonic import IsotonicRegression
        self.iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        self.sigma_floor = sigma_floor

    def fit(self, sig_cal: np.ndarray, mu_cal: np.ndarray, y_cal: np.ndarray) -> None:
        residual = np.abs(y_cal - mu_cal)
        self.iso.fit(sig_cal, residual)

    def predict(self, sig: np.ndarray) -> np.ndarray:
        mapped = self.iso.predict(sig)
        sigma = self.HALF_NORMAL_TO_SIGMA * mapped
        return np.maximum(sigma, self.sigma_floor)


# ─────────────────────────────────────────────────────────────────────────────
# Splitting
# ─────────────────────────────────────────────────────────────────────────────
def random_split(n: int, frac_cal: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_cal = int(round(n * frac_cal))
    return idx[:n_cal], idx[n_cal:]


def protein_split(
    proteins: np.ndarray, frac_cal: float, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split *proteins* (not rows) into cal/eval, then return row indices.

    Cleaner than random row split: avoids near-duplicates from the same
    protein leaking calibration into evaluation.
    """
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(np.unique(proteins)))
    rng.shuffle(unique)
    n_cal = max(1, int(round(len(unique) * frac_cal)))
    cal_set = set(unique[:n_cal].tolist())
    is_cal = np.array([p in cal_set for p in proteins], dtype=bool)
    cal_idx = np.where(is_cal)[0]
    eval_idx = np.where(~is_cal)[0]
    return cal_idx, eval_idx


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
        "name": name, "n_eval": int(len(y)),
        "rmse": rmse, "mae": mae, "nll": nll, "ice": ice,
        "cov@0.50": cov.get("0.50"), "cov@0.80": cov.get("0.80"),
        "cov@0.90": cov.get("0.90"), "cov@0.95": cov.get("0.95"),
        "spearman": sp,
        "top0.10": tk["0.10"], "top0.20": tk["0.20"], "top0.30": tk["0.30"],
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--predictions", required=True, type=Path,
                   help="per_member_predictions_s669.npz from script 18 "
                        "(must contain mu_ens, sigma_ens, y).")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--metadata", type=Path, default=None,
                   help="Optional s669_metadata_processed.csv — used only "
                        "when --split-by-protein is set.")
    p.add_argument("--split-by-protein", action="store_true",
                   help="Split calibration/evaluation by pdb_code instead "
                        "of row.  Requires --metadata.")
    p.add_argument("--frac-cal", type=float, default=0.5,
                   help="Fraction of S669 used for calibration (default 0.5).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", type=str, default="INFO")
    args = p.parse_args()

    log = setup_logging(args.log_level)
    out_dir = ensure_dir(args.out)

    # ── Load ensemble predictions ───────────────────────────────────────────
    data = np.load(args.predictions, allow_pickle=False)
    mu  = np.asarray(data["mu_ens"], dtype=np.float64)
    sig = np.asarray(data["sigma_ens"], dtype=np.float64)
    y   = np.asarray(data["y"], dtype=np.float64)
    n = len(y)
    log.info("Loaded %d ensemble predictions from %s", n, args.predictions)

    # ── Split ───────────────────────────────────────────────────────────────
    if args.split_by_protein:
        if args.metadata is None:
            p.error("--split-by-protein requires --metadata")
        import pandas as pd
        md = pd.read_csv(args.metadata)
        if len(md) != n:
            p.error(
                f"metadata has {len(md)} rows but predictions have {n}.  "
                "Make sure --metadata is the *processed* CSV from script 01."
            )
        proteins = md["pdb_code"].astype(str).to_numpy()
        cal_idx, eval_idx = protein_split(proteins, args.frac_cal, args.seed)
        log.info("Protein-level split: %d cal proteins, %d eval proteins   "
                 "(%d cal rows, %d eval rows)",
                 len(set(proteins[cal_idx])), len(set(proteins[eval_idx])),
                 len(cal_idx), len(eval_idx))
    else:
        cal_idx, eval_idx = random_split(n, args.frac_cal, args.seed)
        log.info("Random split: %d cal rows, %d eval rows", len(cal_idx), len(eval_idx))

    mu_c, sig_c, y_c = mu[cal_idx], sig[cal_idx], y[cal_idx]
    mu_e, sig_e, y_e = mu[eval_idx], sig[eval_idx], y[eval_idx]

    # ── Fit calibrators on cal half ─────────────────────────────────────────
    T = fit_temperature(mu_c, sig_c, y_c)
    log.info("Temperature scaling:  T = %.4f", T)

    iso = IsotonicCalibrator()
    iso.fit(sig_c, mu_c, y_c)
    log.info("Isotonic fit on %d calibration rows.", len(cal_idx))

    # ── Evaluate on eval half ───────────────────────────────────────────────
    rows = [
        metrics_dict("baseline_sigma", mu_e, sig_e, y_e),
        metrics_dict("temperature_scaled", mu_e, T * sig_e, y_e),
        metrics_dict("isotonic_calibrated", mu_e, iso.predict(sig_e), y_e),
    ]

    # ── Save ────────────────────────────────────────────────────────────────
    summary = {
        "n_total": n,
        "n_cal": int(len(cal_idx)),
        "n_eval": int(len(eval_idx)),
        "split_by_protein": bool(args.split_by_protein),
        "frac_cal": args.frac_cal,
        "seed": args.seed,
        "temperature": T,
        "rows": rows,
    }
    (out_dir / "recalibration_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with (out_dir / "recalibration_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows: w.writerow(r)

    # ── Pretty print ────────────────────────────────────────────────────────
    fmt = lambda r: (f"{r['rmse']:9.4f} {r['nll']:9.4f} {r['ice']:9.4f} "
                     f"{r['cov@0.90']:9.4f} {r['cov@0.95']:9.4f} "
                     f"{r['spearman']:10.4f}")
    print()
    print("=" * 100)
    print(f"S669 σ recalibration — eval split (n = {len(eval_idx)})")
    print("=" * 100)
    print(f"{'method':<25} {'RMSE':>9} {'NLL':>9} {'ICE':>9} "
          f"{'cov@0.90':>9} {'cov@0.95':>9} {'Spearman':>10}")
    print("-" * 100)
    for r in rows:
        print(f"{r['name']:<25} {fmt(r)}")
    print("-" * 100)
    print(f"Temperature T = {T:.4f}  (σ multiplier).  "
          "Spearman is preserved by both monotone calibrators.")
    print("=" * 100)
    log.info("Saved %s", out_dir / "recalibration_summary.json")
    log.info("Saved %s", out_dir / "recalibration_summary.csv")


if __name__ == "__main__":
    main()
