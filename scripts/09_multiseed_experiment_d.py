"""Multi-seed Track-D evaluation — does D2/D3 beat D0 statistically?

Single-seed runs of 07_experiment_d_real.py give a Spearman estimate with
CI ~±0.15 on T2837's n=170 test split.  That's too noisy to claim D2 or D3
beat D0.  This script repeats the full ablation with K seeds and reports:

  * per-seed metrics for every (ablation, scaling)
  * mean ± std per ablation
  * paired statistical tests (paired t-test + Wilcoxon signed-rank) of
    each challenger ablation vs D0 on Spearman / ICE / NLL, paired by seed
  * head-to-head wins per seed

Outputs
-------
    <out>/per_seed_results.csv     one row per (seed, ablation, scaling)
    <out>/aggregate.csv            mean / std per (ablation, scaling) per metric
    <out>/significance.json        paired-test results

Each per-seed run reuses scripts/07_experiment_d_real.py via subprocess so
the per-seed runs are guaranteed independent (fresh PyTorch state).

Usage
-----
    python scripts/09_multiseed_experiment_d.py \\
        --embeddings cache/t2837_embeddings_v2.pt \\
        --bio-feats  cache/t2837_bio_features.pt \\
        --out        outputs/experiment_d_multiseed \\
        --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from uapp.utils import setup_logging


METRIC_COLS = [
    "rmse", "mae", "nll", "ice",
    "cov@0.50", "cov@0.80", "cov@0.90", "cov@0.95",
    "spearman", "top0.10", "top0.20", "top0.30",
]


def run_one_seed(seed: int, args, out_subdir: Path) -> list[dict]:
    """Invoke 07_experiment_d_real.py with the given seed; load its JSON output."""
    out_subdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/07_experiment_d_real.py",
        "--embeddings", str(args.embeddings),
        "--bio-feats",  str(args.bio_feats),
        "--out",        str(out_subdir),
        "--seed",       str(seed),
        "--max-epochs", str(args.max_epochs),
        "--patience",   str(args.patience),
        "--log-level",  "WARNING",
    ]
    subprocess.run(cmd, check=True)
    rows = json.loads((out_subdir / "experiment_d_summary.json").read_text())
    for r in rows:
        r["seed"] = seed
    return rows


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (abl, scl), sub in per_seed.groupby(["ablation", "scaling"]):
        row = {"ablation": abl, "scaling": scl, "n_seeds": int(len(sub))}
        for m in METRIC_COLS:
            if m in sub.columns:
                row[f"{m}_mean"] = float(sub[m].mean())
                row[f"{m}_std"]  = float(sub[m].std(ddof=1)) if len(sub) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def paired_test(
    per_seed: pd.DataFrame, baseline: str, challenger: str, metric: str,
) -> dict:
    """Paired t-test + Wilcoxon for challenger vs baseline on a single metric.

    Pairs per-seed values (same seed → paired sample).  Higher = better is
    assumed for spearman/top-k; for nll/ice we still report (challenger −
    baseline) so a negative mean diff is the favourable direction.
    """
    base = (per_seed[(per_seed.ablation == baseline) & (per_seed.scaling == "raw")]
            .sort_values("seed")[metric].to_numpy())
    chal = (per_seed[(per_seed.ablation == challenger) & (per_seed.scaling == "raw")]
            .sort_values("seed")[metric].to_numpy())
    if len(base) != len(chal) or len(base) < 2:
        return {"n": int(len(base)), "skipped": "not enough paired seeds"}

    diff = chal - base
    t_stat, t_p = ttest_rel(chal, base)
    try:
        w_stat, w_p = wilcoxon(chal, base)
    except ValueError:  # all zeros
        w_stat, w_p = float("nan"), float("nan")
    return {
        "n":                  int(len(base)),
        "mean_diff":          float(diff.mean()),
        "std_diff":           float(diff.std(ddof=1)),
        "t_stat":             float(t_stat),
        "t_p_two_sided":      float(t_p),
        "wilcoxon_stat":      float(w_stat),
        "wilcoxon_p_two_sided": float(w_p),
        "wins_for_challenger": int((diff > 0).sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings", required=True, type=Path)
    p.add_argument("--bio-feats",  required=True, type=Path)
    p.add_argument("--out",        required=True, type=Path)
    p.add_argument("--seeds",      type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience",   type=int, default=25)
    p.add_argument("--keep-per-seed-dirs", action="store_true",
                   help="Keep per-seed output dirs (default: clean up to save disk)")
    args = p.parse_args()

    log = setup_logging("INFO")
    args.out.mkdir(parents=True, exist_ok=True)

    # ── Run K seeds ───────────────────────────────────────────────────────────
    all_rows: list[dict] = []
    for seed in args.seeds:
        sub = args.out / f"seed_{seed}"
        log.info("[seed %d] running 07_experiment_d_real.py …", seed)
        all_rows.extend(run_one_seed(seed, args, sub))
        if not args.keep_per_seed_dirs:
            shutil.rmtree(sub, ignore_errors=True)

    per_seed = pd.DataFrame(all_rows)
    per_seed.to_csv(args.out / "per_seed_results.csv", index=False)
    log.info("Saved %s", args.out / "per_seed_results.csv")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg = aggregate(per_seed)
    agg.to_csv(args.out / "aggregate.csv", index=False)
    log.info("Saved %s", args.out / "aggregate.csv")

    # ── Paired stats ──────────────────────────────────────────────────────────
    sig_results: dict[str, dict] = {}
    for metric in ["spearman", "ice", "nll", "top0.20"]:
        sig_results[metric] = {
            f"{chal}_vs_D0": paired_test(per_seed, "D0", chal, metric)
            for chal in ["D1", "D2", "D3"]
        }
    (args.out / "significance.json").write_text(json.dumps(sig_results, indent=2))

    # ── Pretty print ──────────────────────────────────────────────────────────
    print()
    print("=" * 84)
    print(f"Multi-seed Track-D summary  —  K = {len(args.seeds)} seeds: {args.seeds}")
    print("=" * 84)

    raw = agg[agg["scaling"] == "raw"].sort_values("ablation")
    print("\nRaw model metrics (mean ± std across seeds):\n")
    print(f"{'ablation':<6} {'RMSE':>16} {'NLL':>16} {'ICE':>16} {'Spearman':>16} {'top20':>16}")
    for _, r in raw.iterrows():
        def fmt(m: str) -> str:
            return f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_std']:.4f}"
        print(f"{r['ablation']:<6} {fmt('rmse'):>16} {fmt('nll'):>16} "
              f"{fmt('ice'):>16} {fmt('spearman'):>16} {fmt('top0.20'):>16}")

    print("\nPaired tests on Spearman  (challenger − D0):\n")
    print(f"{'comparison':<14} {'mean Δ':>10} {'p (t-test)':>12} {'wins':>10}")
    for k, v in sig_results["spearman"].items():
        if "skipped" in v:
            print(f"{k:<14} {v['skipped']}"); continue
        wins = f"{v['wins_for_challenger']}/{v['n']}"
        star = " *" if v["t_p_two_sided"] < 0.05 else ""
        print(f"{k:<14} {v['mean_diff']:+10.4f} {v['t_p_two_sided']:12.4f} {wins:>10}{star}")

    print("\nDecision (Spearman, α = 0.05):")
    for chal in ["D1", "D2", "D3"]:
        v = sig_results["spearman"][f"{chal}_vs_D0"]
        if "skipped" in v: continue
        verdict = "BEATS D0" if v["t_p_two_sided"] < 0.05 and v["mean_diff"] > 0 else "no significant difference"
        print(f"  {chal}: {verdict}  "
              f"(Δ={v['mean_diff']:+.4f}, p={v['t_p_two_sided']:.3f})")
    print("=" * 84)


if __name__ == "__main__":
    main()
