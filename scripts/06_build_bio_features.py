"""Build aligned biophysical features for an existing embedding cache.

The embedding cache (``cache/t2837_embeddings_v2.pt``) produced by
``scripts/01_cache_embeddings_esm_v2.py`` only stores ``(X, y)`` tensors —
it discards the per-sample mutation metadata (wt, mut, rsa) that the
σ-branch needs.  This script reconstructs those features and emits a
separate ``.pt`` file that is **row-aligned** with the embedding cache.

Which CSV to pass
-----------------
**Do NOT pass the raw T2837.csv here.**  Use the *processed* metadata
CSV that ``01_cache_embeddings_esm_v2.py`` saves alongside the cache
(default path: ``cache/t2837_metadata.csv``).  That file has:

  * exactly the rows that survived position-resolution (the raw CSV may
    have extra rows that were silently dropped)
  * the same row order used when building the embedding tensors
  * a ``split`` column that matches the cache's split keys

Amino-acid code handling
------------------------
The T2837 CSV stores amino acids as 3-letter codes (``ALA``, ``ARG``, …).
This script automatically detects whether values are 3-letter or 1-letter
and converts to 1-letter before computing biophysical features.

Row-count check
---------------
The script asserts ``len(split_in_csv) == cache_X.shape[0]`` for every
split before writing.  Any mismatch is a hard error — fix the upstream
source rather than silently mis-aligning σ-branch features with embeddings.

Output
------
A single ``.pt`` file:

    {
      "train": {"feats": Tensor[N_train, k]},
      "val":   {"feats": Tensor[N_val,   k]},
      "test":  {"feats": Tensor[N_test,  k]},
      "meta":  {"feature_names": [...], "mu": [...], "sd": [...], "k": k},
    }

Usage
-----
    # For the v2 cache (mutation-aware ESM-2 embeddings):
    python scripts/06_build_bio_features.py \\
        --metadata-csv cache/t2837_metadata.csv \\
        --embeddings   cache/t2837_embeddings_v2.pt \\
        --rsa-col      rel_rsa \\
        --out          cache/t2837_bio_features.pt

    # If your metadata CSV lives next to the embeddings (default name):
    python scripts/06_build_bio_features.py \\
        --embeddings cache/t2837_embeddings_v2.pt \\
        --out        cache/t2837_bio_features.pt
    # (script auto-discovers cache/t2837_metadata.csv if --metadata-csv is omitted)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uapp.data import load_cached_embeddings
from uapp.mutation_features import (
    batch_features, feature_dim, feature_names, standardize,
)
from uapp.utils import setup_logging

# Standard 3-letter → 1-letter conversion for T2837's wtAA/mutAA columns.
_AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _to_single_letter(series: pd.Series) -> pd.Series:
    """Convert 3-letter or 1-letter AA codes to upper-case 1-letter codes."""
    def convert(v: str) -> str:
        v = str(v).strip().upper()
        if len(v) == 1:
            return v
        if len(v) == 3 and v in _AA3TO1:
            return _AA3TO1[v]
        raise ValueError(f"unrecognised amino-acid code {v!r}")
    return series.apply(convert)


def _pick_col(df: pd.DataFrame, *candidates: str) -> str:
    """Case-insensitive column lookup; raises with a clear error if not found."""
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    raise KeyError(
        f"none of {candidates} found in CSV (columns: {list(df.columns)})"
    )


def _load_metadata(csv_path: Path, rsa_col: str) -> pd.DataFrame:
    """Load the processed metadata CSV and return a tidy DataFrame.

    Columns returned: wt (1-letter), mut (1-letter), rsa (float), split (str).
    Raises clearly if expected columns are missing.
    """
    df = pd.read_csv(csv_path)

    wt_col   = _pick_col(df, "wtAA", "wt_aa", "wt", "wild_type")
    mut_col  = _pick_col(df, "mutAA", "mut_aa", "mut", "mutant")
    rsa_col_ = _pick_col(df, rsa_col)
    spl_col  = _pick_col(df, "split", "set")

    out = pd.DataFrame({
        "wt":    _to_single_letter(df[wt_col]),
        "mut":   _to_single_letter(df[mut_col]),
        "rsa":   df[rsa_col_].astype(float),
        "split": df[spl_col].astype(str).str.strip().str.lower(),
    })
    return out


def _auto_discover_metadata(embeddings_path: Path) -> Path | None:
    """Try to find t2837_metadata.csv next to the cache file."""
    candidate = embeddings_path.parent / "t2837_metadata.csv"
    return candidate if candidate.exists() else None


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--metadata-csv",  type=Path, default=None,
                   help="Processed metadata CSV from 01_cache_embeddings_esm_v2.py "
                        "(default: auto-discovers <embeddings-dir>/t2837_metadata.csv)")
    p.add_argument("--embeddings",    required=True, type=Path)
    p.add_argument("--out",           required=True, type=Path)
    p.add_argument("--rsa-col",       default="rel_rsa",
                   help="Column name for relative solvent accessibility (default: rel_rsa)")
    p.add_argument("--include-indicators", action="store_true",
                   help="Also emit P/G/aromatic-mut indicator features (k=10 instead of 7)")
    p.add_argument("--log-level",     default="INFO")
    args = p.parse_args()

    log = setup_logging(args.log_level)

    # ── Resolve metadata CSV ──────────────────────────────────────────────────
    metadata_csv = args.metadata_csv
    if metadata_csv is None:
        metadata_csv = _auto_discover_metadata(args.embeddings)
        if metadata_csv is None:
            p.error(
                "--metadata-csv was not supplied and "
                f"no t2837_metadata.csv was found next to {args.embeddings}.  "
                "Please point --metadata-csv at the processed CSV saved by "
                "01_cache_embeddings_esm_v2.py (not at the raw T2837.csv)."
            )
        log.info("Auto-discovered metadata CSV: %s", metadata_csv)
    else:
        if not metadata_csv.exists():
            p.error(f"--metadata-csv not found: {metadata_csv}")

    # ── Load embedding cache (for row-count validation only) ──────────────────
    splits, cache_meta = load_cached_embeddings(args.embeddings)
    cache_sizes = {k: int(v[0].shape[0]) for k, v in splits.items()}
    log.info("Embedding-cache split sizes: %s", cache_sizes)

    # ── Load and validate metadata ────────────────────────────────────────────
    md = _load_metadata(metadata_csv, args.rsa_col)
    log.info("Loaded %d metadata rows from %s", len(md), metadata_csv)

    grouped = {s: g.reset_index(drop=True) for s, g in md.groupby("split")}
    csv_sizes = {k: len(v) for k, v in grouped.items()}
    log.info("CSV split sizes: %s", csv_sizes)

    # Strict alignment check — any size mismatch is a hard error.
    for split, n in cache_sizes.items():
        if split not in grouped:
            p.error(
                f"split {split!r} is in the embedding cache but missing "
                f"from the metadata CSV.  Available splits: {list(grouped)}"
            )
        csv_n = len(grouped[split])
        if csv_n != n:
            p.error(
                f"Row-count mismatch for split {split!r}: "
                f"embedding cache has {n} rows but metadata CSV has {csv_n}.  "
                "Make sure you are passing --metadata-csv to the *processed* "
                "CSV saved by 01_cache_embeddings_esm_v2.py, not the raw T2837.csv."
            )

    # ── Compute biophysical features per split ────────────────────────────────
    raw: dict[str, np.ndarray] = {}
    for split in cache_sizes:
        g = grouped[split]
        feats = batch_features(
            g["wt"].tolist(),
            g["mut"].tolist(),
            g["rsa"].to_numpy(),
            include_indicators=args.include_indicators,
        )
        raw[split] = feats
        log.info("  %s: bio-feature tensor shape %s", split, feats.shape)

    # ── Standardise on train statistics ──────────────────────────────────────
    if "train" not in raw:
        p.error("expected a 'train' split in the cache for fitting the standardiser")

    mu = raw["train"].mean(axis=0)
    sd = raw["train"].std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    standardised = {k: ((v - mu) / sd).astype(np.float32) for k, v in raw.items()}

    # ── Save ──────────────────────────────────────────────────────────────────
    payload: dict = {
        split: {"feats": torch.from_numpy(v)}
        for split, v in standardised.items()
    }
    payload["meta"] = {
        "feature_names":      feature_names(args.include_indicators),
        "k":                  feature_dim(args.include_indicators),
        "mu":                 mu.tolist(),
        "sd":                 sd.tolist(),
        "include_indicators": bool(args.include_indicators),
        "source_metadata":    str(metadata_csv),
        "source_embeddings":  str(args.embeddings),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    log.info("Saved aligned bio-feature tensor to %s", args.out)
    log.info(
        "Feature names (k=%d): %s",
        payload["meta"]["k"],
        payload["meta"]["feature_names"],
    )


if __name__ == "__main__":
    main()
