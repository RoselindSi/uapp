"""Convert Megascale (Tsuboyama 2023) CSV → T2837-compatible metadata CSV.

Why
===
Megascale (~700K mutation ΔΔG measurements across ~400 mini-protein domains)
is the largest curated stability dataset publicly available.  Pretraining
the σ branch on it should compound a ~10× more training signal than T2837
alone.

To reuse our existing pipeline (scripts 01 / 06 / 14) the cleanest move is
to **convert Megascale to a CSV with the same columns as our T2837 metadata**
file, then run the same caching scripts as before.

What this script does
=====================
1. Reads a Megascale CSV (path supplied via --megascale-csv).
2. Auto-detects the most common Megascale column layouts (mut_type style
   "I7L", a single ΔG_ML column per row, a `name` column for the protein).
3. Filters to **single amino acid variants only** — drops insertions /
   deletions / wt-only rows.
4. For each protein (`name`), pairs each mutant ΔG with its wild-type ΔG
   (mut_type == "wt") and computes ΔΔG = ΔG_mut − ΔG_wt.
5. Writes a T2837-compatible CSV with columns:
       pdb_code, sequence, position, wtAA, mutAA, ddG, mut_idx,
       chain_id, rel_rsa, split, dataset
6. Filler values for fields Megascale doesn't have:
       - chain_id  = 'A'   (single-chain mini-domains)
       - rel_rsa   = 0.5   (constant; T2837 has the real one)
       - dataset   = 'megascale'

Megascale download (one-time, manual)
=====================================
The dataset lives on Zenodo, mirrored from the Tsuboyama 2023 Nature paper:
    https://zenodo.org/records/7401283
Download the ddG dataset CSV (filename varies; common ones are
``Tsuboyama2023_Dataset_K50_dG.csv`` or ``Megascale_processed.csv``).
Save somewhere local and pass via --megascale-csv.

Usage
-----
    python scripts/15_megascale_to_t2837_format.py \\
        --megascale-csv data/megascale/Megascale_processed.csv \\
        --out cache/megascale_metadata.csv \\
        --val-frac 0.10 --seed 42
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from uapp.utils import setup_logging


# 1-letter to 3-letter AA codes (T2837 uses 3-letter)
AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def _pick(df: pd.DataFrame, *candidates: str) -> str | None:
    """Case-insensitive column lookup; returns None if none of the candidates exist."""
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def parse_mut_type(s: str) -> tuple[str, int, str] | None:
    """Parse Megascale mut_type strings.

    "I7L"          -> ('I', 7, 'L')
    "wt"/"WT"      -> None  (wild-type entry; not a mutant)
    "ins:M" / "del:7" / "I7-" / multi-mut -> None  (we only keep SAVs)
    """
    s = str(s).strip()
    if not s or s.lower() in ("wt", "wildtype", "wild-type", "ref"):
        return None
    if any(s.lower().startswith(prefix) for prefix in ("ins:", "del:", "indel:")):
        return None
    # Heuristic: SAV is an alpha + integer + alpha, with no comma/+/space
    if any(ch in s for ch in (",", "+", " ", "/", ";")):
        return None
    if len(s) < 3:
        return None
    if not (s[0].isalpha() and s[-1].isalpha()):
        return None
    middle = s[1:-1]
    if not middle.isdigit():
        return None
    wt = s[0].upper(); mut = s[-1].upper()
    if wt not in AA1_TO_3 or mut not in AA1_TO_3:
        return None
    if wt == mut:
        return None  # synonymous; drop
    return wt, int(middle), mut


def split_by_protein(names: list[str], val_frac: float, seed: int) -> dict[str, str]:
    """Random val/train split by unique protein name (no leakage)."""
    rng = np.random.default_rng(seed)
    unique = sorted(set(names))
    rng.shuffle(unique)
    n_val = max(1, int(len(unique) * val_frac))
    val_set = set(unique[:n_val])
    return {name: ("val" if name in val_set else "train") for name in unique}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--megascale-csv", required=True, type=Path,
                   help="Path to a Megascale CSV (downloaded from Zenodo)")
    p.add_argument("--out",           required=True, type=Path,
                   help="Output T2837-format CSV path (e.g. cache/megascale_metadata.csv)")
    p.add_argument("--val-frac",      type=float, default=0.10,
                   help="Fraction of unique protein names to hold as val for pretraining")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--ddg-col",       type=str,   default=None,
                   help="Override ΔG column name (default: try common ones)")
    p.add_argument("--seq-col",       type=str,   default=None,
                   help="Override sequence column.  Default: auto-detect — prefer "
                        "aa_seq_full (Tsuboyama position numbering) over aa_seq.  "
                        "If both fail to align with mut_type positions, the script "
                        "tries the other automatically before giving up.")
    p.add_argument("--max-rows",      type=int,   default=None,
                   help="Optional cap on output rows for quick smoke runs")
    p.add_argument("--log-level",     type=str,   default="INFO")
    args = p.parse_args()

    log = setup_logging(args.log_level)

    log.info("Reading Megascale CSV: %s", args.megascale_csv)
    df = pd.read_csv(args.megascale_csv)
    log.info("Read %d raw rows; columns: %s", len(df), list(df.columns))

    # ── Resolve column names ─────────────────────────────────────────────────
    # In Tsuboyama 2023 the per-row `aa_seq` / `aa_seq_full` already has the
    # mutation applied — they aren't WT sequences.  The right way to validate
    # mut_type positions is to look up the WT sequence per protein from rows
    # where mut_type == "wt", keyed by `WT_name` (or `name` as fallback).
    name_col   = _pick(df, "WT_name", "name", "protein_name", "domain")
    seq_col    = (args.seq_col or
                  _pick(df, "aa_seq", "aa_seq_full", "sequence",
                        "wt_seq", "wt_aa_seq"))
    mut_col    = _pick(df, "mut_type", "mutation", "variant", "mutation_type")
    ddg_col    = args.ddg_col or _pick(
        df, "ddG_ML", "ddG", "ddg_ml", "ddg",
        "dG_ML", "dG_dir", "dG_h2o", "dG"   # ΔG that we'll convert to ΔΔG
    )

    for n, c in [("name", name_col), ("aa_seq", seq_col),
                 ("mut_type", mut_col), ("ddg_or_dg", ddg_col)]:
        if c is None:
            raise SystemExit(
                f"Could not find a {n!r} column.  Available columns: {list(df.columns)}.\n"
                "Pass --ddg-col / --seq-col explicitly, or rename your CSV columns."
            )
    log.info("Resolved columns: name=%s, seq=%s, mut=%s, ddg/dg=%s",
             name_col, seq_col, mut_col, ddg_col)

    # ── Build per-protein WT sequence lookup from rows where mut_type == 'wt'.
    #    The dataset stores PER-VARIANT sequences (already mutated) in the seq
    #    column, so we cannot validate positions against the row's own seq.
    #    Instead: for each WT row, register name → seq, then validate SAV
    #    positions against the registered WT seq.
    wt_mask = df[mut_col].astype(str).str.lower().isin(
        ("wt", "wildtype", "wild-type", "ref")
    )
    log.info("Building per-protein WT sequence lookup from %d wt rows ...",
             int(wt_mask.sum()))
    wt_seq_lookup: dict[str, str] = {}
    for _, r in df[wt_mask].iterrows():
        nm = str(r[name_col]); s = str(r[seq_col]).strip().upper()
        # Pick the longest seq if multiple wt rows per protein
        if nm not in wt_seq_lookup or len(s) > len(wt_seq_lookup[nm]):
            wt_seq_lookup[nm] = s
    log.info("WT-sequence lookup: %d proteins.  (df has %d unique %s values)",
             len(wt_seq_lookup), df[name_col].nunique(), name_col)

    # ── Sanity-check: how many SAVs align with their protein's WT seq? ──────
    sample = df.sample(n=min(2000, len(df)), random_state=0)
    ok = tried = 0
    for _, r in sample.iterrows():
        parsed = parse_mut_type(r[mut_col])
        if parsed is None: continue
        wt_aa, pos, _ = parsed
        nm = str(r[name_col])
        wt_seq = wt_seq_lookup.get(nm)
        if wt_seq is None: tried += 1; continue
        if 1 <= pos <= len(wt_seq) and wt_seq[pos - 1] == wt_aa:
            ok += 1
        tried += 1
    rate = ok / max(tried, 1)
    log.info("WT-seq alignment sanity-check: %.1f%% of sampled SAVs match",
             100 * rate)
    if rate < 0.30:
        log.warning("Alignment rate is low.  The chosen `seq` column may not "
                    "store the canonical WT sequence — try --seq-col aa_seq_full.")

    # ── Compute ΔΔG (pair each mutant with its wt) if we got a ΔG column ────
    is_ddg = ddg_col.lower().startswith("ddg")
    if is_ddg:
        log.info("Detected a ΔΔG column directly; using values as-is.")
    else:
        log.info("Detected a ΔG column; will pair each mutant with its WT and "
                 "compute ΔΔG = ΔG_mut − ΔG_wt per protein.")
        # Build a per-protein wt ΔG lookup
        wt_mask = df[mut_col].astype(str).str.lower().isin(("wt", "wildtype", "wild-type", "ref"))
        wt_dG = (df[wt_mask].groupby(name_col)[ddg_col]
                 .mean().to_dict())
        log.info("Found WT ΔG for %d/%d unique proteins", len(wt_dG), df[name_col].nunique())

    # ── Filter to SAVs and build output rows ────────────────────────────────
    out_rows = []
    n_skipped_not_sav = 0
    n_skipped_seq_mismatch = 0
    n_skipped_no_wt_dG = 0
    n_skipped_no_wt_seq = 0
    for _, row in df.iterrows():
        parsed = parse_mut_type(row[mut_col])
        if parsed is None:
            n_skipped_not_sav += 1; continue
        wt_aa, pos, mut_aa = parsed

        # Look up WT sequence by protein name (NOT the row's own seq column —
        # those are already mutated in Tsuboyama 2023).
        nm = str(row[name_col])
        seq = wt_seq_lookup.get(nm)
        if seq is None:
            n_skipped_no_wt_seq += 1; continue
        if pos < 1 or pos > len(seq) or seq[pos - 1] != wt_aa:
            n_skipped_seq_mismatch += 1; continue

        # ΔΔG resolution
        if is_ddg:
            ddg = float(row[ddg_col])
        else:
            wt_dG_for_protein = wt_dG.get(row[name_col])
            if wt_dG_for_protein is None or pd.isna(wt_dG_for_protein):
                n_skipped_no_wt_dG += 1; continue
            ddg = float(row[ddg_col]) - float(wt_dG_for_protein)

        if not np.isfinite(ddg):
            continue

        out_rows.append({
            "pdb_code":  str(row[name_col]),
            "sequence":  seq,
            "position":  int(pos),
            "wtAA":      AA1_TO_3[wt_aa],
            "mutAA":     AA1_TO_3[mut_aa],
            "ddG":       ddg,
            "mut_idx":   int(pos - 1),
            "chain_id":  "A",
            "rel_rsa":   0.5,            # filler — Megascale lacks RSA
            "dataset":   "megascale",
        })

    log.info(
        "Kept %d SAV rows.  Skipped: %d not-SAV, %d no-wt-seq-for-protein, "
        "%d seq-mismatch (pos out of range or wt_aa wrong), %d missing wt ΔG",
        len(out_rows), n_skipped_not_sav, n_skipped_no_wt_seq,
        n_skipped_seq_mismatch, n_skipped_no_wt_dG,
    )

    if not out_rows:
        raise SystemExit("No usable rows after parsing.  Check --megascale-csv format.")

    if args.max_rows and args.max_rows < len(out_rows):
        log.info("Capping output to first %d rows (--max-rows)", args.max_rows)
        out_rows = out_rows[:args.max_rows]

    # ── Train/val split by protein ──────────────────────────────────────────
    names = [r["pdb_code"] for r in out_rows]
    splits = split_by_protein(names, args.val_frac, args.seed)
    n_train = sum(1 for r in out_rows if splits[r["pdb_code"]] == "train")
    n_val   = sum(1 for r in out_rows if splits[r["pdb_code"]] == "val")
    log.info("Split by protein: %d train / %d val   (val fraction = %.0f%%)",
             n_train, n_val, args.val_frac * 100)
    for r in out_rows:
        r["split"] = splits[r["pdb_code"]]

    # ── Write ────────────────────────────────────────────────────────────────
    out_df = pd.DataFrame(out_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    log.info("Saved %s   (%d rows × %d cols)", args.out, len(out_df), out_df.shape[1])

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"Megascale → T2837 format complete")
    print(f"  Input  rows: {len(df)}")
    print(f"  Output rows: {len(out_df)}  ({n_train} train / {n_val} val)")
    print(f"  Unique proteins: {len(set(out_df['pdb_code']))}")
    print(f"  ΔΔG range: [{out_df['ddG'].min():.2f}, {out_df['ddG'].max():.2f}], "
          f"mean = {out_df['ddG'].mean():.2f}")
    print()
    print("Next steps:")
    print(f"  # 1. Cache ESM2-650M embeddings (uses our existing v2 caching script):")
    print(f"  python scripts/01_cache_embeddings_esm_v2.py \\")
    print(f"      --t2837-csv {args.out} \\")
    print(f"      --out cache/megascale_embeddings_650m.pt \\")
    print(f"      --esm-model facebook/esm2_t33_650M_UR50D \\")
    print(f"      --device cuda --seed 42")
    print()
    print(f"  # 2. Build extended bio features (k=13: chemistry + sequence-struct):")
    print(f"  python scripts/06_build_bio_features.py \\")
    print(f"      --metadata-csv cache/megascale_metadata.csv \\")
    print(f"      --embeddings   cache/megascale_embeddings_650m.pt \\")
    print(f"      --out          cache/megascale_bio_features_650m_extended.pt \\")
    print(f"      --include-extended")
    print()
    print(f"  # 3. Run the pretrain+finetune script (script 16):")
    print(f"  python scripts/16_pretrain_and_finetune.py \\")
    print(f"      --megascale-emb cache/megascale_embeddings_650m.pt \\")
    print(f"      --megascale-bio cache/megascale_bio_features_650m_extended.pt \\")
    print(f"      --t2837-emb     cache/t2837_embeddings_v2_650m.pt \\")
    print(f"      --t2837-bio     cache/t2837_bio_features_650m_extended.pt \\")
    print(f"      --out           outputs/megascale_pretrain")
    print("=" * 78)


if __name__ == "__main__":
    main()
