"""Convert S669 release CSV → T2837-compatible metadata CSV for external evaluation.

Why
===
S669 (Pancotti et al. 2022, *Briefings in Bioinformatics*) is the standard
held-out benchmark for ddG predictors: 669 single-point variants across 94
proteins, deliberately chosen to have low overlap with common training sets.
Reporting on S669 is what turns "we have a number on T2837" into "we have a
generalising model".

What this script does
=====================
1. Reads an S669 release CSV (path supplied via --s669-csv).
2. Auto-detects the most common S669 column layouts:
     * Protein / PDB id     (`Protein`, `PDB`, `pdb_id`)
     * Mutation string      (`Mutation`, `mut`, e.g. "A123V" or "1A2P_A_123_A_V")
     * ddG experimental     (`ddG`, `ddG_exp`, `DDG`, `experimental_ddG`)
     * Sequence (per-row)   (`sequence`, `wt_seq`, `seq`) — optional if --fasta given
     * Chain                (`Chain`, `chain`) — optional, defaults to 'A'
3. Filters to single-AA variants whose `position` and `wtAA` align with the
   provided WT sequence.
4. Writes a T2837-compatible CSV with all rows tagged `split = 'test'` so
   downstream caching / evaluation treats S669 strictly as a test fold.

Output columns (matching T2837):
    pdb_code, sequence, position, wtAA, mutAA, ddG, mut_idx,
    chain_id, rel_rsa, split, dataset

Filler values for fields S669 doesn't always have:
    chain_id  : from --chain or row's Chain column, default 'A'
    rel_rsa   : 0.5  (constant; downstream script 06 / 14 will recompute real values)
    split     : 'test'  (S669 is held-out; no train/val split)
    dataset   : 's669'

S669 download
=============
The most-cited release is Pancotti et al. 2022 supplementary, mirrored at:
    https://github.com/JinyuanSun/MutationDDG-Bench/tree/main/s669
    https://zenodo.org/records/7268129
Save the CSV locally and pass it via --s669-csv.

Usage
-----
    # Per-row sequence column present:
    python scripts/17_s669_to_t2837_format.py \\
        --s669-csv data/s669/S669.csv \\
        --out cache/s669_metadata.csv

    # Per-row sequence missing, supply a FASTA keyed by protein id:
    python scripts/17_s669_to_t2837_format.py \\
        --s669-csv data/s669/S669.csv \\
        --fasta    data/s669/wildtypes.fasta \\
        --out cache/s669_metadata.csv

    # Override column auto-detection:
    python scripts/17_s669_to_t2837_format.py \\
        --s669-csv data/s669/S669.csv \\
        --protein-col PDB --mut-col Mutation --ddg-col DDG \\
        --out cache/s669_metadata.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from uapp.utils import setup_logging


AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "E": "GLU", "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def _pick(df: pd.DataFrame, *candidates: str) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


# Match either "A123V" or "1A2P_A_123_A_V" / "1A2P:A:A123V"
_MUT_SHORT = re.compile(r"^([A-Z])(\d+)([A-Z])$")
_MUT_LONG  = re.compile(r"^(?P<pdb>[A-Za-z0-9]+)[_:](?P<chain>[A-Za-z])[_:]"
                        r"(?:(?P<wt1>[A-Z])(?P<pos1>\d+)(?P<mut1>[A-Z])"
                        r"|(?P<pos2>\d+)[_:](?P<wt2>[A-Z])[_:](?P<mut2>[A-Z]))$")


def parse_mutation(s: str) -> tuple[str, int, str] | None:
    """Parse an S669 mutation string into (wtAA, position, mutAA).  None if unparseable."""
    s = str(s).strip()
    if not s:
        return None
    m = _MUT_SHORT.match(s)
    if m:
        wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
    else:
        m = _MUT_LONG.match(s)
        if not m:
            return None
        wt  = m.group("wt1")  or m.group("wt2")
        pos = int(m.group("pos1") or m.group("pos2"))
        mut = m.group("mut1") or m.group("mut2")
    if wt not in AA1_TO_3 or mut not in AA1_TO_3 or wt == mut:
        return None
    return wt, pos, mut


def load_fasta(path: Path) -> dict[str, str]:
    """Tiny FASTA reader: returns dict[protein_id → uppercase WT sequence]."""
    seqs: dict[str, str] = {}
    cur_id, cur_buf = None, []
    with path.open() as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if cur_id is not None:
                    seqs[cur_id] = "".join(cur_buf).upper()
                # Take first whitespace-separated token after '>' as id
                cur_id = line[1:].split()[0]
                cur_buf = []
            else:
                cur_buf.append(line.strip())
    if cur_id is not None:
        seqs[cur_id] = "".join(cur_buf).upper()
    return seqs


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--s669-csv", required=True, type=Path)
    p.add_argument("--out",      required=True, type=Path)
    p.add_argument("--fasta",    type=Path, default=None,
                   help="Optional FASTA of WT sequences keyed by protein id "
                        "(used when the CSV has no per-row sequence column).")
    p.add_argument("--protein-col", type=str, default=None)
    p.add_argument("--mut-col",     type=str, default=None)
    p.add_argument("--ddg-col",     type=str, default=None)
    p.add_argument("--seq-col",     type=str, default=None)
    p.add_argument("--chain-col",   type=str, default=None)
    p.add_argument("--chain",       type=str, default="A",
                   help="Default chain id when CSV/fasta provide none.")
    p.add_argument("--log-level",   type=str, default="INFO")
    args = p.parse_args()

    log = setup_logging(args.log_level)

    log.info("Reading S669 CSV: %s", args.s669_csv)
    df = pd.read_csv(args.s669_csv)
    log.info("Read %d raw rows; columns: %s", len(df), list(df.columns))

    protein_col = args.protein_col or _pick(
        df, "pdb_code", "PDB", "Protein", "protein", "pdb_id", "pdb",
    )
    mut_col = args.mut_col or _pick(
        df, "Mutation", "mutation", "mut", "variant", "mut_type",
    )
    ddg_col = args.ddg_col or _pick(
        df, "ddG", "ddG_exp", "DDG", "ddg", "experimental_ddG", "ddg_exp",
    )
    seq_col = args.seq_col or _pick(
        df, "sequence", "wt_seq", "wt_sequence", "seq", "aa_seq",
    )
    chain_col = args.chain_col or _pick(df, "Chain", "chain", "chain_id")

    for n, c in [("protein", protein_col), ("mutation", mut_col), ("ddG", ddg_col)]:
        if c is None:
            raise SystemExit(
                f"Could not auto-detect a {n!r} column.  Available: {list(df.columns)}.  "
                f"Pass --{n}-col explicitly."
            )

    fasta_seqs: dict[str, str] = {}
    if args.fasta is not None:
        log.info("Reading FASTA: %s", args.fasta)
        fasta_seqs = load_fasta(args.fasta)
        log.info("FASTA: %d sequences", len(fasta_seqs))

    if seq_col is None and not fasta_seqs:
        raise SystemExit(
            "S669 CSV has no sequence column and no --fasta supplied.  "
            "Provide one of: --seq-col, --fasta."
        )

    # ── Build per-protein WT sequence lookup ────────────────────────────────
    wt_seq_lookup: dict[str, str] = dict(fasta_seqs)
    if seq_col is not None:
        for _, r in df.iterrows():
            nm = str(r[protein_col]).strip()
            seq = str(r[seq_col]).strip().upper() if pd.notna(r[seq_col]) else ""
            if not seq:
                continue
            # Keep the longest seq per protein id (defensive against truncated dupes)
            if nm not in wt_seq_lookup or len(seq) > len(wt_seq_lookup[nm]):
                wt_seq_lookup[nm] = seq
    log.info("WT-sequence lookup: %d proteins", len(wt_seq_lookup))

    # ── Parse rows ──────────────────────────────────────────────────────────
    out_rows = []
    n_skipped_unparseable = 0
    n_skipped_no_seq = 0
    n_skipped_seq_mismatch = 0
    n_skipped_bad_ddg = 0

    for _, row in df.iterrows():
        parsed = parse_mutation(row[mut_col])
        if parsed is None:
            n_skipped_unparseable += 1; continue
        wt_aa, pos, mut_aa = parsed

        nm = str(row[protein_col]).strip()
        seq = wt_seq_lookup.get(nm)
        if seq is None:
            # Try a normalised key (some S669 distributions use "1A2P_A" vs "1A2P")
            base = nm.split("_")[0].split(":")[0]
            seq = wt_seq_lookup.get(base)
        if seq is None:
            n_skipped_no_seq += 1; continue

        if pos < 1 or pos > len(seq) or seq[pos - 1] != wt_aa:
            n_skipped_seq_mismatch += 1; continue

        try:
            ddg = float(row[ddg_col])
        except (TypeError, ValueError):
            n_skipped_bad_ddg += 1; continue
        if not np.isfinite(ddg):
            n_skipped_bad_ddg += 1; continue

        chain = str(row[chain_col]) if (chain_col and pd.notna(row[chain_col])) else args.chain

        out_rows.append({
            "pdb_code":  nm,
            "sequence":  seq,
            "position":  int(pos),
            "wtAA":      AA1_TO_3[wt_aa],
            "mutAA":     AA1_TO_3[mut_aa],
            "ddG":       ddg,
            "mut_idx":   int(pos - 1),
            "chain_id":  chain,
            "rel_rsa":   0.5,        # filler; script 06 / 14 recompute
            "split":     "test",     # S669 is held-out
            "dataset":   "s669",
        })

    log.info(
        "Kept %d rows.  Skipped: %d unparseable mutations, %d no-WT-seq, "
        "%d seq-mismatch, %d bad-ddG",
        len(out_rows), n_skipped_unparseable, n_skipped_no_seq,
        n_skipped_seq_mismatch, n_skipped_bad_ddg,
    )
    if not out_rows:
        raise SystemExit("No usable rows after parsing.  Check column overrides.")

    out_df = pd.DataFrame(out_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    log.info("Saved %s   (%d rows × %d cols)", args.out, len(out_df), out_df.shape[1])

    print("\n" + "=" * 78)
    print("S669 → T2837 format complete")
    print(f"  Input  rows: {len(df)}")
    print(f"  Output rows: {len(out_df)}  (all split='test')")
    print(f"  Unique proteins: {out_df['pdb_code'].nunique()}")
    print(f"  ΔΔG range: [{out_df['ddG'].min():.2f}, {out_df['ddG'].max():.2f}], "
          f"mean = {out_df['ddG'].mean():.2f}")
    print()
    print("Next steps:")
    print(f"  # 1. Cache ESM2-650M embeddings for S669:")
    print(f"  python scripts/01_cache_embeddings_esm_v2.py \\")
    print(f"      --t2837-csv {args.out} \\")
    print(f"      --out cache/s669_embeddings_650m.pt \\")
    print(f"      --metadata-out cache/s669_metadata_processed.csv \\")
    print(f"      --val-fraction 0 --test-fraction 1.0 \\")
    print(f"      --esm-model facebook/esm2_t33_650M_UR50D \\")
    print(f"      --device cuda --seed 42")
    print()
    print(f"  # 2. Build extended bio features (k=13):")
    print(f"  python scripts/06_build_bio_features.py \\")
    print(f"      --metadata-csv cache/s669_metadata_processed.csv \\")
    print(f"      --embeddings   cache/s669_embeddings_650m.pt \\")
    print(f"      --out          cache/s669_bio_features_650m_extended.pt \\")
    print(f"      --include-extended")
    print()
    print(f"  # 3. (Optional) Add DSSP + pLDDT (k=21) for D6:")
    print(f"  python scripts/14_compute_structural_features.py \\")
    print(f"      --metadata-csv  cache/s669_metadata_processed.csv \\")
    print(f"      --embeddings    cache/s669_embeddings_650m.pt \\")
    print(f"      --extended-bio  cache/s669_bio_features_650m_extended.pt \\")
    print(f"      --out           cache/s669_bio_features_650m_dssp.pt")
    print()
    print(f"  # 4. Train D6 ensemble on T2837, evaluate on S669:")
    print(f"  python scripts/18_evaluate_on_s669.py \\")
    print(f"      --t2837-emb cache/t2837_embeddings_v2_650m.pt \\")
    print(f"      --t2837-bio cache/t2837_bio_features_650m_dssp.pt \\")
    print(f"      --s669-emb  cache/s669_embeddings_650m.pt \\")
    print(f"      --s669-bio  cache/s669_bio_features_650m_dssp.pt \\")
    print(f"      --ablation D6 --members 5 --device cuda")
    print("=" * 78)


if __name__ == "__main__":
    main()
