"""Cache mutation-aware ESM-IF1 embeddings (structure-aware, pure-AA tokens).

Why ESM-IF
==========
§12 (SaProt) confirmed the encoder hypothesis — a structure-aware
backbone drops S669 RMSE from 2.83 to 2.53 — but cost T2837 σ-ranking
(0.348 → 0.218).  The plausible failure mode: SaProt's combined AA+3Di
tokens change the input vocabulary the σ branch attends to, breaking
the input-side regularities ESM2 had learned.

ESM-IF1 (Hsu et al. 2022, `esm.pretrained.esm_if1_gvp4_t16_142M_UR50`)
is a structure-aware encoder that takes a PDB and produces per-residue
embeddings via GVP-Transformer — but its input vocabulary is **pure
amino-acid tokens**, with structure entering through the GVP geometric
features, not the token alphabet.

If the σ-ranking degradation is intrinsic to *any* structure-aware
encoder, ESM-IF will reproduce SaProt's drop on T2837.  If it's
specific to SaProt's token-mixing, ESM-IF will preserve T2837 σ-ranking
while still helping S669 μ-accuracy — a strict win over the §11
ESM2-650M production model.

Output
======
Mutation-aware feature vector with the same 4-component shape as
scripts/01 and scripts/20:

    x = [ h_site, h_window, e(wtAA), e(mutAA) ]

ESM-IF's encoder hidden size is 512 (vs ESM2-650M's 1280), so the
total feature dim is 2·512 + 40 = **1064-d** (vs 2600-d for ESM2/SaProt).
The downstream FeatureAugmentedHead reads `d_in` dynamically, so all
existing scripts (06, 18, 19) work unchanged.

Prerequisites
-------------
- ``pip install fair-esm`` (or ``pip install 'fair-esm[esmfold]'``)
- Per-protein PDB files (provided externally; this script does NOT
  download them — see scripts/14 for AlphaFold-DB and S669 Zenodo zip
  for bundled WT PDBs).

Usage
-----
    # T2837:
    python scripts/22_cache_embeddings_esmif.py \\
        --metadata-csv cache/t2837_metadata.csv \\
        --pdb-dir      af_pdbs \\
        --pdb-pattern  'AF-{uniprot_id}.pdb' \\
        --pdb-chain    'A' \\
        --out          cache/t2837_embeddings_esmif.pt \\
        --metadata-out cache/t2837_metadata_esmif.csv \\
        --device cuda --seed 42

    # S669 (uses chain id from metadata):
    python scripts/22_cache_embeddings_esmif.py \\
        --metadata-csv cache/s669_metadata_with_pdb.csv \\
        --pdb-dir      data/s669/S669/pdbs \\
        --pdb-pattern  '{wt_pdb_basename}' \\
        --pdb-chain-col chain_id \\
        --val-fraction 0 --test-fraction 1.0 \\
        --out          cache/s669_embeddings_esmif.pt \\
        --metadata-out cache/s669_metadata_esmif.csv \\
        --device cuda --seed 42
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from uapp.data import save_cached_embeddings
from uapp.utils import set_seed, setup_logging


ESMIF_HF_ID = "esm.pretrained.esm_if1_gvp4_t16_142M_UR50"

AA3to1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}
AA1_LIST = sorted(set(AA3to1.values()))
AA1_TO_IDX = {aa: i for i, aa in enumerate(AA1_LIST)}


def aa_one_hot(aa_3letter: str) -> torch.Tensor:
    aa1 = AA3to1.get(aa_3letter, None)
    vec = torch.zeros(20)
    if aa1 is not None and aa1 in AA1_TO_IDX:
        vec[AA1_TO_IDX[aa1]] = 1.0
    return vec


def find_mutation_index(sequence: str, position: int, wt_aa_3: str) -> int | None:
    wt1 = AA3to1.get(wt_aa_3, '?')
    if wt1 == '?':
        return None
    idx = position - 1
    if 0 <= idx < len(sequence) and sequence[idx] == wt1:
        return idx
    for offset in range(1, 50):
        for sign in [-1, 1]:
            candidate = idx + sign * offset
            if 0 <= candidate < len(sequence) and sequence[candidate] == wt1:
                return candidate
    return None


def resolve_pdb_path(pdb_dir: Path, row: pd.Series, primary: str,
                     fallback: str | None) -> Path | None:
    fields = dict(row)
    fields["pdb_code_lower"] = str(row.get("pdb_code", "")).lower()
    for pat in [primary, fallback]:
        if pat is None:
            continue
        try:
            name = pat.format(**fields)
        except KeyError:
            continue
        candidate = pdb_dir / name
        if candidate.exists():
            return candidate
    return None


def build_mutation_feature(residue_embeddings: torch.Tensor, mut_idx: int,
                           wt_aa_3: str, mut_aa_3: str,
                           window_size: int = 5) -> torch.Tensor:
    seq_len = residue_embeddings.shape[0]
    h_i = residue_embeddings[mut_idx]
    win_start = max(0, mut_idx - window_size)
    win_end = min(seq_len, mut_idx + window_size + 1)
    h_window = residue_embeddings[win_start:win_end].mean(dim=0)
    return torch.cat([h_i, h_window, aa_one_hot(wt_aa_3), aa_one_hot(mut_aa_3)])


def split_by_protein(df: pd.DataFrame, val_fraction: float,
                     test_fraction: float, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    pdbs = df["pdb_code"].unique().tolist()
    rng.shuffle(pdbs)
    n_test = max(1, int(len(pdbs) * test_fraction))
    n_val = max(1, int(len(pdbs) * val_fraction))
    test_pdbs = set(pdbs[:n_test])
    val_pdbs = set(pdbs[n_test:n_test + n_val])
    df = df.copy()
    df["split"] = df["pdb_code"].apply(
        lambda p: "test" if p in test_pdbs else ("val" if p in val_pdbs else "train")
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ESM-IF model + structural representation
# ─────────────────────────────────────────────────────────────────────────────
def load_esmif(device: torch.device):
    """Load the ESM-IF1 inverse-folding model from the fair-esm package."""
    log = setup_logging()
    try:
        import esm
        import esm.inverse_folding  # noqa: F401  (registers the inverse-folding submodule)
    except ImportError as e:
        raise SystemExit(
            "ESM-IF requires the `fair-esm` package.  Install via "
            "`pip install fair-esm` then retry."
        ) from e
    log.info("loading ESM-IF1 (esm_if1_gvp4_t16_142M_UR50)")
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    log.info("ESM-IF1 loaded (%d params)",
             sum(p.numel() for p in model.parameters()))
    return model, alphabet


@torch.no_grad()
def get_residue_embeddings_esmif(pdb_path: Path, chain_id: str,
                                 model, alphabet, device: torch.device,
                                 ) -> tuple[torch.Tensor, str] | None:
    """Per-residue ESM-IF embeddings.

    Returns (rep, native_seq) where:
      rep        : Tensor of shape (L, 512) — encoder output.
      native_seq : 1-letter AA string of length L extracted from the PDB.

    Returns None if loading or inference fails.
    """
    import esm.inverse_folding as eif
    try:
        coords, native_seq = eif.util.load_coords(str(pdb_path), chain_id)
    except Exception as e:
        return None
    if coords is None or native_seq is None or len(native_seq) == 0:
        return None
    try:
        rep = eif.util.get_encoder_output(model, alphabet, coords)
        rep = rep.to("cpu")  # (L, 512)
    except Exception:
        return None
    if rep.shape[0] != len(native_seq):
        return None
    return rep, native_seq


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata-csv", required=True, type=Path)
    p.add_argument("--pdb-dir",      required=True, type=Path)
    p.add_argument("--pdb-pattern",  required=True, type=str)
    p.add_argument("--pdb-pattern-fallback", type=str, default=None)
    p.add_argument("--pdb-chain",     type=str, default="A",
                   help="Default chain id passed to esm.inverse_folding.util.load_coords. "
                        "Override with --pdb-chain-col to read it per-row from the CSV.")
    p.add_argument("--pdb-chain-col", type=str, default=None,
                   help="Column in the metadata CSV holding per-row chain id.")
    p.add_argument("--out",          required=True, type=Path)
    p.add_argument("--metadata-out", type=Path, default=None)
    p.add_argument("--window-size",  type=int, default=5)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--device",        type=str, default="auto")
    args = p.parse_args()

    log = setup_logging()
    set_seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    log.info("device: %s", device)

    df = pd.read_csv(args.metadata_csv)
    log.info("loaded %d rows from %s", len(df), args.metadata_csv)

    # 1. Resolve mutation indices against per-row sequence (same as script 01)
    log.info("resolving mutation positions to sequence indices...")
    df["mut_idx"] = [
        find_mutation_index(row["sequence"], int(row["position"]), row["wtAA"])
        for _, row in df.iterrows()
    ]
    n_before = len(df)
    df = df[df["mut_idx"].notna()].copy()
    df["mut_idx"] = df["mut_idx"].astype(int)
    log.info("position resolution: %d/%d mapped (dropped %d)",
             len(df), n_before, n_before - len(df))

    # 2. Splits
    if "split" not in df.columns or df["split"].isna().any():
        df = split_by_protein(df, args.val_fraction, args.test_fraction, args.seed)
    df["split"] = df["split"].astype(str).str.lower()
    for split in ("train", "val", "test"):
        log.info("  %s: %d mutations", split, (df["split"] == split).sum())

    # 3. Per unique protein: resolve PDB and chain, load ESM-IF representation
    model, alphabet = load_esmif(device)
    unique = df.drop_duplicates(subset=["pdb_code"]).reset_index(drop=True)
    log.info("computing ESM-IF representations for %d unique proteins...",
             len(unique))

    rep_cache: dict[str, torch.Tensor] = {}
    seq_cache: dict[str, str] = {}
    n_no_pdb = n_load_fail = 0
    for _, row in tqdm(unique.iterrows(), total=len(unique),
                       desc="ESM-IF reps", unit="prot"):
        pdb_path = resolve_pdb_path(args.pdb_dir, row, args.pdb_pattern,
                                    args.pdb_pattern_fallback)
        if pdb_path is None:
            n_no_pdb += 1; continue
        chain = (str(row[args.pdb_chain_col]).strip()
                 if args.pdb_chain_col else args.pdb_chain)
        result = get_residue_embeddings_esmif(pdb_path, chain, model, alphabet, device)
        if result is None:
            n_load_fail += 1; continue
        rep, native_seq = result
        rep_cache[str(row["pdb_code"])] = rep
        seq_cache[str(row["pdb_code"])] = native_seq
    log.info("ESM-IF coverage: %d ok, %d missing PDB, %d load/inference failed",
             len(rep_cache), n_no_pdb, n_load_fail)

    keep = df["pdb_code"].astype(str).isin(rep_cache)
    if int((~keep).sum()):
        log.warning("dropping %d rows whose protein has no ESM-IF representation",
                    int((~keep).sum()))
    df = df[keep].copy().reset_index(drop=True)

    # 4. Re-resolve mut_idx against the PDB-derived AA string
    log.info("re-resolving mut_idx against PDB-derived AA strings...")
    new_idx, n_aligned, n_unaligned = [], 0, 0
    for _, row in df.iterrows():
        pdb_aa = seq_cache[str(row["pdb_code"])]
        idx = find_mutation_index(pdb_aa, int(row["position"]), row["wtAA"])
        if idx is None:
            new_idx.append(None); n_unaligned += 1
        else:
            new_idx.append(idx); n_aligned += 1
    df["mut_idx_pdb"] = new_idx
    log.info("PDB-AA alignment: %d ok, %d unalignable (will be dropped)",
             n_aligned, n_unaligned)
    if n_unaligned:
        df = df[df["mut_idx_pdb"].notna()].copy().reset_index(drop=True)
    df["mut_idx"] = df["mut_idx_pdb"].astype(int)
    df.drop(columns=["mut_idx_pdb"], inplace=True)

    # 5. Build mutation-aware features
    embed_dim = next(iter(rep_cache.values())).shape[-1]
    log.info("building mutation-aware features (embed_dim=%d, window=%d)",
             embed_dim, args.window_size)
    features: list[torch.Tensor] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="mutation features",
                       unit="mut"):
        rep = rep_cache[str(row["pdb_code"])]
        idx = int(row["mut_idx"])
        if idx >= rep.shape[0]:
            idx = rep.shape[0] - 1
        features.append(build_mutation_feature(
            rep, idx, row["wtAA"], row["mutAA"], args.window_size,
        ))

    X_all = torch.stack(features, dim=0)
    y_all = torch.tensor(df["ddG"].values, dtype=torch.float32)
    splits_col = df["split"].values
    log.info("feature tensor: %s", tuple(X_all.shape))

    splits: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for split in ("train", "val", "test"):
        mask = splits_col == split
        splits[split] = (X_all[mask], y_all[mask])
        log.info("  %s: X=%s, y=%s", split,
                 tuple(splits[split][0].shape), tuple(splits[split][1].shape))

    # 6. Save
    log.info("saving to %s", args.out)
    save_cached_embeddings(
        args.out, splits,
        meta={
            "source": str(args.metadata_csv),
            "backbone": "ESM-IF1 (esm_if1_gvp4_t16_142M_UR50) mutation-aware",
            "embed_dim": int(X_all.shape[-1]),
            "components": (
                f"h_site({embed_dim}) + h_window({embed_dim}) "
                f"+ wt_onehot(20) + mut_onehot(20)"
            ),
            "window_size": args.window_size,
            "n_unique_proteins": len(rep_cache),
            "n_mapped": len(df),
            "seed": args.seed,
            "split_strategy": "by_pdb_code",
        },
    )
    log.info("done. %s (%.1f MB)", args.out, args.out.stat().st_size / 1e6)

    if args.metadata_out is not None:
        df.to_csv(args.metadata_out, index=False)
        log.info("saved processed metadata to %s", args.metadata_out)


if __name__ == "__main__":
    main()
