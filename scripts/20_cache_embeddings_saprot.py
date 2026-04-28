"""Cache mutation-aware SaProt embeddings.

Why SaProt
==========
SaProt (Westlake/SJTU 2024) is a structure-augmented ESM-style PLM that
takes a *combined* AA + 3Di token per residue.  The 3Di alphabet
(20 letters) comes from FoldSeek's structural tokenisation of a PDB
file.  Where ESM2 sees only sequence, SaProt sees `<aa><3di>` — making
it sensitive to local geometry that vanilla ESM2 cannot encode.

§11 of REPORT.md showed that ESM2-650M's σ-branch ranking transferred
from T2837 to S669 (Spearman 0.348 → 0.434), but μ-accuracy did not
(RMSE 1.50 → 2.83).  The campaign so far swapped datasets twice but
never the encoder; structure-awareness is the most-targeted next axis
for the μ failure.

Pipeline
========
1. For each unique protein in the metadata CSV, locate its PDB file
   (T2837 uses cached AlphaFold-DB models keyed by ``uniprot_id``;
   S669 uses the WT PDBs bundled in the Zenodo release).
2. Run ``foldseek structureto3didescriptor`` on each PDB to extract
   the per-residue 3Di letter string.
3. Build SaProt combined tokens (`<aa><3di>` per residue) and run the
   SaProt model to get per-residue embeddings (1280-d).
4. Build the same mutation-aware feature vector script 01 builds:
       x = [ h_site, h_window±k, e(wtAA), e(mutAA) ]   → 2600-d
   so the saved cache is a drop-in replacement for the ESM2 cache and
   the existing scripts (06, 14, 18, 19) work unchanged.

The 1280 hidden dim is identical to ESM2-650M, so the downstream
heads, ablations, and metric tooling don't need any changes.

Prerequisites
-------------
- ``foldseek`` binary on PATH (``apt-get install foldseek`` or build
  from https://github.com/steineggerlab/foldseek).
- HF model ``westlake-repl/SaProt_650M_AF2``  (≈ 2.6 GB, public).
- Per-protein PDB files (provided externally; this script does NOT
  download them — see scripts/14 for the AlphaFold-DB downloader and
  the S669 Zenodo zip for the bundled WT PDBs).

Usage
-----
    # T2837 (uses the AlphaFold-DB cache built by scripts/14):
    python scripts/20_cache_embeddings_saprot.py \\
        --metadata-csv cache/t2837_metadata.csv \\
        --pdb-dir      af_pdbs \\
        --pdb-pattern  "AF-{uniprot_id}.pdb" \\
        --out          cache/t2837_embeddings_saprot.pt \\
        --metadata-out cache/t2837_metadata_saprot.csv \\
        --device cuda --seed 42

    # S669 (uses the Zenodo bundled WT PDBs):
    python scripts/20_cache_embeddings_saprot.py \\
        --metadata-csv cache/s669_metadata.csv \\
        --pdb-dir      data/s669/S669/pdbs \\
        --pdb-pattern  "{pdb_code}.pdb" \\
        --pdb-pattern-fallback "{pdb_code_lower}.pdb" \\
        --val-fraction 0 --test-fraction 1.0 \\
        --out          cache/s669_embeddings_saprot.pt \\
        --metadata-out cache/s669_metadata_saprot.csv \\
        --device cuda --seed 42
"""
from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from uapp.data import save_cached_embeddings
from uapp.utils import set_seed, setup_logging


SAPROT_MODEL = "westlake-repl/SaProt_650M_AF2"

AA3to1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}
AA1_LIST = sorted(set(AA3to1.values()))
AA1_TO_IDX = {aa: i for i, aa in enumerate(AA1_LIST)}

# 3Di alphabet — 20 lowercase letters (per FoldSeek's spec)
THREE_DI_ALPHABET = set("acdefghiklmnpqrstvwy")


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


# ─────────────────────────────────────────────────────────────────────────────
# FoldSeek 3Di extraction
# ─────────────────────────────────────────────────────────────────────────────
def check_foldseek_available() -> None:
    if shutil.which("foldseek") is None:
        raise SystemExit(
            "foldseek binary not found on PATH.  Install it via "
            "`apt-get install foldseek` (Ubuntu / Colab) or download a "
            "binary from https://github.com/steineggerlab/foldseek."
        )


def extract_3di(pdb_path: Path, log) -> tuple[str, str] | None:
    """Run `foldseek structureto3didescriptor` on a PDB.

    Returns
    -------
    (aa_seq, three_di_seq)  both uppercase / lowercase strings of equal
    length, or None if the call failed or output was empty.

    The output format is one line per chain:
        > name <tab> aa_seq <tab> three_di_seq
    We take the longest chain (= primary) and keep its tokens.
    """
    with tempfile.TemporaryDirectory() as td:
        out_tsv = Path(td) / "out.tsv"
        cmd = [
            "foldseek", "structureto3didescriptor",
            str(pdb_path), str(out_tsv),
        ]
        try:
            res = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        except subprocess.CalledProcessError as e:
            log.warning("foldseek failed for %s: %s", pdb_path.name, e.stderr.strip()[:200])
            return None
        except subprocess.TimeoutExpired:
            log.warning("foldseek timed out for %s", pdb_path.name)
            return None
        if not out_tsv.exists():
            return None
        candidates: list[tuple[str, str]] = []
        for line in out_tsv.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            aa = parts[1].strip().upper()
            tdi = parts[2].strip().lower()
            if len(aa) != len(tdi) or not aa:
                continue
            candidates.append((aa, tdi))
        if not candidates:
            return None
        # Pick the longest chain
        aa, tdi = max(candidates, key=lambda x: len(x[0]))
        return aa, tdi


# ─────────────────────────────────────────────────────────────────────────────
# PDB resolution: look up a per-row PDB filename via a pattern
# ─────────────────────────────────────────────────────────────────────────────
def resolve_pdb_path(
    pdb_dir: Path, row: pd.Series, primary: str, fallback: str | None,
) -> Path | None:
    """Substitute row fields into a filename pattern and return the resolved path.

    Pattern syntax: standard ``str.format`` with one extra convenience key,
    ``{pdb_code_lower}`` (lowercase ``pdb_code``).  If primary doesn't exist
    on disk and ``fallback`` is set, try it too.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# SaProt model
# ─────────────────────────────────────────────────────────────────────────────
def load_saprot(device: torch.device, model_id: str = SAPROT_MODEL):
    from transformers import AutoModel, AutoTokenizer
    log = setup_logging()
    log.info("loading SaProt: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    log.info("SaProt loaded (%d params, hidden_size=%d)",
             sum(p.numel() for p in model.parameters()),
             model.config.hidden_size)
    return tokenizer, model


@torch.no_grad()
def get_residue_embeddings_saprot(
    aa_seq: str, tdi_seq: str, tokenizer, model, device: torch.device,
    max_len: int = 1022,
) -> torch.Tensor:
    """Per-residue SaProt embeddings of shape (seq_len, hidden_size).

    SaProt's tokenizer expects the sequence as `<aa1><tdi1><aa2><tdi2>...`
    with whitespace separating tokens — i.e. `"Aa Cc Dd ..."` style.
    """
    assert len(aa_seq) == len(tdi_seq), "aa/3di length mismatch"
    if len(aa_seq) > max_len:
        aa_seq = aa_seq[:max_len]
        tdi_seq = tdi_seq[:max_len]
    combined = " ".join(f"{a}{d}" for a, d in zip(aa_seq, tdi_seq))
    inputs = tokenizer(combined, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    return outputs.last_hidden_state[0, 1:-1, :].cpu()  # strip CLS/EOS


def build_mutation_feature(
    residue_embeddings: torch.Tensor, mut_idx: int,
    wt_aa_3: str, mut_aa_3: str, window_size: int = 5,
) -> torch.Tensor:
    seq_len = residue_embeddings.shape[0]
    h_i = residue_embeddings[mut_idx]
    win_start = max(0, mut_idx - window_size)
    win_end = min(seq_len, mut_idx + window_size + 1)
    h_window = residue_embeddings[win_start:win_end].mean(dim=0)
    wt_oh = aa_one_hot(wt_aa_3)
    mut_oh = aa_one_hot(mut_aa_3)
    return torch.cat([h_i, h_window, wt_oh, mut_oh])


def split_by_protein(
    df: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int,
) -> pd.DataFrame:
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


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--metadata-csv", required=True, type=Path,
                   help="T2837/S669-format metadata CSV (pdb_code, sequence, "
                        "position, wtAA, mutAA, ddG, ...)")
    p.add_argument("--pdb-dir", required=True, type=Path,
                   help="Directory containing per-protein PDB files")
    p.add_argument("--pdb-pattern", required=True, type=str,
                   help="Filename pattern with {field} placeholders, e.g. "
                        "'AF-{uniprot_id}.pdb' or '{pdb_code}.pdb'.  Available "
                        "extra placeholder: {pdb_code_lower}.")
    p.add_argument("--pdb-pattern-fallback", type=str, default=None,
                   help="Optional secondary filename pattern if --pdb-pattern misses.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--metadata-out", type=Path, default=None)
    p.add_argument("--window-size", type=int, default=5)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--saprot-model", type=str, default=SAPROT_MODEL)
    p.add_argument("--max-len", type=int, default=1022)
    args = p.parse_args()

    log = setup_logging()
    set_seed(args.seed)
    check_foldseek_available()

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

    # 1. Load metadata
    log.info("loading metadata: %s", args.metadata_csv)
    df = pd.read_csv(args.metadata_csv)
    log.info("loaded %d rows", len(df))

    # 2. Resolve mutation indices against per-row sequence (same as script 01)
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

    # 3. Assign splits if not already provided
    if "split" not in df.columns or df["split"].isna().any():
        df = split_by_protein(df, args.val_fraction, args.test_fraction, args.seed)
    df["split"] = df["split"].astype(str).str.lower()
    for split in ("train", "val", "test"):
        n = (df["split"] == split).sum()
        log.info("  %s: %d mutations", split, n)

    # 4. For each unique protein, resolve its PDB and run FoldSeek
    unique = df.drop_duplicates(subset=["pdb_code"]).reset_index(drop=True)
    log.info("extracting 3Di tokens for %d unique proteins via foldseek...",
             len(unique))

    aa3di_cache: dict[str, tuple[str, str]] = {}    # pdb_code → (aa, 3di)
    n_no_pdb = n_3di_fail = 0
    for _, row in tqdm(unique.iterrows(), total=len(unique),
                       desc="foldseek 3Di", unit="prot"):
        pdb_path = resolve_pdb_path(
            args.pdb_dir, row, args.pdb_pattern, args.pdb_pattern_fallback,
        )
        if pdb_path is None:
            n_no_pdb += 1; continue
        result = extract_3di(pdb_path, log)
        if result is None:
            n_3di_fail += 1; continue
        aa3di_cache[str(row["pdb_code"])] = result
    log.info("3Di extraction: %d ok, %d missing PDB, %d foldseek failed",
             len(aa3di_cache), n_no_pdb, n_3di_fail)

    # 5. Drop rows whose protein lost its 3Di
    keep = df["pdb_code"].astype(str).isin(aa3di_cache)
    n_drop = int((~keep).sum())
    if n_drop > 0:
        log.warning("dropping %d rows whose protein has no 3Di", n_drop)
    df = df[keep].copy().reset_index(drop=True)

    # 6. Re-resolve mut_idx against the PDB-derived AA string.  The metadata's
    #    `mut_idx` was found against the metadata's per-row `sequence` string,
    #    but the PDB-derived AA from FoldSeek may have different numbering
    #    (signal peptides cleaved, chain offsets, missing termini).  Sampling
    #    SaProt embeddings at the metadata index would land at the wrong
    #    residue for any protein where those two sequences disagree.
    log.info("re-resolving mut_idx against PDB-derived AA strings...")
    new_idx, n_aligned, n_unaligned = [], 0, 0
    for _, row in df.iterrows():
        pdb_aa, _ = aa3di_cache[str(row["pdb_code"])]
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
    n_pdb_seq_mismatch = 0  # always 0 by construction now
    if n_pdb_seq_mismatch:
        log.warning(
            "%d rows have a wtAA / PDB-AA mismatch at mut_idx — these will use "
            "PDB-numbering best-effort and may give noisy embeddings.  This is "
            "expected to be small if the metadata sequence and the PDB sequence "
            "agree.  If this number is large, check that the PDB chain matches "
            "the metadata sequence.", n_pdb_seq_mismatch,
        )

    # 7. Run SaProt on each unique (aa, 3di) pair
    tokenizer, model = load_saprot(device, args.saprot_model)
    embed_dim = int(model.config.hidden_size)

    residue_cache: dict[str, torch.Tensor] = {}
    for pdb_code, (aa, tdi) in tqdm(aa3di_cache.items(), desc="SaProt embeddings", unit="seq"):
        residue_cache[pdb_code] = get_residue_embeddings_saprot(
            aa, tdi, tokenizer, model, device, max_len=args.max_len,
        )

    # 8. Build mutation-aware features per row
    log.info("building mutation-aware features (window=%d)...", args.window_size)
    features: list[torch.Tensor] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="mutation features", unit="mut"):
        residue_embs = residue_cache[str(row["pdb_code"])]
        mut_idx = int(row["mut_idx"])
        if mut_idx >= residue_embs.shape[0]:
            mut_idx = residue_embs.shape[0] - 1
        feat = build_mutation_feature(
            residue_embs, mut_idx, row["wtAA"], row["mutAA"], args.window_size,
        )
        features.append(feat)

    X_all = torch.stack(features, dim=0)  # (N, 2*embed_dim + 40)
    y_all = torch.tensor(df["ddG"].values, dtype=torch.float32)
    splits_col = df["split"].values
    log.info("feature tensor: %s", tuple(X_all.shape))

    # 9. Split into train/val/test
    splits: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for split in ("train", "val", "test"):
        mask = splits_col == split
        splits[split] = (X_all[mask], y_all[mask])
        log.info("  %s: X=%s, y=%s", split,
                 tuple(splits[split][0].shape), tuple(splits[split][1].shape))

    # 10. Save
    feat_dim = int(X_all.shape[-1])
    log.info("saving to %s", args.out)
    save_cached_embeddings(
        args.out,
        splits,
        meta={
            "source": str(args.metadata_csv),
            "backbone": f"SaProt ({args.saprot_model}) mutation-aware",
            "embed_dim": feat_dim,
            "components": (
                f"h_site({embed_dim}) + h_window({embed_dim}) "
                f"+ wt_onehot(20) + mut_onehot(20)"
            ),
            "window_size": args.window_size,
            "n_unique_proteins": len(residue_cache),
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
