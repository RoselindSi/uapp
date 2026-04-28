"""Cache mutation-aware ESM-2 embeddings for T2837.

Instead of mean-pooling the whole sequence into a single d-dim vector
(which loses all mutation-site information), this script constructs a
*mutation-specific* feature vector for each row in T2837:

    x = [ h_i, mean(h_{i-k:i+k}), e(wtAA), e(mutAA) ]

Where:
    h_i             = ESM-2 embedding at the mutation site  (d-dim)
    mean(h_window)  = mean-pooled local window around site  (d-dim)
    e(wtAA)         = one-hot encoding of wild-type AA      (20-d)
    e(mutAA)        = one-hot encoding of mutant AA         (20-d)

Total input dimension: 2·d + 40.

Backbone selection (``--esm-model``)
------------------------------------
+--------------------------------+--------+----------+-------------------------------+
| --esm-model                    |  d     | params   | use case                       |
+--------------------------------+--------+----------+-------------------------------+
| facebook/esm2_t6_8M_UR50D      |   320  |  8 M     | default; fast, weak signal     |
| facebook/esm2_t12_35M_UR50D    |   480  | 35 M     | medium                         |
| facebook/esm2_t30_150M_UR50D   |   640  | 150 M    | strong, ~4× slower than 8M     |
| facebook/esm2_t33_650M_UR50D   | 1280   | 650 M    | strongest open ESM2; needs GPU |
+--------------------------------+--------+----------+-------------------------------+

Usage
-----
    # Default (ESM2-8M, fast):
    python scripts/01_cache_embeddings_esm_v2.py \
        --t2837-csv StabilityOracle/data/datasets/T2837.csv \
        --out cache/t2837_embeddings_v2.pt \
        --seed 42

    # ESM2-650M backbone (recommended next step; needs GPU or several hours on CPU):
    python scripts/01_cache_embeddings_esm_v2.py \
        --t2837-csv StabilityOracle/data/datasets/T2837.csv \
        --out cache/t2837_embeddings_v2_650m.pt \
        --esm-model facebook/esm2_t33_650M_UR50D \
        --seed 42

Expected runtime: 5-15 min CPU for the 8M model, several hours for 650M
without a GPU.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from uapp.data import save_cached_embeddings
from uapp.utils import set_seed, setup_logging

DEFAULT_ESM_MODEL = "facebook/esm2_t6_8M_UR50D"

AA3to1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}
AA1_LIST = sorted(set(AA3to1.values()))  # 20 amino acids, sorted
AA1_TO_IDX = {aa: i for i, aa in enumerate(AA1_LIST)}


def aa_one_hot(aa_3letter: str) -> torch.Tensor:
    """Convert 3-letter AA code to a 20-d one-hot vector."""
    aa1 = AA3to1.get(aa_3letter, None)
    vec = torch.zeros(20)
    if aa1 is not None and aa1 in AA1_TO_IDX:
        vec[AA1_TO_IDX[aa1]] = 1.0
    return vec


def find_mutation_index(sequence: str, position: int, wt_aa_3: str) -> int | None:
    """Find the correct 0-based index in sequence for the mutation site.

    PDB residue numbers don't always map directly to sequence indices.
    Strategy: start at (position-1) and search outward for a match.
    """
    wt1 = AA3to1.get(wt_aa_3, '?')
    if wt1 == '?':
        return None

    # Try exact 1-indexed first
    idx = position - 1
    if 0 <= idx < len(sequence) and sequence[idx] == wt1:
        return idx

    # Search outward in a window of ±50
    for offset in range(1, 50):
        for sign in [-1, 1]:
            candidate = idx + sign * offset
            if 0 <= candidate < len(sequence) and sequence[candidate] == wt1:
                return candidate

    return None


def load_esm2(device: torch.device, esm_model: str = DEFAULT_ESM_MODEL):
    """Load ESM-2 tokenizer and model, frozen."""
    from transformers import EsmModel, EsmTokenizer
    log = setup_logging()
    log.info("loading ESM-2: %s", esm_model)
    tokenizer = EsmTokenizer.from_pretrained(esm_model)
    model = EsmModel.from_pretrained(esm_model)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    log.info("ESM-2 loaded (%d params, hidden_size=%d)",
             sum(p.numel() for p in model.parameters()),
             model.config.hidden_size)
    return tokenizer, model


@torch.no_grad()
def get_residue_embeddings(
    seq: str, tokenizer, model, device: torch.device, max_len: int = 1022,
) -> torch.Tensor:
    """Return per-residue ESM-2 embeddings of shape (seq_len, 320)."""
    if len(seq) > max_len:
        seq = seq[:max_len]
    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    # Strip [CLS] and [EOS] tokens
    return outputs.last_hidden_state[0, 1:-1, :].cpu()  # (seq_len, 320)


def build_mutation_feature(
    residue_embeddings: torch.Tensor,  # (seq_len, 320)
    mut_idx: int,
    wt_aa_3: str,
    mut_aa_3: str,
    window_size: int = 5,
) -> torch.Tensor:
    """Build the mutation-aware feature vector (680-d).

    Components:
        h_i          : embedding at mutation site (320)
        h_window     : mean of local window embeddings (320)
        e(wtAA)      : one-hot of wild-type AA (20)
        e(mutAA)     : one-hot of mutant AA (20)
    """
    seq_len = residue_embeddings.shape[0]

    # Site embedding
    h_i = residue_embeddings[mut_idx]  # (320,)

    # Local window (clamp to sequence boundaries)
    win_start = max(0, mut_idx - window_size)
    win_end = min(seq_len, mut_idx + window_size + 1)
    h_window = residue_embeddings[win_start:win_end].mean(dim=0)  # (320,)

    # AA one-hots
    wt_oh = aa_one_hot(wt_aa_3)    # (20,)
    mut_oh = aa_one_hot(mut_aa_3)  # (20,)

    return torch.cat([h_i, h_window, wt_oh, mut_oh])  # (680,)


def split_by_protein(
    df: pd.DataFrame, val_fraction: float, test_fraction: float, seed: int,
) -> pd.DataFrame:
    """Assign train/val/test by protein to avoid data leakage."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t2837-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--window-size", type=int, default=5,
                        help="Half-window size k for local context (default: 5)")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--esm-model", type=str, default=DEFAULT_ESM_MODEL,
                        help="HuggingFace ESM-2 model id (default: 8M).  "
                             "Use facebook/esm2_t33_650M_UR50D for the 650M backbone.")
    parser.add_argument("--metadata-out", type=Path, default=None,
                        help="Where to write the processed metadata CSV (with the "
                             "fold splits used to build this cache).  Defaults to "
                             "<out_stem>_metadata.csv next to --out, so different "
                             "datasets don't overwrite each other's metadata.")
    args = parser.parse_args()

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

    # 1. Load T2837
    log.info("loading T2837 from %s", args.t2837_csv)
    df = pd.read_csv(args.t2837_csv)
    log.info("loaded %d mutations", len(df))

    # 2. Find mutation indices and filter unmappable rows
    log.info("resolving mutation positions to sequence indices...")
    mut_indices = []
    for _, row in df.iterrows():
        idx = find_mutation_index(row["sequence"], row["position"], row["wtAA"])
        mut_indices.append(idx)
    df["mut_idx"] = mut_indices
    n_before = len(df)
    df = df[df["mut_idx"].notna()].copy()
    df["mut_idx"] = df["mut_idx"].astype(int)
    n_after = len(df)
    log.info("position resolution: %d/%d mapped (dropped %d unmappable)",
             n_after, n_before, n_before - n_after)

    # 3. Assign splits
    df = split_by_protein(df, args.val_fraction, args.test_fraction, args.seed)
    for split in ("train", "val", "test"):
        n = (df["split"] == split).sum()
        n_pdbs = df[df["split"] == split]["pdb_code"].nunique()
        log.info("  %s: %d mutations from %d proteins", split, n, n_pdbs)

    # 4. Load ESM-2
    tokenizer, model = load_esm2(device, args.esm_model)
    embed_dim = int(model.config.hidden_size)

    # 5. Cache per-residue embeddings per unique sequence
    unique_seqs = df.drop_duplicates(subset=["pdb_code", "sequence"])[
        ["pdb_code", "sequence"]
    ].reset_index(drop=True)
    log.info("extracting per-residue embeddings for %d unique sequences", len(unique_seqs))

    residue_cache: dict[str, torch.Tensor] = {}
    for _, row in tqdm(unique_seqs.iterrows(), total=len(unique_seqs),
                       desc="ESM-2 residue embeddings", unit="seq"):
        pdb = row["pdb_code"]
        seq = row["sequence"]
        residue_cache[pdb] = get_residue_embeddings(seq, tokenizer, model, device)

    # 6. Build mutation-aware features for every row
    log.info("building mutation-aware features (window=%d)...", args.window_size)
    features: list[torch.Tensor] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="mutation features", unit="mut"):
        pdb = row["pdb_code"]
        residue_embs = residue_cache[pdb]
        mut_idx = int(row["mut_idx"])

        # Clamp index to actual embedding length (truncated sequences)
        if mut_idx >= residue_embs.shape[0]:
            mut_idx = residue_embs.shape[0] - 1

        feat = build_mutation_feature(
            residue_embs, mut_idx, row["wtAA"], row["mutAA"], args.window_size
        )
        features.append(feat)

    X_all = torch.stack(features, dim=0)  # (N, 680)
    y_all = torch.tensor(df["ddG"].values, dtype=torch.float32)
    splits_col = df["split"].values
    log.info("feature tensor: %s", tuple(X_all.shape))

    # 7. Split into train/val/test
    splits: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for split in ("train", "val", "test"):
        mask = splits_col == split
        splits[split] = (X_all[mask], y_all[mask])
        log.info("  %s: X=%s, y=%s", split, tuple(splits[split][0].shape), tuple(splits[split][1].shape))

    # 8. Save
    feat_dim = int(X_all.shape[-1])
    log.info("saving to %s", args.out)
    save_cached_embeddings(
        args.out,
        splits,
        meta={
            "source": "T2837",
            "backbone": f"ESM-2 ({args.esm_model}) mutation-aware",
            "embed_dim": feat_dim,
            "components": (
                f"h_site({embed_dim}) + h_window({embed_dim}) "
                f"+ wt_onehot(20) + mut_onehot(20)"
            ),
            "window_size": args.window_size,
            "n_unique_sequences": len(unique_seqs),
            "n_mapped": n_after,
            "n_dropped": n_before - n_after,
            "seed": args.seed,
            "split_strategy": "by_pdb_code",
        },
    )
    log.info("done. %s (%.1f MB)", args.out, args.out.stat().st_size / 1e6)

    # Also save the metadata DataFrame for the notebook's interpretability analysis
    if args.metadata_out is not None:
        meta_csv_path = args.metadata_out
    else:
        meta_csv_path = args.out.parent / f"{args.out.stem}_metadata.csv"
    df.to_csv(meta_csv_path, index=False)
    log.info("saved mutation metadata to %s", meta_csv_path)


if __name__ == "__main__":
    main()
