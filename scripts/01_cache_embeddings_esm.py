"""Cache ESM-2 embeddings for T2837 mutations.

Uses the smallest ESM-2 model (esm2_t6_8M_UR50D, ~33MB download) to
extract per-residue embeddings from protein sequences, then mean-pools
to get a single (320,)-dim vector per protein. These cached embeddings
replace the Stability Oracle graph-transformer embeddings — the
downstream heads, training, and evaluation are identical either way.

Usage
-----
    python scripts/01_cache_embeddings_esm.py \
        --t2837-csv StabilityOracle/data/datasets/T2837.csv \
        --out cache/t2837_embeddings.pt \
        --val-fraction 0.15 \
        --seed 42

Expected runtime: 5-15 minutes on CPU (M1/M2 Mac).
Output file: ~3-5 MB.
"""
from __future__ import annotations

import argparse
import hashlib
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

# ESM-2 model name — the smallest variant, runs fine on CPU
ESM_MODEL = "facebook/esm2_t6_8M_UR50D"
EMBED_DIM = 320  # output dim of this model


def load_esm2(device: torch.device):
    """Load ESM-2 tokenizer and model, frozen."""
    from transformers import EsmModel, EsmTokenizer

    log = setup_logging()
    log.info("loading ESM-2 model: %s", ESM_MODEL)
    tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL)
    model = EsmModel.from_pretrained(ESM_MODEL)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    log.info("ESM-2 loaded (%d params, device=%s)",
             sum(p.numel() for p in model.parameters()), device)
    return tokenizer, model


@torch.no_grad()
def embed_sequence(
    seq: str,
    tokenizer,
    model,
    device: torch.device,
    max_len: int = 1022,  # ESM-2 max input length
) -> torch.Tensor:
    """Extract mean-pooled ESM-2 embedding for one protein sequence.

    Returns a (320,) tensor on CPU.
    """
    # Truncate very long sequences (ESM-2 limit is 1022 tokens)
    if len(seq) > max_len:
        seq = seq[:max_len]

    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)

    # outputs.last_hidden_state: (1, seq_len+2, 320)
    # Strip the [CLS] and [EOS] tokens, then mean-pool over residues
    hidden = outputs.last_hidden_state[0, 1:-1, :]  # (seq_len, 320)
    h_G = hidden.mean(dim=0)  # (320,)
    return h_G.cpu()


def embed_sequence_at_mutation(
    seq: str,
    position: int,
    wt_aa: str,
    mut_aa: str,
    tokenizer,
    model,
    device: torch.device,
) -> torch.Tensor:
    """Embed the wild-type sequence via ESM-2.

    We use the wild-type sequence (not mutant) because Stability Oracle
    also uses only the wild-type structure. The mutation information
    (from_aa, to_aa) is captured in the head's input via the ddG target,
    not in the embedding itself.

    For a more mutation-aware embedding, you could embed both wt and
    mutant sequences and concatenate or subtract — but that's a future
    extension, not the baseline.
    """
    return embed_sequence(seq, tokenizer, model, device)


def split_by_dataset_column(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Assign train/val/test splits based on the 'dataset' column.

    T2837.csv has a 'dataset' column with values like 's669', 't2226',
    'ssym', 'p53', 'myoglobin'. Following Stability Oracle's convention:
    - All rows are part of the T2837 TEST set (this is their held-out eval)
    - For our project, we need train/val/test, so we split T2837 itself

    Strategy: split by protein (pdb_code) to avoid data leakage.
    80% of proteins → train, 5% → val, 15% → test.
    """
    rng = random.Random(seed)
    pdb_codes = df["pdb_code"].unique().tolist()
    rng.shuffle(pdb_codes)

    n_test = max(1, int(len(pdb_codes) * 0.15))
    n_val = max(1, int(len(pdb_codes) * val_fraction))
    n_train = len(pdb_codes) - n_test - n_val

    test_pdbs = set(pdb_codes[:n_test])
    val_pdbs = set(pdb_codes[n_test:n_test + n_val])
    train_pdbs = set(pdb_codes[n_test + n_val:])

    def assign(row):
        pdb = row["pdb_code"]
        if pdb in test_pdbs:
            return "test"
        if pdb in val_pdbs:
            return "val"
        return "train"

    df = df.copy()
    df["split"] = df.apply(assign, axis=1)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t2837-csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        help="'cpu', 'cuda', 'mps', or 'auto'")
    args = parser.parse_args()

    log = setup_logging()
    set_seed(args.seed)

    # Device selection
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

    # 1. Load T2837 CSV
    log.info("loading T2837 from %s", args.t2837_csv)
    df = pd.read_csv(args.t2837_csv)
    log.info("loaded %d mutations", len(df))

    # 2. Assign splits (by protein to avoid leakage)
    df = split_by_dataset_column(df, args.val_fraction, args.seed)
    for split in ("train", "val", "test"):
        n = (df["split"] == split).sum()
        n_pdbs = df[df["split"] == split]["pdb_code"].nunique()
        log.info("  %s: %d mutations from %d proteins", split, n, n_pdbs)

    # 3. Load ESM-2
    tokenizer, model = load_esm2(device)

    # 4. Embed each unique sequence ONCE, then map to mutations
    # Many mutations share the same wild-type sequence, so we cache by
    # (pdb_code, sequence) to avoid redundant forward passes.
    unique_seqs = df.drop_duplicates(subset=["pdb_code", "sequence"])[
        ["pdb_code", "sequence"]
    ].reset_index(drop=True)
    log.info("embedding %d unique protein sequences", len(unique_seqs))

    seq_embeddings: dict[str, torch.Tensor] = {}
    for _, row in tqdm(unique_seqs.iterrows(), total=len(unique_seqs),
                       desc="ESM-2 embedding", unit="seq"):
        pdb = row["pdb_code"]
        seq = row["sequence"]
        h = embed_sequence(seq, tokenizer, model, device)
        seq_embeddings[pdb] = h

    # 5. Build (X, y) tensors per split
    splits: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for split in ("train", "val", "test"):
        mask = df["split"] == split
        sub = df[mask]
        X_list = [seq_embeddings[pdb] for pdb in sub["pdb_code"]]
        X = torch.stack(X_list, dim=0)  # (N, 320)
        y = torch.tensor(sub["ddG"].values, dtype=torch.float32)
        splits[split] = (X, y)
        log.info("  %s: X=%s, y=%s", split, tuple(X.shape), tuple(y.shape))

    # 6. Save
    log.info("saving cached embeddings to %s", args.out)
    save_cached_embeddings(
        args.out,
        splits,
        meta={
            "source": "T2837",
            "backbone": f"ESM-2 ({ESM_MODEL})",
            "embed_dim": EMBED_DIM,
            "n_unique_sequences": len(unique_seqs),
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "split_strategy": "by_pdb_code",
        },
    )
    log.info("done. cache saved to %s (%.1f MB)",
             args.out, args.out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
