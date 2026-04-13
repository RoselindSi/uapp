"""Data loading.

Two concerns:

1. **T2837 mutation list**: the raw CSV/TSV of mutations + ddG labels
   from the Stability Oracle release. We need protein IDs, mutation
   positions, and the ddG target.

2. **Cached embeddings**: after running the backbone once, we save a
   dict {split: (embeddings, targets)} to disk. All downstream training
   and evaluation loads from this cached tensor file, not the backbone.

The cached format is intentionally simple — a single torch .pt file:

    {
        "train": {"X": Tensor[N_train, d], "y": Tensor[N_train]},
        "val":   {"X": Tensor[N_val, d],   "y": Tensor[N_val]},
        "test":  {"X": Tensor[N_test, d],  "y": Tensor[N_test]},
        "meta":  {"d": int, "created_at": str, ...},
    }
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


# ---------------------------------------------------------------------------
# T2837 raw-data loading
# ---------------------------------------------------------------------------

@dataclass
class Mutation:
    """A single T2837 row.

    Attributes
    ----------
    protein_id : PDB ID or identifier used by Stability Oracle (e.g., '1ABC')
    chain : chain letter (usually 'A')
    position : residue number (int)
    wt_aa : wild-type amino acid, single letter
    mut_aa : mutant amino acid, single letter
    ddg : experimental delta-delta-G in kcal/mol
    split : 'train' | 'val' | 'test'
    """
    protein_id: str
    chain: str
    position: int
    wt_aa: str
    mut_aa: str
    ddg: float
    split: str


def load_t2837_mutations(csv_path: str | Path) -> list[Mutation]:
    """Load T2837 mutations from the Stability Oracle CSV release.

    The exact column names in the released CSV may differ slightly; this
    function tries a few common variants. If none match, it raises with
    a clear error message listing the columns it found.

    Parameters
    ----------
    csv_path : path to the T2837 mutations file

    Returns
    -------
    list of Mutation. Split assignment comes from a 'split' column if
    present; otherwise, the function assigns all rows to 'test' and
    leaves train/val splitting to the caller.
    """
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}

    def pick(*candidates: str) -> str | None:
        for c in candidates:
            if c.lower() in cols:
                return cols[c.lower()]
        return None

    pid_col = pick("protein_id", "pdb", "pdb_id", "protein")
    pos_col = pick("position", "resi", "residue_number")
    wt_col = pick("wt_aa", "wt", "wild_type")
    mut_col = pick("mut_aa", "mut", "mutant")
    ddg_col = pick("ddg", "ddG", "delta_delta_g")
    chain_col = pick("chain")
    split_col = pick("split", "set")

    missing = [
        name for name, c in [
            ("protein_id", pid_col),
            ("position", pos_col),
            ("wt_aa", wt_col),
            ("mut_aa", mut_col),
            ("ddg", ddg_col),
        ] if c is None
    ]
    if missing:
        raise ValueError(
            f"T2837 CSV is missing columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    mutations: list[Mutation] = []
    for _, row in df.iterrows():
        mutations.append(
            Mutation(
                protein_id=str(row[pid_col]),
                chain=str(row[chain_col]) if chain_col else "A",
                position=int(row[pos_col]),
                wt_aa=str(row[wt_col]).strip().upper(),
                mut_aa=str(row[mut_col]).strip().upper(),
                ddg=float(row[ddg_col]),
                split=str(row[split_col]).strip().lower() if split_col else "test",
            )
        )
    return mutations


# ---------------------------------------------------------------------------
# Cached-embedding dataset
# ---------------------------------------------------------------------------

def save_cached_embeddings(
    out_path: str | Path,
    splits: dict[str, tuple[torch.Tensor, torch.Tensor]],
    meta: dict | None = None,
) -> None:
    """Save a dict of {split: (X, y)} tensors to a .pt file."""
    payload = {
        split: {"X": X.detach().cpu(), "y": y.detach().cpu()}
        for split, (X, y) in splits.items()
    }
    if splits:
        first_X = next(iter(splits.values()))[0]
        d = int(first_X.shape[-1])
    else:
        d = 0
    payload["meta"] = {
        "d": d,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **(meta or {}),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def load_cached_embeddings(
    cache_path: str | Path,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict]:
    """Load a cached-embedding .pt file.

    Returns
    -------
    splits : dict[str, (X, y)]
    meta : dict
    """
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    meta = payload.pop("meta", {})
    splits = {k: (v["X"], v["y"]) for k, v in payload.items()}
    return splits, meta


def make_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Wrap (X, y) tensors in a DataLoader."""
    ds = TensorDataset(X.float(), y.float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
