#!/usr/bin/env python3
"""
generate_ablation_embeddings.py
===============================
Generate embedding caches for UAPP ablation Block A.

Existing caches:
  A0  cache/t2837_embeddings.pt      — global mean-pooled 320-d
  A3  cache/t2837_embeddings_v2.pt   — site(320)+window(320)+AA(40) = 680-d

This script creates:
  A1  cache/t2837_embeddings_A1.pt   — site-only 320-d  (residue at mut_idx)
  A2  cache/t2837_embeddings_A2.pt   — site+window 640-d (residue + ±3 window)
  A4  cache/t2837_embeddings_A4.pt   — AA one-hot only 40-d (wtAA 20 + mutAA 20)

Usage:
  python generate_ablation_embeddings.py [--model facebook/esm2_t6_8M_UR50D]
                                         [--window 3]
                                         [--device cpu]
                                         [--cache-dir cache]
                                         [--metadata cache/t2837_metadata.csv]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

# Use the project's own save function so the format matches load_cached_embeddings
sys.path.insert(0, ".")
from uapp.data import save_cached_embeddings, load_cached_embeddings

# ── Amino-acid encoding ────────────────────────────────────────────────
AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}


def aa_onehot(aa: str) -> torch.Tensor:
    """20-d one-hot for a single amino acid (zeros if unknown)."""
    vec = torch.zeros(20)
    if aa in AA_TO_IDX:
        vec[AA_TO_IDX[aa]] = 1.0
    return vec


def aa_pair_encoding(wt: str, mut: str) -> torch.Tensor:
    """40-d vector = concat(wt_onehot, mut_onehot)."""
    return torch.cat([aa_onehot(wt), aa_onehot(mut)])


# ── ESM-2 helpers ──────────────────────────────────────────────────────

def load_esm2(model_name: str, device: str):
    """Load ESM-2 model + tokenizer, return (model, tokenizer)."""
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    return model, tokenizer


@torch.no_grad()
def get_residue_embeddings(model, tokenizer, sequence: str, device: str) -> torch.Tensor:
    """
    Return per-residue embeddings (L×D) from ESM-2 last hidden state.
    Strips [CLS] and [EOS] tokens.
    """
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(device)
    out = model(**inputs).last_hidden_state[0]  # (seq_len+2, D)
    # Strip special tokens: ESM-2 adds <cls> at 0 and <eos> at end
    return out[1:-1].cpu()  # (L, D)


def extract_site_embedding(residue_embs: torch.Tensor, mut_idx: int) -> torch.Tensor:
    """320-d embedding at the mutation site."""
    L = residue_embs.size(0)
    idx = min(max(mut_idx, 0), L - 1)
    return residue_embs[idx]


def extract_window_embedding(residue_embs: torch.Tensor, mut_idx: int, window: int) -> torch.Tensor:
    """320-d mean-pooled embedding over [mut_idx-window, mut_idx+window]."""
    L = residue_embs.size(0)
    start = max(0, mut_idx - window)
    end = min(L, mut_idx + window + 1)
    return residue_embs[start:end].mean(dim=0)


# ── Cache builder ──────────────────────────────────────────────────────

def build_split_cache(df_split: pd.DataFrame, model, tokenizer,
                      device: str, window: int, variant: str) -> tuple:
    """
    Build (X, y) tensors for one data split.

    variant: 'A1' | 'A2' | 'A4'
    """
    X_list, y_list = [], []
    # Group by sequence to avoid redundant forward passes
    seq_cache = {}

    for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"  {variant}"):
        seq = row["sequence"]
        mut_idx = int(row["mut_idx"])
        wt_aa = row["wtAA"]
        mut_aa = row["mutAA"]
        ddG = float(row["ddG"])

        # Get per-residue embeddings (cached per unique sequence)
        if variant != "A4":
            if seq not in seq_cache:
                seq_cache[seq] = get_residue_embeddings(model, tokenizer, seq, device)
            residue_embs = seq_cache[seq]

        if variant == "A1":
            x = extract_site_embedding(residue_embs, mut_idx)           # 320-d
        elif variant == "A2":
            site = extract_site_embedding(residue_embs, mut_idx)        # 320-d
            win = extract_window_embedding(residue_embs, mut_idx, window)  # 320-d
            x = torch.cat([site, win])                                  # 640-d
        elif variant == "A4":
            x = aa_pair_encoding(wt_aa, mut_aa)                        # 40-d
        else:
            raise ValueError(f"Unknown variant: {variant}")

        X_list.append(x)
        y_list.append(ddG)

    X = torch.stack(X_list)
    y = torch.tensor(y_list, dtype=torch.float32)
    return X, y


def build_and_save(df: pd.DataFrame, model, tokenizer,
                   device: str, window: int, variant: str, cache_dir: Path):
    """Build cache for all splits and save using uapp.data.save_cached_embeddings."""
    splits = {}
    for split_name in ["train", "val", "test"]:
        df_split = df[df["split"] == split_name].reset_index(drop=True)
        if len(df_split) == 0:
            print(f"  WARNING: no rows for split '{split_name}', skipping")
            continue
        X, y = build_split_cache(df_split, model, tokenizer, device, window, variant)
        splits[split_name] = (X, y)
        print(f"  {split_name}: X={X.shape}, y={y.shape}")

    out_path = cache_dir / f"t2837_embeddings_{variant}.pt"

    # Use the library's save function so the on-disk format is:
    #   { "train": {"X": Tensor, "y": Tensor}, ..., "meta": {...} }
    # which is exactly what load_cached_embeddings expects.
    save_cached_embeddings(
        out_path,
        splits,
        meta={"variant": variant, "window": window},
    )
    print(f"  ✓ Saved → {out_path}\n")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate ablation embedding caches")
    parser.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--metadata", default="cache/t2837_metadata.csv")
    parser.add_argument("--variants", nargs="+", default=["A1", "A2", "A4"],
                        choices=["A1", "A2", "A4"])
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True)

    df = pd.read_csv(args.metadata)
    print(f"Loaded metadata: {len(df)} mutations, splits: {df['split'].value_counts().to_dict()}")

    # Validate required columns
    required = ["sequence", "mut_idx", "wtAA", "mutAA", "ddG", "split"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: metadata missing columns: {missing}")

    # Only load ESM-2 if we need structural embeddings (A1 or A2)
    need_esm = any(v in args.variants for v in ["A1", "A2"])
    if need_esm:
        model, tokenizer = load_esm2(args.model, args.device)
    else:
        model, tokenizer = None, None

    for variant in args.variants:
        print(f"\n{'='*60}")
        print(f"Building {variant} cache ...")
        print(f"{'='*60}")
        build_and_save(df, model, tokenizer, args.device, args.window, variant, cache_dir)

    # ── Summary ──
    print("\n" + "="*60)
    print("Ablation cache summary:")
    print("="*60)
    cache_files = {
        "A0": "t2837_embeddings.pt",
        "A1": "t2837_embeddings_A1.pt",
        "A2": "t2837_embeddings_A2.pt",
        "A3": "t2837_embeddings_v2.pt",
        "A4": "t2837_embeddings_A4.pt",
    }
    for tag, fname in cache_files.items():
        p = cache_dir / fname
        if p.exists():
            try:
                splits, meta = load_cached_embeddings(p)
                dim = meta.get("d", "?")
                n = sum(X.shape[0] for X, y in splits.values())
                print(f"  {tag}: {fname:40s}  dim={dim}  samples={n}")
            except Exception as e:
                print(f"  {tag}: {fname:40s}  [load error: {e}]")
        else:
            print(f"  {tag}: {fname:40s}  [not found]")

    print("\nDone! You can now run the ablation notebook.")


if __name__ == "__main__":
    main()
