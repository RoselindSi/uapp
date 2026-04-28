"""Filter an embedding+bio-feature cache to the row subset of a reference cache.

Why
===
§12 (SaProt) lost 8 T2837 proteins because their `uniprot_id` had no
AlphaFold-DB model — leaving SaProt with n=137 test rows vs ESM2's
n=170.  Comparing SaProt's RMSE on the 137-row test against ESM2's
RMSE on the 170-row test is sample-shifted, not a clean encoder
comparison.  This script produces a row-aligned ESM2 cache so the same
137 rows can be evaluated under both encoders.

What it does
============
1. Reads a *reference* metadata CSV (e.g. `cache/t2837_metadata_saprot.csv`)
   and its accompanying embedding cache + bio features.
2. Reads a *source* metadata CSV (e.g. `cache/t2837_metadata.csv`) and
   its caches.
3. For each split, computes the row indices in source whose
   (pdb_code, position, wtAA, mutAA) key is also in reference's
   subset for that split.
4. Writes filtered source caches (`<out_emb>` and `<out_bio>`) with row
   counts that match the reference, *split-by-split*.

The filtered caches are drop-in replacements: hand them to
``scripts/18_evaluate_on_s669.py`` exactly like the originals and the
T2837 sanity row will now be on the same 137 rows as the SaProt run.

Usage
-----
    python scripts/21_align_cache_to_reference.py \\
        --reference-metadata cache/t2837_metadata_saprot.csv \\
        --source-metadata    cache/t2837_metadata.csv \\
        --source-emb         cache/t2837_embeddings_v2_650m.pt \\
        --source-bio         cache/t2837_bio_features_650m_extended.pt \\
        --out-emb            cache/t2837_embeddings_v2_650m_saprot_aligned.pt \\
        --out-bio            cache/t2837_bio_features_650m_extended_saprot_aligned.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from uapp.data import load_cached_embeddings, save_cached_embeddings
from uapp.utils import setup_logging


KEY_COLS = ["pdb_code", "position", "wtAA", "mutAA"]


def make_key(df: pd.DataFrame) -> pd.Series:
    return df[KEY_COLS].astype(str).agg("|".join, axis=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference-metadata", required=True, type=Path)
    p.add_argument("--source-metadata",    required=True, type=Path)
    p.add_argument("--source-emb",         required=True, type=Path)
    p.add_argument("--source-bio",         required=True, type=Path)
    p.add_argument("--out-emb",            required=True, type=Path)
    p.add_argument("--out-bio",            required=True, type=Path)
    p.add_argument("--log-level",          type=str, default="INFO")
    args = p.parse_args()

    log = setup_logging(args.log_level)

    ref_md = pd.read_csv(args.reference_metadata)
    src_md = pd.read_csv(args.source_metadata)
    ref_md["split"] = ref_md["split"].astype(str).str.lower()
    src_md["split"] = src_md["split"].astype(str).str.lower()
    ref_md["key"] = make_key(ref_md)
    src_md["key"] = make_key(src_md)

    src_splits, src_meta = load_cached_embeddings(args.source_emb)
    src_bio = torch.load(args.source_bio, map_location="cpu", weights_only=False)

    new_splits: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    new_bio: dict = {"meta": {**src_bio["meta"],
                              "aligned_to": str(args.reference_metadata)}}

    for split in ("train", "val", "test"):
        ref_keys = set(ref_md[ref_md["split"] == split]["key"])
        src_rows = src_md[src_md["split"] == split].reset_index(drop=True)
        keep_mask = src_rows["key"].isin(ref_keys).to_numpy()
        n_ref = len(ref_keys)
        n_src = len(src_rows)
        n_keep = int(keep_mask.sum())
        if n_keep != n_ref:
            log.warning(
                "split %s: %d reference rows but %d source rows match — "
                "%d reference rows have no source counterpart",
                split, n_ref, n_keep, n_ref - n_keep,
            )
        log.info("split %s: src %d → keep %d (ref has %d)",
                 split, n_src, n_keep, n_ref)

        keep_idx = torch.from_numpy(keep_mask).nonzero(as_tuple=True)[0]
        Xs, ys = src_splits[split]
        new_splits[split] = (Xs[keep_idx], ys[keep_idx])
        new_bio[split] = {"feats": src_bio[split]["feats"][keep_idx]}

    args.out_emb.parent.mkdir(parents=True, exist_ok=True)
    save_cached_embeddings(args.out_emb, new_splits,
                           meta={**src_meta,
                                 "aligned_to": str(args.reference_metadata)})
    torch.save(new_bio, args.out_bio)

    log.info("Saved %s   (sizes: %s)",
             args.out_emb,
             {k: int(v[0].shape[0]) for k, v in new_splits.items()})
    log.info("Saved %s", args.out_bio)


if __name__ == "__main__":
    main()
