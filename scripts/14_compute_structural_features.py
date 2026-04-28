"""Compute DSSP + pLDDT structural features from AlphaFold DB structures.

Why
===
The Day-2 sequence-derived structural proxies (Chou-Fasman propensity,
local entropy, etc.) added a small but inconclusive lift over D3 in
K-fold CV (Δ +0.011, p=0.21).  They *approximate* what DSSP gives on real
structures.  This script produces the real thing.

We use AlphaFold DB structures (one per uniprot_id) for two reasons:
  1. One download per protein covers BOTH DSSP (real SS / ASA / φ / ψ)
     AND pLDDT (per-residue confidence stored in the B-factor field).
  2. AF predicts every residue, so unlike experimental PDBs there are
     no missing-residue gaps to align around.

Features extracted per mutation (8 new dims):
    ss_helix         (1 if DSSP SS ∈ {H, G, I}, else 0)
    ss_sheet         (1 if DSSP SS ∈ {E, B}, else 0)
    rsa_dssp         (DSSP-computed relative SASA, 0..1)
    phi              (backbone phi angle, degrees)
    psi              (backbone psi angle, degrees)
    plddt            (AlphaFold per-residue confidence, 0..100)
    local_helix_frac (fraction of helix in ±5 window)
    local_sheet_frac (fraction of sheet in ±5 window)

These are appended to an existing extended bio-features tensor (k=13)
to produce a k=21 file usable as ablation D6 in scripts 07/10/11.

Requirements
============
- biopython          ``pip install biopython``
- DSSP binary        ``apt-get install -y dssp`` (Linux/Colab) or
                     ``brew install dssp``       (macOS).
                     Newer name is ``mkdssp`` — script auto-detects.
- Internet access (one-time AF DB download per uniprot_id)

Inputs
------
- Existing extended bio features file  (k=13, output of script 06 with
  --include-extended)
- T2837 metadata CSV with `uniprot_id`, `position`, `wtAA`, `mut_idx`,
  `split` columns
- Embedding cache (used only for row-count alignment validation)

Outputs (under --out)
=====================
- ``t2837_dssp_plddt_features.pt``   k=21 bio-feature tensor (extended ⊕ DSSP)
- ``af_pdbs/``                        cache of downloaded AlphaFold PDB files
- ``failed_uniprot_ids.csv``          uniprot_ids the AF download / DSSP failed
                                      on (their mutations get NaN-filled and
                                      excluded from standardisation stats)
- ``coverage_report.json``            fraction of mutations with valid features

Usage
-----
    python scripts/14_compute_structural_features.py \\
        --metadata-csv     cache/t2837_metadata.csv \\
        --embeddings       cache/t2837_embeddings_v2_650m.pt \\
        --extended-bio     cache/t2837_bio_features_650m_extended.pt \\
        --out              cache/t2837_bio_features_650m_dssp.pt \\
        --pdb-cache        cache/af_pdbs/
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uapp.data import load_cached_embeddings
from uapp.utils import setup_logging


# ─────────────────────────────────────────────────────────────────────────────
# DSSP secondary-structure → 3-class
# ─────────────────────────────────────────────────────────────────────────────
HELIX_SS = set("HGI")    # alpha, 3-10, pi
SHEET_SS = set("EB")     # extended, beta-bridge


def _classify_ss(ss: str) -> tuple[float, float]:
    """Return (is_helix, is_sheet) one-hot for a DSSP single-letter code."""
    return (float(ss in HELIX_SS), float(ss in SHEET_SS))


# ─────────────────────────────────────────────────────────────────────────────
# AlphaFold DB download
# ─────────────────────────────────────────────────────────────────────────────
# We don't hard-code a model version (v4/v5/v6 etc.) because AF DB rotates
# them — direct construction "AF-{uniprot}-F1-model_v4.pdb" goes 404 once
# the canonical version moves on.  Instead we ask the AF DB JSON API for
# the canonical PDB URL per accession, which always points at the live model.
AF_API = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"


def get_af_pdb_url(uniprot_id: str, log, timeout: float = 10.0) -> str | None:
    """Resolve the canonical AF PDB URL for an accession via the public API."""
    api = AF_API.format(accession=uniprot_id)
    try:
        with urllib.request.urlopen(api, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        log.warning("AF API failed for %s: HTTP %s", uniprot_id, e.code)
        return None
    except Exception as e:
        log.warning("AF API failed for %s: %s", uniprot_id, e)
        return None
    if not data:
        log.warning("AF API returned empty list for %s (no prediction available)",
                    uniprot_id)
        return None
    return data[0].get("pdbUrl")


def download_af(uniprot_id: str, cache_dir: Path, log) -> Path | None:
    """Fetch AF model PDB into cache_dir; return path or None on failure.

    Uses the AF DB API to resolve the canonical PDB URL (handles version
    rotation; the direct .../files/AF-X-F1-model_v4.pdb pattern goes 404 once
    the API rotates the canonical version).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / f"AF-{uniprot_id}.pdb"
    if dst.exists() and dst.stat().st_size > 0:
        return dst

    pdb_url = get_af_pdb_url(uniprot_id, log)
    if pdb_url is None:
        return None
    try:
        urllib.request.urlretrieve(pdb_url, dst)
        return dst
    except urllib.error.HTTPError as e:
        log.warning("AF download failed for %s (%s): HTTP %s",
                    uniprot_id, pdb_url, e.code)
    except Exception as e:
        log.warning("AF download failed for %s (%s): %s",
                    uniprot_id, pdb_url, e)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Resolve DSSP binary name
# ─────────────────────────────────────────────────────────────────────────────
def find_dssp_binary() -> str | None:
    for name in ("mkdssp", "dssp"):
        if shutil.which(name) is not None:
            return name
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-protein DSSP + pLDDT extraction
# ─────────────────────────────────────────────────────────────────────────────
def _empty_per_residue(seq_len: int) -> dict[int, dict]:
    """Sentinel: residue-id keyed dict where every requested residue is missing."""
    return {}


def compute_per_residue_features(
    pdb_path: Path,
    dssp_binary: str,
    log,
) -> dict[int, dict] | None:
    """Run DSSP on an AF PDB and extract per-residue features keyed by PDB resseq.

    Returns dict[resseq] -> {ss, asa_rel, phi, psi, plddt, aa}.
    Returns None on parse / DSSP failure.
    """
    # Defer biopython imports so the script can at least --help on machines
    # without it installed.
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.DSSP import DSSP
    except ImportError as e:
        raise SystemExit(
            "biopython is required.  `pip install biopython`."
        ) from e

    try:
        structure = PDBParser(QUIET=True).get_structure(pdb_path.stem, str(pdb_path))
        model = next(structure.get_models())
        chain = next(iter(model))     # AF DB is always single-chain
        chain_id = chain.id

        # Pre-collect pLDDT per residue (B-factor of CA atom).  AF stores the
        # same pLDDT on every atom of a residue but CA is canonical.
        plddt = {}
        for res in chain:
            try:
                ca = res["CA"]
                plddt[res.id[1]] = float(ca.get_bfactor())
            except KeyError:
                continue

        # Run DSSP
        dssp = DSSP(model, str(pdb_path), dssp=dssp_binary)
    except Exception as e:
        log.warning("DSSP failed for %s: %s", pdb_path.name, e)
        return None

    out: dict[int, dict] = {}
    for key, val in dssp.property_dict.items():
        ch_id, res_id = key
        if ch_id != chain_id:
            continue
        het, resseq, icode = res_id
        if het != " " or icode != " ":
            continue   # skip HETATMs and insertions
        # DSSP value tuple: (idx, aa, ss, rel_asa, phi, psi, ...)
        idx, aa, ss, rel_asa, phi, psi = val[:6]
        out[int(resseq)] = {
            "aa":      aa,
            "ss":      str(ss),
            "asa_rel": float(rel_asa) if rel_asa != "NA" else float("nan"),
            "phi":     float(phi),
            "psi":     float(psi),
            "plddt":   plddt.get(int(resseq), float("nan")),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-mutation feature builder
# ─────────────────────────────────────────────────────────────────────────────
def per_mutation_struct_features(
    res_features: dict[int, dict],
    mut_position: int,            # 1-based PDB resseq
    half_window: int = 5,
) -> np.ndarray:
    """Return 8 features for one mutation.  NaN-filled if mutation residue missing."""
    nan8 = np.full(8, np.nan, dtype=np.float32)
    if mut_position not in res_features:
        return nan8
    r = res_features[mut_position]

    h, e = _classify_ss(r["ss"])
    rsa  = r["asa_rel"] if not np.isnan(r["asa_rel"]) else 0.0
    phi  = r["phi"] / 180.0     # normalise to roughly [-1, 1]
    psi  = r["psi"] / 180.0
    pl   = r["plddt"]
    if np.isnan(pl): pl = 50.0  # AF DB pLDDT typical range; fallback to mid

    # Local helix / sheet fractions in ±half_window
    lo = mut_position - half_window
    hi = mut_position + half_window + 1
    n_h = n_e = n_total = 0
    for p in range(lo, hi):
        if p in res_features:
            h_, e_ = _classify_ss(res_features[p]["ss"])
            n_h += int(h_); n_e += int(e_); n_total += 1
    local_h = n_h / max(n_total, 1)
    local_e = n_e / max(n_total, 1)

    return np.array([h, e, rsa, phi, psi, pl, local_h, local_e], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata-csv",  required=True, type=Path)
    p.add_argument("--embeddings",    required=True, type=Path,
                   help="Embedding cache, used only for row-count alignment check")
    p.add_argument("--extended-bio",  required=True, type=Path,
                   help="Existing bio_features file built with --include-extended (k=13)")
    p.add_argument("--out",           required=True, type=Path,
                   help="Output bio-features file with DSSP/pLDDT appended (k=21)")
    p.add_argument("--pdb-cache",     type=Path, default=Path("cache/af_pdbs"),
                   help="Local cache dir for downloaded AlphaFold PDBs")
    p.add_argument("--uniprot-col",   default="uniprot_id")
    p.add_argument("--position-col",  default="position",
                   help="Column with 1-based PDB residue position (NOT mut_idx)")
    p.add_argument("--half-window",   type=int, default=5)
    p.add_argument("--log-level",     default="INFO")
    args = p.parse_args()

    log = setup_logging(args.log_level)
    out_dir = args.out.parent; out_dir.mkdir(parents=True, exist_ok=True)

    # ── Sanity: DSSP installed? ──────────────────────────────────────────────
    dssp_bin = find_dssp_binary()
    if dssp_bin is None:
        raise SystemExit(
            "Neither `mkdssp` nor `dssp` found on PATH.  Install with:\n"
            "  Linux/Colab:  apt-get install -y dssp\n"
            "  macOS:        brew install dssp"
        )
    log.info("DSSP binary: %s", dssp_bin)

    # ── Load metadata + extended bio features + embedding cache ──────────────
    df = pd.read_csv(args.metadata_csv)
    df["__split"] = df["split"].astype(str).str.strip().str.lower()
    if args.uniprot_col not in df.columns:
        raise SystemExit(f"Metadata CSV missing column {args.uniprot_col!r}.")
    if args.position_col not in df.columns:
        raise SystemExit(f"Metadata CSV missing column {args.position_col!r}.")

    splits, _ = load_cached_embeddings(args.embeddings)
    cache_sizes = {k: int(v[0].shape[0]) for k, v in splits.items()}

    bio = torch.load(args.extended_bio, map_location="cpu", weights_only=False)
    if bio["meta"].get("k") != 13:
        raise SystemExit(
            f"--extended-bio file has k={bio['meta'].get('k')}; expected k=13.  "
            "Re-run scripts/06_build_bio_features.py with --include-extended."
        )

    # Per-split row counts must align with both the embedding cache AND the bio file
    grouped = {s: g.reset_index(drop=True) for s, g in df.groupby("__split")}
    for split in ("train", "val", "test"):
        n_emb = cache_sizes[split]
        n_bio = bio[split]["feats"].shape[0]
        n_md  = len(grouped[split])
        if not (n_emb == n_bio == n_md):
            raise SystemExit(
                f"Alignment failure on split {split!r}: "
                f"embeddings={n_emb} bio={n_bio} metadata={n_md}"
            )

    # ── Download AlphaFold structures (one per unique uniprot_id) ────────────
    unique_uniprots = sorted(set(df[args.uniprot_col].astype(str).str.strip()))
    log.info("Need AF structures for %d unique uniprot_ids", len(unique_uniprots))

    args.pdb_cache.mkdir(parents=True, exist_ok=True)
    pdb_paths: dict[str, Path] = {}
    failed_downloads: list[str] = []
    for i, uid in enumerate(unique_uniprots, 1):
        if not uid or uid.lower() in ("nan", "none"):
            failed_downloads.append(uid); continue
        path = download_af(uid, args.pdb_cache, log)
        if path is None:
            failed_downloads.append(uid)
        else:
            pdb_paths[uid] = path
        if i % 20 == 0 or i == len(unique_uniprots):
            log.info("  Downloaded %d/%d  (%d failures so far)",
                     i, len(unique_uniprots), len(failed_downloads))

    log.info("AF download complete: %d ok, %d failed",
             len(pdb_paths), len(failed_downloads))

    # ── Run DSSP per protein, cache per-residue features ─────────────────────
    log.info("Running DSSP on %d structures …", len(pdb_paths))
    per_protein_features: dict[str, dict[int, dict]] = {}
    failed_dssp: list[str] = []
    for i, (uid, path) in enumerate(pdb_paths.items(), 1):
        feats = compute_per_residue_features(path, dssp_bin, log)
        if feats is None:
            failed_dssp.append(uid); continue
        per_protein_features[uid] = feats
        if i % 20 == 0 or i == len(pdb_paths):
            log.info("  DSSP %d/%d  (%d failures so far)",
                     i, len(pdb_paths), len(failed_dssp))

    log.info("DSSP complete: %d ok, %d failed", len(per_protein_features), len(failed_dssp))

    # ── Compute per-mutation features for every row in metadata ──────────────
    n_total = len(df)
    raw_struct = np.full((n_total, 8), np.nan, dtype=np.float32)
    n_with_features = 0
    for i, row in df.iterrows():
        uid = str(row[args.uniprot_col]).strip()
        pos = int(row[args.position_col])
        feats = per_protein_features.get(uid)
        if feats is None: continue
        v = per_mutation_struct_features(feats, pos, half_window=args.half_window)
        if not np.all(np.isnan(v)):
            raw_struct[i] = v
            n_with_features += 1
    log.info("Per-mutation extraction: %d/%d (%.1f%%) mutations have DSSP features",
             n_with_features, n_total, 100 * n_with_features / max(n_total, 1))

    # ── Standardise on TRAIN-split rows that have valid features ─────────────
    train_mask = df["__split"].values == "train"
    train_rows_with_feats = ~np.isnan(raw_struct[train_mask]).any(axis=1)
    train_valid = raw_struct[train_mask][train_rows_with_feats]
    if len(train_valid) == 0:
        raise SystemExit("No train mutations have valid DSSP features — cannot standardise.")

    mu = train_valid.mean(axis=0)
    sd = train_valid.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)

    # NaN -> standardised zero (i.e., the train mean) so the σ MLP sees a
    # neutral signal for missing-structure samples.  Track which rows were
    # imputed so we can report coverage.
    imputed_mask = np.isnan(raw_struct).any(axis=1)
    raw_filled = np.where(imputed_mask[:, None], mu, raw_struct)
    standardised_struct = ((raw_filled - mu) / sd).astype(np.float32)

    log.info("Standardisation done.  %d (%.1f%%) mutations were NaN-imputed at train mean.",
             int(imputed_mask.sum()), 100 * imputed_mask.mean())

    # ── Concatenate to existing extended (k=13) bio features ─────────────────
    new_bio: dict = {}
    base_meta = bio["meta"]
    for split in ("train", "val", "test"):
        sub = grouped[split]
        # Indices into raw_struct for this split, in the same order as bio[split]["feats"]
        # (we built `df` and `grouped[split]` from the same DataFrame, so order matches)
        split_idx = sub.index.values   # original df index
        struct_for_split = standardised_struct[split_idx]
        base_for_split   = bio[split]["feats"].numpy()
        combined         = np.concatenate([base_for_split, struct_for_split], axis=1)
        new_bio[split] = {"feats": torch.from_numpy(combined.astype(np.float32))}

    new_feature_names = list(base_meta["feature_names"]) + [
        "ss_helix", "ss_sheet", "rsa_dssp", "phi_norm", "psi_norm",
        "plddt", "local_helix_frac", "local_sheet_frac",
    ]
    new_bio["meta"] = {
        "feature_names":      new_feature_names,
        "k":                  21,
        "include_indicators": base_meta.get("include_indicators", False),
        "include_extended":   True,
        "include_dssp_plddt": True,
        "source_metadata":    str(args.metadata_csv),
        "source_embeddings":  str(args.embeddings),
        "source_extended":    str(args.extended_bio),
        "dssp_mu":            mu.tolist(),
        "dssp_sd":            sd.tolist(),
        "n_total":            int(n_total),
        "n_with_features":    int(n_with_features),
        "n_imputed":          int(imputed_mask.sum()),
        "failed_uniprot_ids": failed_downloads + failed_dssp,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_bio, args.out)
    log.info("Saved %s  (k=21)", args.out)

    # ── Reports ──────────────────────────────────────────────────────────────
    if failed_downloads or failed_dssp:
        fail_df = pd.DataFrame({
            "uniprot_id": failed_downloads + failed_dssp,
            "stage": (["download"] * len(failed_downloads)
                      + ["dssp"] * len(failed_dssp)),
        })
        fail_path = out_dir / "failed_uniprot_ids.csv"
        fail_df.to_csv(fail_path, index=False)
        log.info("Saved failure list: %s", fail_path)

    coverage = {
        "n_total":           int(n_total),
        "n_with_features":   int(n_with_features),
        "n_imputed":         int(imputed_mask.sum()),
        "coverage_fraction": float(n_with_features / max(n_total, 1)),
        "failed_downloads":  len(failed_downloads),
        "failed_dssp":       len(failed_dssp),
    }
    (out_dir / "coverage_report.json").write_text(json.dumps(coverage, indent=2))
    log.info("Coverage: %.1f%%  (%d/%d mutations have real DSSP features)",
             100 * coverage["coverage_fraction"], n_with_features, n_total)

    print("\n" + "=" * 78)
    print(f"Saved DSSP+pLDDT features  (k=21)  to: {args.out}")
    print(f"Coverage:  {n_with_features}/{n_total}  ({100*n_with_features/n_total:.1f}%)")
    print(f"Run K-fold with the new ablation D6 (uses all 21 features):")
    print(f"  python scripts/11_kfold_cv_track_d.py \\")
    print(f"      --embeddings {args.embeddings} \\")
    print(f"      --bio-feats  {args.out} \\")
    print(f"      --out        outputs/cv_d5_vs_d6 \\")
    print(f"      --ablations D0 D5 D6 \\")
    print(f"      --folds 5 --seeds 0 1 2 3 4 \\")
    print(f"      --device cuda")
    print("=" * 78)


if __name__ == "__main__":
    main()
