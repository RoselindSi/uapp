"""Biophysical mutation features for the variance branch.

Given a (wt_aa, mut_aa, rsa) triple, produce a low-dimensional feature
vector that captures *what kind of substitution this is* — the biological
signal the σ branch is trying to use to estimate uncertainty.

Features (in order):
    0. RSA                          (relative solvent accessibility, 0–1)
    1. BLOSUM62 score               (substitution likelihood)
    2. Grantham distance            (composite physicochemical distance)
    3. Δ charge                     (mut_charge − wt_charge)
    4. Δ polarity                   (1 if polarity class flips, else 0)
    5. Δ hydrophobicity (KD scale)  (mut − wt, Kyte-Doolittle)
    6. Δ volume                     (mut − wt, Å³)

Indicators (optional extended set, ``include_indicators=True``):
    7. mut_is_proline
    8. mut_is_glycine
    9. mut_is_aromatic              (F, W, Y)

Use ``compute_features(...)`` for one mutation, ``batch_features(...)`` for
arrays, and ``feature_dim()`` / ``feature_names()`` to query the output layout.
"""
from __future__ import annotations

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Amino-acid order and per-AA scalar properties
# ─────────────────────────────────────────────────────────────────────────────
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
AA_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}

# Charge at physiological pH (default 0 if not listed).
_CHARGE = {
    "K": +1, "R": +1, "H": +1,
    "D": -1, "E": -1,
}

# Polarity class — polar (incl. acidic + basic) vs nonpolar.
_POLAR = set("DENQHKRSTY")

# Kyte–Doolittle hydrophobicity scale (more positive = more hydrophobic).
_HYDRO = {
    "I":  4.5, "V":  4.2, "L":  3.8, "F":  2.8, "C":  2.5,
    "M":  1.9, "A":  1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
    "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}

# Side-chain volume (Å³) — Zamyatnin 1972 averages.
_VOLUME = {
    "A":  88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G":  60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S":  89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# BLOSUM62 (Henikoff & Henikoff 1992).  Standard 20×20 substitution matrix,
# row-major order = AA_ORDER.
# ─────────────────────────────────────────────────────────────────────────────
_BLOSUM62 = np.array([
    # A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],  # A
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],  # R
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],  # N
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],  # D
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],  # C
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],  # Q
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],  # E
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],  # G
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],  # H
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],  # I
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],  # L
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],  # K
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],  # M
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],  # F
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],  # P
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],  # S
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],  # T
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],  # W
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],  # Y
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],  # V
], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Grantham distance (Grantham 1974, Science 185:862–864).
# Composite physicochemical distance combining composition, polarity, volume.
# Self-distance = 0; matrix is symmetric.
# ─────────────────────────────────────────────────────────────────────────────
_GRANTHAM_PAIRS: dict[tuple[str, str], int] = {
    ("A", "R"): 112, ("A", "N"): 111, ("A", "D"): 126, ("A", "C"): 195,
    ("A", "Q"):  91, ("A", "E"): 107, ("A", "G"):  60, ("A", "H"):  86,
    ("A", "I"):  94, ("A", "L"):  96, ("A", "K"): 106, ("A", "M"):  84,
    ("A", "F"): 113, ("A", "P"):  27, ("A", "S"):  99, ("A", "T"):  58,
    ("A", "W"): 148, ("A", "Y"): 112, ("A", "V"):  64,
    ("R", "N"):  86, ("R", "D"):  96, ("R", "C"): 180, ("R", "Q"):  43,
    ("R", "E"):  54, ("R", "G"): 125, ("R", "H"):  29, ("R", "I"):  97,
    ("R", "L"): 102, ("R", "K"):  26, ("R", "M"):  91, ("R", "F"):  97,
    ("R", "P"): 103, ("R", "S"): 110, ("R", "T"):  71, ("R", "W"): 101,
    ("R", "Y"):  77, ("R", "V"):  96,
    ("N", "D"):  23, ("N", "C"): 139, ("N", "Q"):  46, ("N", "E"):  42,
    ("N", "G"):  80, ("N", "H"):  68, ("N", "I"): 149, ("N", "L"): 153,
    ("N", "K"):  94, ("N", "M"): 142, ("N", "F"): 158, ("N", "P"):  91,
    ("N", "S"):  46, ("N", "T"):  65, ("N", "W"): 174, ("N", "Y"): 143,
    ("N", "V"): 133,
    ("D", "C"): 154, ("D", "Q"):  61, ("D", "E"):  45, ("D", "G"):  94,
    ("D", "H"):  81, ("D", "I"): 168, ("D", "L"): 172, ("D", "K"): 101,
    ("D", "M"): 160, ("D", "F"): 177, ("D", "P"): 108, ("D", "S"):  65,
    ("D", "T"):  85, ("D", "W"): 181, ("D", "Y"): 160, ("D", "V"): 152,
    ("C", "Q"): 154, ("C", "E"): 170, ("C", "G"): 159, ("C", "H"): 174,
    ("C", "I"): 198, ("C", "L"): 198, ("C", "K"): 202, ("C", "M"): 196,
    ("C", "F"): 205, ("C", "P"): 169, ("C", "S"): 112, ("C", "T"): 149,
    ("C", "W"): 215, ("C", "Y"): 194, ("C", "V"): 192,
    ("Q", "E"):  29, ("Q", "G"):  87, ("Q", "H"):  24, ("Q", "I"): 109,
    ("Q", "L"): 113, ("Q", "K"):  53, ("Q", "M"): 101, ("Q", "F"): 116,
    ("Q", "P"):  76, ("Q", "S"):  68, ("Q", "T"):  42, ("Q", "W"): 130,
    ("Q", "Y"):  99, ("Q", "V"):  96,
    ("E", "G"):  98, ("E", "H"):  40, ("E", "I"): 134, ("E", "L"): 138,
    ("E", "K"):  56, ("E", "M"): 126, ("E", "F"): 140, ("E", "P"):  93,
    ("E", "S"):  80, ("E", "T"):  65, ("E", "W"): 152, ("E", "Y"): 122,
    ("E", "V"): 121,
    ("G", "H"):  98, ("G", "I"): 135, ("G", "L"): 138, ("G", "K"): 127,
    ("G", "M"): 127, ("G", "F"): 153, ("G", "P"):  42, ("G", "S"):  56,
    ("G", "T"):  59, ("G", "W"): 184, ("G", "Y"): 147, ("G", "V"): 109,
    ("H", "I"):  94, ("H", "L"):  99, ("H", "K"):  32, ("H", "M"):  87,
    ("H", "F"): 100, ("H", "P"):  77, ("H", "S"):  89, ("H", "T"):  47,
    ("H", "W"): 115, ("H", "Y"):  83, ("H", "V"):  84,
    ("I", "L"):   5, ("I", "K"): 102, ("I", "M"):  10, ("I", "F"):  21,
    ("I", "P"):  95, ("I", "S"): 142, ("I", "T"):  89, ("I", "W"):  61,
    ("I", "Y"):  33, ("I", "V"):  29,
    ("L", "K"): 107, ("L", "M"):  15, ("L", "F"):  22, ("L", "P"):  98,
    ("L", "S"): 145, ("L", "T"):  92, ("L", "W"):  61, ("L", "Y"):  36,
    ("L", "V"):  32,
    ("K", "M"):  95, ("K", "F"): 102, ("K", "P"): 103, ("K", "S"): 121,
    ("K", "T"):  78, ("K", "W"): 110, ("K", "Y"):  85, ("K", "V"):  97,
    ("M", "F"):  28, ("M", "P"):  87, ("M", "S"): 135, ("M", "T"):  81,
    ("M", "W"):  67, ("M", "Y"):  36, ("M", "V"):  21,
    ("F", "P"): 114, ("F", "S"): 155, ("F", "T"): 103, ("F", "W"):  40,
    ("F", "Y"):  22, ("F", "V"):  50,
    ("P", "S"):  74, ("P", "T"):  38, ("P", "W"): 147, ("P", "Y"): 110,
    ("P", "V"):  68,
    ("S", "T"):  58, ("S", "W"): 177, ("S", "Y"): 144, ("S", "V"): 124,
    ("T", "W"): 128, ("T", "Y"):  92, ("T", "V"):  69,
    ("W", "Y"):  37, ("W", "V"):  88,
    ("Y", "V"):  55,
}


def _build_grantham_matrix() -> np.ndarray:
    n = len(AA_ORDER)
    mat = np.zeros((n, n), dtype=np.float32)
    for (a, b), d in _GRANTHAM_PAIRS.items():
        i, j = AA_INDEX[a], AA_INDEX[b]
        mat[i, j] = float(d)
        mat[j, i] = float(d)
    return mat


_GRANTHAM = _build_grantham_matrix()


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "rsa",
    "blosum62",
    "grantham",
    "delta_charge",
    "delta_polarity",
    "delta_hydrophobicity",
    "delta_volume",
]
INDICATOR_NAMES = [
    "mut_is_proline",
    "mut_is_glycine",
    "mut_is_aromatic",
]
# Extended set — sequence-derived structural proxies.
# These need (sequence, mut_idx) in addition to (wt, mut, rsa).
EXTENDED_NAMES = [
    "delta_helix_propensity",       # Chou-Fasman P_α(mut) − P_α(wt)
    "delta_sheet_propensity",       # Chou-Fasman P_β(mut) − P_β(wt)
    "local_entropy",                # Shannon entropy of AA composition in ±10 window
    "local_hydrophobic_count",      # # of {AILMVF} in ±5 window
    "local_charged_count",          # # of {DEKR} in ±5 window
    "position_relative",            # mut_idx / max(seq_len-1, 1) ∈ [0, 1]
]


# Chou-Fasman 1978 secondary-structure propensities.
_HELIX_PROP = {
    "A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70,
    "Q": 1.11, "E": 1.51, "G": 0.57, "H": 1.00, "I": 1.08,
    "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13, "P": 0.57,
    "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06,
}
_SHEET_PROP = {
    "A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19,
    "Q": 1.10, "E": 0.37, "G": 0.75, "H": 0.87, "I": 1.60,
    "L": 1.30, "K": 0.74, "M": 1.05, "F": 1.38, "P": 0.55,
    "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.70,
}
_HYDROPHOBIC_SET = set("AILMVF")
_CHARGED_SET    = set("DEKR")


def feature_dim(include_indicators: bool = False, include_extended: bool = False) -> int:
    """Output dimensionality of the feature vector."""
    n = len(FEATURE_NAMES)
    if include_indicators: n += len(INDICATOR_NAMES)
    if include_extended:   n += len(EXTENDED_NAMES)
    return n


def feature_names(include_indicators: bool = False, include_extended: bool = False) -> list[str]:
    """Ordered feature names produced by ``batch_features`` / ``batch_features_extended``."""
    out = list(FEATURE_NAMES)
    if include_indicators: out += list(INDICATOR_NAMES)
    if include_extended:   out += list(EXTENDED_NAMES)
    return out


def _aa(c: str) -> str:
    c = c.strip().upper()
    if c not in AA_INDEX:
        raise ValueError(f"unknown amino acid {c!r}")
    return c


def compute_features(
    wt: str,
    mut: str,
    rsa: float,
    include_indicators: bool = False,
) -> np.ndarray:
    """Feature vector for a single mutation. Returns shape (k,)."""
    wt, mut = _aa(wt), _aa(mut)
    iw, im = AA_INDEX[wt], AA_INDEX[mut]

    blosum   = float(_BLOSUM62[iw, im])
    grantham = float(_GRANTHAM[iw, im])
    d_charge = float(_CHARGE.get(mut, 0) - _CHARGE.get(wt, 0))
    d_polar  = float((wt in _POLAR) != (mut in _POLAR))
    d_hydro  = _HYDRO[mut] - _HYDRO[wt]
    d_vol    = _VOLUME[mut] - _VOLUME[wt]

    out = [float(rsa), blosum, grantham, d_charge, d_polar, d_hydro, d_vol]
    if include_indicators:
        out += [
            float(mut == "P"),
            float(mut == "G"),
            float(mut in {"F", "W", "Y"}),
        ]
    return np.asarray(out, dtype=np.float32)


def batch_features(
    wt: list[str] | np.ndarray,
    mut: list[str] | np.ndarray,
    rsa: list[float] | np.ndarray,
    include_indicators: bool = False,
) -> np.ndarray:
    """Vectorised feature extraction.

    Parameters
    ----------
    wt, mut : iterables of single-letter amino-acid codes (length n)
    rsa     : iterable of floats in [0, 1] (length n)

    Returns
    -------
    (n, k) float32 array, where k = ``feature_dim(include_indicators)``.
    """
    wt = [str(s).strip().upper() for s in wt]
    mut = [str(s).strip().upper() for s in mut]
    rsa = np.asarray(rsa, dtype=np.float32)
    n = len(wt)
    if not (len(mut) == n == len(rsa)):
        raise ValueError("wt, mut and rsa must have the same length")

    iw = np.fromiter((AA_INDEX[c] for c in wt), dtype=np.int32, count=n)
    im = np.fromiter((AA_INDEX[c] for c in mut), dtype=np.int32, count=n)

    blosum   = _BLOSUM62[iw, im]
    grantham = _GRANTHAM[iw, im]

    wt_ch  = np.array([_CHARGE.get(c, 0) for c in wt], dtype=np.float32)
    mut_ch = np.array([_CHARGE.get(c, 0) for c in mut], dtype=np.float32)
    d_ch   = mut_ch - wt_ch

    wt_pol  = np.array([c in _POLAR for c in wt], dtype=np.float32)
    mut_pol = np.array([c in _POLAR for c in mut], dtype=np.float32)
    d_pol   = (wt_pol != mut_pol).astype(np.float32)

    wt_hy  = np.array([_HYDRO[c] for c in wt], dtype=np.float32)
    mut_hy = np.array([_HYDRO[c] for c in mut], dtype=np.float32)
    d_hy   = mut_hy - wt_hy

    wt_v   = np.array([_VOLUME[c] for c in wt], dtype=np.float32)
    mut_v  = np.array([_VOLUME[c] for c in mut], dtype=np.float32)
    d_v    = mut_v - wt_v

    parts = [rsa, blosum, grantham, d_ch, d_pol, d_hy, d_v]
    if include_indicators:
        parts += [
            np.array([c == "P" for c in mut], dtype=np.float32),
            np.array([c == "G" for c in mut], dtype=np.float32),
            np.array([c in {"F", "W", "Y"} for c in mut], dtype=np.float32),
        ]
    return np.stack(parts, axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Extended sequence-derived features (need full sequence + mut_idx)
# ─────────────────────────────────────────────────────────────────────────────
def _local_window(seq: str, mut_idx: int, half_window: int) -> str:
    """Extract a substring of length ≤ 2·half_window+1 centred on ``mut_idx``."""
    lo = max(0, mut_idx - half_window)
    hi = min(len(seq), mut_idx + half_window + 1)
    return seq[lo:hi]


def _shannon_entropy(s: str) -> float:
    """Shannon entropy of AA frequencies in ``s`` (in bits, base 2)."""
    if not s:
        return 0.0
    n = len(s)
    counts: dict[str, int] = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1
    p = np.array(list(counts.values()), dtype=np.float64) / n
    return float(-(p * np.log2(p + 1e-12)).sum())


def _count_in_set(s: str, charset: set[str]) -> int:
    return sum(1 for c in s if c in charset)


def batch_features_extended(
    wt: list[str] | np.ndarray,
    mut: list[str] | np.ndarray,
    rsa: list[float] | np.ndarray,
    sequences: list[str],
    mut_indices: list[int] | np.ndarray,
    *,
    include_indicators: bool = False,
    half_window_short: int = 5,
    half_window_long:  int = 10,
) -> np.ndarray:
    """Extended feature builder: same 7 base features as ``batch_features`` plus
    six sequence-derived structural proxies that don't need a PDB.

    Extra features (in order):
        - delta_helix_propensity      (Chou-Fasman P_α(mut) − P_α(wt))
        - delta_sheet_propensity      (Chou-Fasman P_β(mut) − P_β(wt))
        - local_entropy               (Shannon entropy in ±``half_window_long``)
        - local_hydrophobic_count     (#{AILMVF} in ±``half_window_short``)
        - local_charged_count         (#{DEKR} in ±``half_window_short``)
        - position_relative           (mut_idx / max(seq_len-1, 1))

    Parameters
    ----------
    wt, mut, rsa : iterables, length n   (single-letter AA codes for wt/mut)
    sequences    : list of full protein sequences, length n
    mut_indices  : 0-based mutation positions in ``sequences[i]``, length n
    include_indicators : also append the 3 indicator features (mut is P / G / aromatic)

    Returns
    -------
    (n, k) float32 array, k = ``feature_dim(include_indicators, include_extended=True)``.
    """
    base = batch_features(wt, mut, rsa, include_indicators=include_indicators)
    n = base.shape[0]
    if not (len(sequences) == len(mut_indices) == n):
        raise ValueError(
            f"sequences/mut_indices length mismatch: "
            f"got {len(sequences)} / {len(mut_indices)}, expected {n}"
        )

    wt_arr  = [str(s).strip().upper() for s in wt]
    mut_arr = [str(s).strip().upper() for s in mut]

    d_helix = np.array(
        [_HELIX_PROP[m] - _HELIX_PROP[w] for w, m in zip(wt_arr, mut_arr)],
        dtype=np.float32,
    )
    d_sheet = np.array(
        [_SHEET_PROP[m] - _SHEET_PROP[w] for w, m in zip(wt_arr, mut_arr)],
        dtype=np.float32,
    )

    local_entropy = np.zeros(n, dtype=np.float32)
    local_hydro   = np.zeros(n, dtype=np.float32)
    local_charged = np.zeros(n, dtype=np.float32)
    position_rel  = np.zeros(n, dtype=np.float32)

    for i in range(n):
        seq = str(sequences[i]).strip().upper()
        mi  = int(mut_indices[i])
        # Clamp into valid range; long sequences may be truncated upstream.
        mi  = max(0, min(mi, max(len(seq) - 1, 0)))

        long_win  = _local_window(seq, mi, half_window_long)
        short_win = _local_window(seq, mi, half_window_short)

        local_entropy[i] = _shannon_entropy(long_win)
        local_hydro[i]   = float(_count_in_set(short_win, _HYDROPHOBIC_SET))
        local_charged[i] = float(_count_in_set(short_win, _CHARGED_SET))
        position_rel[i]  = mi / max(len(seq) - 1, 1)

    extended = np.stack(
        [d_helix, d_sheet, local_entropy, local_hydro, local_charged, position_rel],
        axis=1,
    ).astype(np.float32)

    return np.concatenate([base, extended], axis=1)


def standardize(
    feats_train: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Z-score features using train statistics.

    BLOSUM/Grantham/volume/hydrophobicity have very different scales — the
    σ MLP trains better when they are standardised.  RSA + indicator dims
    are bounded so the transform is harmless for them.

    Returns standardised arrays in the same order as the inputs.
    """
    mu = feats_train.mean(axis=0)
    sd = feats_train.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return tuple(((arr - mu) / sd).astype(np.float32) for arr in (feats_train, *others))


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert a numpy feature array to a float32 torch tensor."""
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))
