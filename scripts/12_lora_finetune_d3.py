"""LoRA fine-tune ESM2-650M end-to-end for the D3 σ-branch on T2837.

Why D3 (and not D1)
===================
K-fold CV (script 11, K=5 folds × 3 seeds = 15 paired observations on
n=2584 OOS) is the *only* run with enough power to reach the noise
floor.  Result:

    D1_vs_D0   Δ +0.010   p=0.39    no signal
    D2_vs_D0   Δ +0.012   p=0.43    no signal
    D3_vs_D0   Δ +0.023   p=0.040   * SIGNIFICANT (12/15 wins)

Single-seed and 5-seed runs that previously suggested D1 was the winner
were under-powered.  The combination of RSA + chemistry in the σ branch
(D3) is the only ablation that reliably beats the no-extras baseline.
LoRA fine-tune therefore targets D3 architecture.

Why LoRA fine-tune
==================
Pooled OOS metrics on the 650M backbone with frozen embeddings:
    RMSE 1.85   NLL 2.5   ICE 0.10
RMSE is the binding constraint.  The 650M backbone has plenty of capacity
but its general-purpose embedding is not tuned to ddG.  LoRA adapters on
the last L attention layers (default L=full, rank 8) unlock per-task
representations with ~0.5% of the full backbone parameter budget.

Architecture
============
1. Load frozen ESM2-650M.
2. Wrap with peft LoRA on attention `query` and `value` projections.
3. Build (input_ids, attention_mask, mut_idx, wt_oh, mut_oh, rsa, chem, y)
   per mutation.
4. Forward sequence → per-residue embeddings (with LoRA).
5. Build per-mutation feature: [h_site, h_window, e(wtAA), e(mutAA)].
6. Pass through FeatureAugmentedHead (D3 = 7 extras: RSA + 6 chemistry).
7. Loss = Student-t NLL (ν=3).  Gradients flow through head → LoRA only;
   base ESM2 weights stay frozen.

Compute estimate (M-series MPS, batch_size=8, 20 epochs)
========================================================
Per epoch  ≈ 5-10 min.  Total  ≈ 2-3 hours.  Recommend running in the
background; the script saves checkpoints + a JSON summary at the end.

Required dependency: ``pip install peft``

Outputs (under --out)
=====================
- ``lora_adapter/``       LoRA adapter weights (small, ~10-50 MB)
- ``head_state.pt``       FeatureAugmentedHead state dict
- ``training_log.json``   per-epoch train/val loss
- ``test_metrics.json``   final test-set metrics
- ``test_predictions.npz`` test mu, sigma, y arrays

Usage
-----
    python scripts/12_lora_finetune_d3.py \\
        --metadata-csv cache/t2837_metadata.csv \\
        --bio-feats    cache/t2837_bio_features_650m.pt \\
        --out          outputs/lora_d3_650m \\
        --device mps --batch-size 8 --max-epochs 20
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uapp.evaluate import (
    compute_gaussian_nll, compute_ice, compute_mae, compute_rmse,
    compute_spearman_sigma_error, compute_top_k_risk_capture,
)
from uapp.heads import FeatureAugmentedHead
from uapp.losses import student_t_nll_loss, uncertainty_ranking_loss
from uapp.utils import ensure_dir, get_device, set_seed, setup_logging


# ─────────────────────────────────────────────────────────────────────────────
# T2837 row → tokens + auxiliary features
# ─────────────────────────────────────────────────────────────────────────────
AA3to1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
AA1_LIST = sorted(set(AA3to1.values()))
AA1_TO_IDX = {a: i for i, a in enumerate(AA1_LIST)}


def aa_one_hot(aa3: str) -> torch.Tensor:
    aa1 = AA3to1.get(aa3.strip().upper(), None)
    v = torch.zeros(20)
    if aa1 is not None and aa1 in AA1_TO_IDX:
        v[AA1_TO_IDX[aa1]] = 1.0
    return v


def find_mutation_index(seq: str, position: int, wt_aa3: str) -> int | None:
    """Mirror of 01_cache_embeddings_esm_v2's resolution logic."""
    wt1 = AA3to1.get(wt_aa3, "?")
    if wt1 == "?":
        return None
    idx = int(position) - 1
    if 0 <= idx < len(seq) and seq[idx] == wt1:
        return idx
    for offset in range(1, 50):
        for sign in (-1, 1):
            c = idx + sign * offset
            if 0 <= c < len(seq) and seq[c] == wt1:
                return c
    return None


class MutationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, bio_feats: torch.Tensor, tokenizer, max_len: int = 1022):
        assert len(df) == bio_feats.shape[0], (
            f"DF rows {len(df)} != bio_feats rows {bio_feats.shape[0]} — alignment broken."
        )
        self.df = df.reset_index(drop=True)
        self.bio = bio_feats
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        seq = str(row["sequence"])
        if len(seq) > self.max_len:
            seq = seq[: self.max_len]
        toks = self.tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        mut_idx = int(row["mut_idx"])
        # Clamp into valid range [0, len(seq)-1]
        mut_idx = max(0, min(mut_idx, len(seq) - 1))
        return {
            "input_ids":      toks["input_ids"][0],
            "attention_mask": toks["attention_mask"][0],
            "seq_len":        len(seq),
            "mut_idx":        mut_idx,
            "wt_oh":          aa_one_hot(row["wtAA"]),
            "mut_oh":         aa_one_hot(row["mutAA"]),
            "rsa":            self.bio[i, 0:1].clone(),    # (1,)
            "chemistry":      self.bio[i, 1:7].clone(),    # (6,)
            "y":              torch.tensor(float(row["ddG"])),
        }


def collate(batch: list[dict], pad_token_id: int) -> dict:
    return {
        "input_ids":      pad_sequence([b["input_ids"]      for b in batch],
                                       batch_first=True, padding_value=pad_token_id),
        "attention_mask": pad_sequence([b["attention_mask"] for b in batch],
                                       batch_first=True, padding_value=0),
        "seq_len":  torch.tensor([b["seq_len"]  for b in batch], dtype=torch.long),
        "mut_idx":  torch.tensor([b["mut_idx"]  for b in batch], dtype=torch.long),
        "wt_oh":    torch.stack([b["wt_oh"]    for b in batch]),
        "mut_oh":   torch.stack([b["mut_oh"]   for b in batch]),
        "rsa":      torch.stack([b["rsa"]      for b in batch]),
        "chemistry":torch.stack([b["chemistry"]for b in batch]),
        "y":        torch.stack([b["y"]        for b in batch]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build per-mutation feature from per-residue embeddings (vectorised)
# ─────────────────────────────────────────────────────────────────────────────
def build_mutation_features(
    residue_embs: torch.Tensor,    # (B, L_max, D)
    seq_len: torch.Tensor,         # (B,)  effective sequence length (after stripping CLS/EOS)
    mut_idx: torch.Tensor,         # (B,)
    wt_oh: torch.Tensor,           # (B, 20)
    mut_oh: torch.Tensor,          # (B, 20)
    window_size: int = 5,
) -> torch.Tensor:
    """Return (B, 2*D + 40) feature tensor."""
    B, L_max, D = residue_embs.shape
    device = residue_embs.device

    # Clamp mut_idx into valid range per sample
    mut_idx_c = torch.clamp(mut_idx, min=0)
    mut_idx_c = torch.minimum(mut_idx_c, seq_len.to(device) - 1)

    # h_site: gather residue at mut_idx for each sample
    idx_expand = mut_idx_c.view(B, 1, 1).expand(B, 1, D)
    h_site = residue_embs.gather(1, idx_expand).squeeze(1)        # (B, D)

    # h_window: mean over [mut_idx-window, mut_idx+window], masked to seq_len
    pos = torch.arange(L_max, device=device).view(1, L_max)        # (1, L_max)
    lo = (mut_idx_c.view(B, 1) - window_size).clamp(min=0)
    hi = (mut_idx_c.view(B, 1) + window_size + 1).clamp(max=seq_len.to(device).view(B, 1))
    win_mask = ((pos >= lo) & (pos < hi)).float().unsqueeze(-1)    # (B, L_max, 1)
    win_sum = (residue_embs * win_mask).sum(dim=1)                 # (B, D)
    win_count = win_mask.sum(dim=1).clamp(min=1.0)                 # (B, 1)
    h_window = win_sum / win_count

    return torch.cat([h_site, h_window, wt_oh.to(device).float(), mut_oh.to(device).float()], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Training / eval loops
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(
    backbone, head, loader, optimizer, device,
    *, train: bool, nu: float,
    ranking_lambda: float = 0.0, ranking_margin: float = 0.05,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    backbone.train(train); head.train(train)
    total_loss, total_n = 0.0, 0
    mus, sigs, ys = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in loader:
            for k_ in batch:
                batch[k_] = batch[k_].to(device)
            if train:
                optimizer.zero_grad()

            out = backbone(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"])
            # Strip [CLS] (pos 0) and [EOS] (pos seq_len+1).  Use seq_len as the
            # mutation-index frame: the dataset's mut_idx is in 0..len(seq)-1.
            residue = out.last_hidden_state[:, 1:-1, :]
            feats = build_mutation_features(
                residue, batch["seq_len"], batch["mut_idx"],
                batch["wt_oh"], batch["mut_oh"], window_size=5,
            )
            extras = torch.cat([batch["rsa"], batch["chemistry"]], dim=-1).float()

            mu, sigma = head(feats, extras)
            base = student_t_nll_loss(mu, sigma, batch["y"].float(), nu=nu)
            if ranking_lambda > 0.0 and train:
                rank = uncertainty_ranking_loss(
                    pred_mean=mu, pred_sigma=sigma,
                    target=batch["y"].float(), margin=ranking_margin,
                )
                loss = base + ranking_lambda * rank
            else:
                loss = base

            if train:
                loss.backward()
                optimizer.step()

            n = batch["y"].size(0)
            total_loss += float(loss.item()) * n
            total_n += n
            mus.append(mu.detach().cpu().numpy())
            sigs.append(sigma.detach().cpu().numpy())
            ys.append(batch["y"].detach().cpu().numpy())

    return (total_loss / max(total_n, 1),
            np.concatenate(mus), np.concatenate(sigs), np.concatenate(ys))


def metrics_dict(mu, sigma, y) -> dict:
    rmse = compute_rmse(mu, y); mae = compute_mae(mu, y)
    nll  = compute_gaussian_nll(mu, sigma, y)
    ice, cov = compute_ice(mu, sigma, y)
    sp = compute_spearman_sigma_error(mu, sigma, y)
    tk = compute_top_k_risk_capture(mu, sigma, y, k_fracs=[0.10, 0.20, 0.30])
    return {
        "n":     int(len(y)),
        "rmse": rmse, "mae": mae, "nll": nll, "ice": ice,
        "cov@0.50": cov.get("0.50"), "cov@0.80": cov.get("0.80"),
        "cov@0.90": cov.get("0.90"), "cov@0.95": cov.get("0.95"),
        "spearman": sp,
        "top0.10":  tk["0.10"], "top0.20": tk["0.20"], "top0.30": tk["0.30"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata-csv", required=True, type=Path,
                   help="Processed CSV from 01_cache_embeddings_esm_v2 (has split + mut_idx)")
    p.add_argument("--bio-feats",    required=True, type=Path,
                   help="Aligned bio features from script 06 (must match metadata-csv rows)")
    p.add_argument("--out",          required=True, type=Path)

    p.add_argument("--esm-model",    default="facebook/esm2_t33_650M_UR50D")
    p.add_argument("--lora-r",       type=int, default=8)
    p.add_argument("--lora-alpha",   type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", nargs="+",
                   default=["query", "value"],
                   help="ESM2 attention sub-modules to LoRA-ify")

    p.add_argument("--batch-size",   type=int, default=8)
    p.add_argument("--d-hidden",     type=int, default=128)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--max-epochs",   type=int, default=20)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience",     type=int, default=5)
    p.add_argument("--nu",           type=float, default=3.0)
    p.add_argument("--ranking-lambda", type=float, default=0.0,
                   help="Pairwise ranking loss weight (0 = NLL only).  "
                        "Recommended 0.05–0.10 to combat σ-branch variance collapse.")
    p.add_argument("--ranking-margin", type=float, default=0.05)

    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--device",       type=str, default="auto")
    p.add_argument("--log-level",    type=str, default="INFO")
    args = p.parse_args()

    log = setup_logging(args.log_level)
    set_seed(args.seed)
    device = get_device(device_str=args.device)
    out_dir = ensure_dir(args.out)
    log.info("Device: %s", device)

    # ── Imports that might be missing ────────────────────────────────────────
    try:
        from transformers import EsmModel, EsmTokenizer
        from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    except ImportError as e:
        raise SystemExit(
            f"Missing required dependency: {e}.  Run `pip install transformers peft`."
        )

    # ── Load tokenizer + base model ──────────────────────────────────────────
    log.info("Loading tokenizer + base model: %s", args.esm_model)
    tokenizer = EsmTokenizer.from_pretrained(args.esm_model)
    base_model = EsmModel.from_pretrained(args.esm_model)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    backbone = get_peft_model(base_model, lora_cfg)
    backbone.print_trainable_parameters()
    backbone.to(device)

    # ── Load metadata + bio features ─────────────────────────────────────────
    log.info("Loading metadata CSV: %s", args.metadata_csv)
    df = pd.read_csv(args.metadata_csv)
    df["__split"] = df["split"].astype(str).str.strip().str.lower()

    # Resolve mut_idx if missing (the v2 cache script saves one)
    if "mut_idx" not in df.columns:
        log.info("Resolving mut_idx for each row …")
        mi = df.apply(lambda r: find_mutation_index(r["sequence"], r["position"], r["wtAA"]), axis=1)
        df["mut_idx"] = mi
    df = df.dropna(subset=["mut_idx"]).copy()
    df["mut_idx"] = df["mut_idx"].astype(int)

    log.info("Loading bio features: %s", args.bio_feats)
    bio_payload = torch.load(args.bio_feats, map_location="cpu", weights_only=False)
    log.info("Bio features: %s", bio_payload["meta"].get("feature_names"))

    # Group rows by split, validate alignment with bio_feats
    split_dfs: dict[str, pd.DataFrame] = {}
    for s in ("train", "val", "test"):
        sub = df[df["__split"] == s].reset_index(drop=True)
        bio_n = bio_payload[s]["feats"].shape[0]
        if len(sub) != bio_n:
            raise RuntimeError(
                f"Alignment failure on split {s!r}: "
                f"metadata rows={len(sub)} but bio_feats rows={bio_n}.  "
                "Re-run script 06."
            )
        split_dfs[s] = sub

    # ── DataLoaders ──────────────────────────────────────────────────────────
    pad_token_id = tokenizer.pad_token_id

    def loader(split: str, shuffle: bool) -> DataLoader:
        ds = MutationDataset(split_dfs[split], bio_payload[split]["feats"], tokenizer)
        return DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle,
            collate_fn=lambda b: collate(b, pad_token_id),
            num_workers=0,
        )

    train_loader = loader("train", shuffle=True)
    val_loader   = loader("val",   shuffle=False)
    test_loader  = loader("test",  shuffle=False)

    # ── Build head (D3 = RSA + 6 chemistry features = 7 extras) ─────────────
    d_in = backbone.config.hidden_size * 2 + 40
    log.info("Head input dim = 2·%d + 40 = %d  |  d_extra = 7 (RSA + chemistry)",
             backbone.config.hidden_size, d_in)
    head = FeatureAugmentedHead(
        d_in, d_extra=7,
        d_hidden=args.d_hidden, dropout=args.dropout,
        init_sigma_bias=0.5,
    ).to(device)

    # ── Optimiser: LoRA params + head ────────────────────────────────────────
    trainable = [p for p in backbone.parameters() if p.requires_grad] + list(head.parameters())
    n_trainable = sum(p.numel() for p in trainable)
    log.info("Trainable params: %d (LoRA + head)", n_trainable)
    opt = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)

    # ── Training loop ────────────────────────────────────────────────────────
    history: list[dict] = []
    best_val = float("inf")
    best_lora_state = None
    best_head_state = None
    best_epoch = -1
    patience_ctr = 0
    t0 = time.time()

    for epoch in range(1, args.max_epochs + 1):
        ep_t0 = time.time()
        train_loss, _, _, _ = run_epoch(
            backbone, head, train_loader, opt, device, train=True, nu=args.nu,
            ranking_lambda=args.ranking_lambda, ranking_margin=args.ranking_margin,
        )
        val_loss, _, _, _ = run_epoch(
            backbone, head, val_loader, None, device, train=False, nu=args.nu,
            ranking_lambda=args.ranking_lambda, ranking_margin=args.ranking_margin,
        )
        ep_dur = time.time() - ep_t0
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "duration_s": ep_dur})

        improved = val_loss < best_val - 1e-4
        if improved:
            best_val = val_loss
            best_epoch = epoch
            # Snapshot only the LoRA params + head
            best_lora_state = {k: v.detach().cpu().clone()
                               for k, v in backbone.state_dict().items()
                               if "lora_" in k}
            best_head_state = copy.deepcopy(head.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        log.info(
            "epoch %3d/%3d  train %.4f  val %.4f  best %.4f @ ep %d  (%.1fs)",
            epoch, args.max_epochs, train_loss, val_loss,
            best_val, best_epoch, ep_dur,
        )

        if patience_ctr >= args.patience:
            log.info("early stop at epoch %d", epoch)
            break

    log.info("Training done in %.1f min", (time.time() - t0) / 60)

    # ── Restore best checkpoint, evaluate on test ───────────────────────────
    if best_head_state is not None:
        head.load_state_dict(best_head_state)
    if best_lora_state is not None:
        # Inject best LoRA weights back into the model
        sd = backbone.state_dict()
        sd.update({k: v.to(device) for k, v in best_lora_state.items()})
        backbone.load_state_dict(sd)
    log.info("Restored best checkpoint (epoch %d, val_loss %.4f)", best_epoch, best_val)

    _, mu_te, sig_te, y_te = run_epoch(
        backbone, head, test_loader, None, device, train=False, nu=args.nu,
    )
    te_metrics = metrics_dict(mu_te, sig_te, y_te)
    log.info("Test metrics: %s", json.dumps(te_metrics, indent=2))

    # ── Persist ─────────────────────────────────────────────────────────────
    backbone.save_pretrained(out_dir / "lora_adapter")
    torch.save(head.state_dict(), out_dir / "head_state.pt")
    np.savez(out_dir / "test_predictions.npz", mu=mu_te, sigma=sig_te, y=y_te)
    (out_dir / "training_log.json").write_text(json.dumps(history, indent=2))
    (out_dir / "test_metrics.json").write_text(json.dumps(te_metrics, indent=2))

    # ── Pretty print final summary ──────────────────────────────────────────
    print("\n" + "=" * 78)
    print("LoRA fine-tune of ESM2-650M on D3 (RSA + chemistry)")
    print("=" * 78)
    print(f"Best epoch:    {best_epoch}  (val NLL = {best_val:.4f})")
    print(f"\nTest metrics  (n = {te_metrics['n']}):")
    print(f"  RMSE        = {te_metrics['rmse']:.4f}")
    print(f"  MAE         = {te_metrics['mae']:.4f}")
    print(f"  NLL         = {te_metrics['nll']:.4f}")
    print(f"  ICE         = {te_metrics['ice']:.4f}")
    print(f"  cov@90      = {te_metrics['cov@0.90']:.4f}")
    print(f"  Spearman    = {te_metrics['spearman']:+.4f}")
    print(f"  top-20 risk = {te_metrics['top0.20']:.4f}")
    print()
    print("Compare against the frozen-backbone D3 baseline (script 09 fixed-split):")
    print("  Frozen  RMSE 1.50   NLL 1.89   ICE 0.069   Spearman 0.333")
    print("=" * 78)


if __name__ == "__main__":
    main()
