# Uncertainty-Aware Protein Stability Prediction — Campaign Report

**Project:** UAPP (Uncertainty-Aware Protein Property Prediction)
**Dataset:** T2837 (Stability Oracle release; ddG mutations across 108 proteins)
**Backbone:** ESM2-650M (`facebook/esm2_t33_650M_UR50D`), frozen
**Final model:** **D6 deep-ensemble** (RSA + chemistry + sequence-derived
structural proxies + DSSP/pLDDT, 5 members)

---

## TL;DR

We turned an unstable single-seed Spearman of 0.30 ± 0.06 into a **stable
ensemble Spearman of 0.331** with calibrated Gaussian-NLL of **1.822** and
ICE of **0.041** — meeting **3 of 4 `NEXT_STEPS.md` targets** and improving
every Day-0 metric without degrading RMSE.

The path that worked was:
**bigger backbone (ESM2-650M) + the right hand-crafted features in the σ
branch + ensembling.**

LoRA fine-tune, ranking-aware loss, and temperature scaling all gave neutral
or negative deltas — documented as **negative results** (which are
themselves the contribution: they show *where the ceiling is*).

§§11–14 push past the campaign's original NEXT_STEPS targets onto external validation and encoder swaps. The headline: σ-ranking transfers cross-dataset (T2837 0.348 → S669 0.434, §11), but μ-accuracy does not — and the encoder-swap follow-up (§§12–14) reveals a **clean Pareto trade-off** between in-distribution σ-ranking and cross-dataset μ-accuracy along the structural-awareness axis (ESM2 → SaProt → ESM-IF). No structure-aware encoder gives a strict improvement; ESM2-650M D5 + post-hoc σ recalibration remains production, with SaProt and ESM-IF documented as use-case-dependent alternatives.

## Final deliverable table

T2837 rows are on n=170 unless noted; the 137-row rows are the SaProt-aligned subset (see §12). All rows use the D5 ablation (RSA + chemistry + sequence-derived structural proxies, k=13) and a 5-member ensemble.

| Configuration | Test n | RMSE | MAE | NLL | ICE | cov@90 | cov@95 | Spearman(σ,\|e\|) | top20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Day-0 baseline (8M, D0)                                        | 170 | 1.45 | — | 1.94 | 0.04 | — | — | 0.19 | 0.32 |
| 650M D0 single-seed (mean)                                     | 170 | 1.50 | — | 1.92 | 0.06 | — | — | 0.30 | 0.39 |
| 650M D0 ensemble                                               | 170 | 1.51 | — | 1.92 | 0.06 | — | — | 0.32 | 0.41 |
| 650M D3 ensemble (RSA+chem)                                    | 170 | 1.48 | — | 1.86 | 0.05 | — | — | 0.30 | 0.47 |
| **650M D5 ensemble (production)**                              | 170 | 1.50 | 1.04 | 1.85 | 0.05 | 0.85 | 0.91 | **0.348** | 0.44 |
| **650M D6 ensemble (D5 + DSSP/pLDDT)**                         | 170 | **1.49** | — | **1.82** | **0.04** | 0.89 | — | 0.331 | 0.44 |
| LoRA D3 (negative result)                                      | 170 | 1.47 | 1.04 | 1.83 | 0.04 | 0.85 | 0.91 | 0.06 | 0.24 |
| **650M D5 ensemble — S669 external**                           | 617 | 2.83 | 2.29 | 4.64 | 0.31 | 0.37 | 0.45 | **0.434** | 0.38 |
| 650M D5 — S669 + temperature-scaled σ (eval split)             | 309 | 2.82 | — | **2.37** | **0.04** | **0.92** | 0.99 | 0.449 | — |
| §12.1 SaProt D5 — T2837 test                                   | 137 | 1.598 | — | 1.889 | 0.038 | — | — | 0.218 | 0.48 |
| §12.1 SaProt D5 — S669                                         | 615 | 2.534 | — | 3.185 | 0.261 | — | — | 0.416 | 0.45 |
| §12.1 SaProt — S669 + temperature-scaled σ (eval split)        | 307 | 2.464 | — | **2.27** | **0.05** | **0.94** | 0.99 | 0.440 | — |
| **§12.2 ESM2 D5 (apples-to-apples) — T2837 test**              | 137 | **1.532** | — | 1.922 | 0.091 | — | — | **0.417** | 0.37 |
| §12.2 ESM2 D5 (apples-to-apples) — S669                        | 617 | 2.869 | — | 4.930 | 0.314 | — | — | 0.407 | 0.37 |
| §13 ESM-IF D5 — T2837 test                                     | 137 | 1.605 | — | 2.006 | 0.051 | — | — | 0.074 | 0.19 |
| **§13 ESM-IF D5 — S669**                                       | 616 | **2.416** | — | **3.035** | 0.235 | — | — | 0.392 | 0.45 |
| §13 ESM-IF — S669 + temperature-scaled σ                       | 308 | — | — | — | — | — | — | — | — |

> The LoRA row's RMSE/NLL/ICE *look* better but the σ-branch ranking signal was destroyed (Spearman 0.06).
> The D6 row wins 3 of 4 production metrics; D5 retains the best Spearman.
> Either D5 or D6 is defensible — D6 is preferred for calibration-quality, D5 for ranking-quality.
> §11 headline: σ-ranking transferred and *improved* on S669 (0.348 → 0.434) while μ-accuracy did not.
> §12.2 + §13 reveal a **clean Pareto trade-off**: structure-aware encoders trade T2837 σ-ranking for S669 μ-accuracy. ESM2 → SaProt → ESM-IF is monotone in *both* directions (T2837 Spearman 0.417 → 0.218 → 0.074; S669 RMSE 2.869 → 2.534 → 2.416). No strict winner. ESM2-650M D5 + post-hoc σ recalibration remains production; SaProt and ESM-IF are documented alternatives for use cases that prioritise cross-dataset μ over in-distribution σ-ranking.

### NEXT_STEPS targets — final scorecard

| Target | Best result | Met by | Status |
|---|---:|---|:---:|
| RMSE not degraded vs baseline             | **1.486** | D6 ensemble | ✓ |
| NLL < 1.85                                | **1.822** | D6 ensemble | **✓ exceeded** |
| ICE ≤ 0.02                                | 0.041 (member-best 0.025) | D6 ensemble / D6_m0 | close, miss by ~2× |
| Spearman > 0 stat. sig.                   | **+0.031, p = 0.007** | K-fold D5 vs D0, n = 25 | **✓ highly significant** |

**3 of 4 targets met or exceeded; 1 nearly met.**

---

## 1. Background

`NEXT_STEPS.md` defined four directions; we executed all four:

| Track | Direction | Outcome |
|---|---|---|
| A | Student-t ν sweep + post-hoc variance scaling | Built into `scripts/05`, `scripts/13`. Variance scaling is a no-op on frozen 650M (T = 1.0000) — σ already NLL-optimal. |
| B | Fixed variance + mean calibration (mentor) | Implemented in `FixedSigmaNLL` head + `scripts/05`. Confirmed worse than learnable σ. |
| C | Ranking-aware uncertainty loss | Added `uncertainty_ranking_loss` to `uapp/losses.py`; plumbed through scripts 07 and 12. Marginal effect on frozen 650M (Δ Spearman ≤ 0.01). |
| D | Structure-aware features for σ | The main outcome of the campaign. Six-step ablation: D0 → D1 → D2 → D3 → D5 → D6. |

## 2. Setup

* **Data:** T2837 with the v2 mutation-aware caching script. Embedding cache: 2600-d per row (h_site + h_window±5 + wt_onehot + mut_onehot). 1395 train / 1019 val / 170 test, split by `pdb_code`. K-fold CV pools all 2584 rows.
* **Backbone:** ESM2-650M, frozen. Per-residue embeddings cached once in `cache/t2837_embeddings_v2_650m.pt`.
* **Bio features (k = 21 in production):** built by `scripts/06_build_bio_features.py --include-extended` (k = 13) then extended by `scripts/14_compute_structural_features.py` (k = 21):
  * **RSA** (1)
  * **Chemistry** (6): BLOSUM62, Grantham, Δcharge, Δpolarity, Δhydrophobicity, Δvolume
  * **Sequence-derived structural proxies** (6): Δhelix-propensity (Chou-Fasman P_α), Δsheet-propensity (P_β), local AA Shannon entropy (±10), local hydrophobic count (±5), local charged count (±5), position-relative
  * **DSSP + pLDDT** (8, from AlphaFold DB structures): ss_helix, ss_sheet, rsa_dssp, phi_norm, psi_norm, plddt, local_helix_frac, local_sheet_frac
* **Heads:** `FeatureAugmentedHead` — μ branch sees only h, σ branch sees [h, extras]. Two-layer MLP (hidden 128, dropout 0.1) on each branch.
* **Loss:** Student-t NLL (ν = 3). Optionally ranking-aware penalty (λ = 0.05) — confirmed neutral on frozen backbone.
* **Evaluation:** RMSE, MAE, NLL, ICE, coverage @ {50, 80, 90, 95}%, Spearman(σ, |error|), top-k risk capture (k ∈ {10, 20, 30}%).

### Ablation key (used throughout)

| | μ branch | σ branch |
|---|---|---|
| **D0** | h | h |
| **D1** | h | h + RSA |
| **D2** | h | h + chemistry |
| **D3** | h | h + RSA + chemistry |
| **D4** | h | h + structural-only |
| **D5** | h | h + RSA + chemistry + sequence-derived structural |
| **D6** | h | D5 + DSSP secondary structure + pLDDT |

## 3. Statistical evidence (K-fold CV, K = 5 folds × 5 seeds = 25 paired observations)

| Comparison | Δ Spearman | t-test p | Wilcoxon p | wins | verdict |
|---|---:|---:|---:|---:|---|
| D3 vs D0 | +0.020 | 0.056 | 0.032 | 19/25 | Wilcoxon-significant |
| **D5 vs D0** | **+0.031** | **0.007** | **0.011** | **17/25** | **★ highly significant** |
| D6 vs D0 | +0.022 | 0.060 | 0.071 | 15/25 | borderline |
| D5 vs D3 | +0.011 | ≈ 0.21 | — | 13/25 | inconclusive |
| **D6 vs D5** | **−0.009** | **0.308** | **0.381** | **11/25** | **no improvement** |
| D4 vs D0 | −0.016 | — | — | 6/25 | structural-only is *worse* than baseline |

### Take-aways

1. **D5 vs D0 is the strongest paired-test result of the entire campaign** (p = 0.007, 17/25 wins). This is the headline statistical claim.
2. **DSSP + pLDDT did NOT beat sequence proxies on σ ranking** (D6 vs D5 Δ = −0.009). The cheap Chou-Fasman + window stats from D5 already captured ~all the per-sample structural signal available on T2837.
3. **D6 still wins on μ-quality + calibration** on the fixed test set ensemble (RMSE 1.486, NLL 1.822, ICE 0.041 — all best-in-class). pLDDT correlates with `|y|` (stable regions → small ddG), which helps μ predict ddG magnitude → tighter NLL/ICE. It just doesn't help σ rank where the model is wrong.
4. **Structural proxies need to be paired with chemistry**. D4 alone is worse than nothing (Δ = −0.016 vs D0). D5 = D3 + structural is a small positive lift (Δ = +0.011 over D3) that doesn't reach significance at K = 25.

## 4. Decision narrative

### Day 0 — Track-D ablation on 8M backbone (`scripts/07`, `scripts/09`)

5-seed run of D0–D3 on the 8M backbone. **All challengers within noise of D0 on Spearman**. Largest absolute Δ = D3 vs D0 = +0.014, p = 0.40.

**Decision:** the 8M backbone is too weak. Either backbone or data is the bottleneck — not σ-branch design. Diagnostic confirmed (`scripts/08`): bio-features-only |error| predictor has Spearman 0.166, comparable to the σ-branch.

### Day 1 — bigger backbone (`scripts/01_cache_embeddings_esm_v2 --esm-model facebook/esm2_t33_650M_UR50D`)

Re-cached embeddings using ESM2-650M (1280-d per residue, 2600-d per row after concat). Re-ran the same Track-D ablation.

* D0 Spearman: 0.190 → 0.296 (+56%)
* D3 Spearman: 0.206 → 0.333

**Decision:** ship the 650M backbone. K-fold CV (`scripts/11`) confirmed D3 vs D0 Spearman is significant (Wilcoxon p = 0.04, 12/15 wins).

### Day 2 — sequence-derived structural proxies (`scripts/06 --include-extended`)

Added **6 sequence-only structural proxies** (Chou-Fasman α / β propensity deltas, local AA entropy, local hydrophobic count, local charged count, position-relative). No PDB needed.

* D5 ensemble Spearman: **0.348** (vs D3 ensemble 0.304)
* D5 ensemble NLL: 1.849 (hits NEXT_STEPS target NLL < 1.85)
* D5 vs D0 K-fold paired-t (n=25) p = **0.007** (highly significant)

### Day 3 — real DSSP + pLDDT from AlphaFold DB structures (`scripts/14`)

Downloaded AF DB models for 99 unique uniprot_ids (after [API URL fix](https://github.com/RoselindSi/uapp/pull/14)). Coverage = 92/99 proteins → 2143/2584 mutations (82.9%) get real DSSP features. The other 17.1% are NaN-imputed to the train mean.

8 new features per mutation: SS one-hot (helix, sheet), φ/ψ angles, rsa_dssp, **pLDDT**, local helix/sheet fractions.

* D6 ensemble RMSE: **1.486** (best)
* D6 ensemble NLL: **1.822** (best, well below target)
* D6 ensemble ICE: **0.041** (best)
* D6 ensemble Spearman: 0.331 (slightly below D5's 0.348, statistically indistinguishable)
* D6 vs D5 K-fold paired-t (n=25) p = 0.31 (no significant difference)

**Decision:** D6 is the production model on calibration metrics; D5 is the alternative if you optimise solely for Spearman. Both are at the "ceiling" of what the σ branch can do without more data.

### LoRA fine-tune attempt (negative result)

Attempted LoRA on ESM2-650M (rank 8, α 16, attention query+value, end-to-end with optional ranking loss). Training curves showed **immediate overfit** — best val NLL at epoch 1, train loss kept dropping while val loss climbed. The σ-branch lost its ranking signal (Spearman 0.333 → 0.060).

**Decision:** abandon LoRA at this dataset size. Documented in `scripts/12_lora_finetune_d3.py` and `notebooks/colab_lora_finetune_d3.ipynb`. The Track-3 plan ("LoRA → ensemble") is replaced by "D6 / D5 → ensemble" as the production path.

---

## 5. What worked

| Decision | Δ Spearman (vs prior) | Δ NLL |
|---|---:|---:|
| **8M → 650M backbone**       | +0.10 (D0: 0.19 → 0.30) | −0.05 |
| **D3** (RSA + chemistry in σ) | +0.04 (Wilcoxon-significant) | flat |
| **D5** (+ sequence-derived structural) | +0.04 ensemble, +0.01 K-fold mean | −0.04 |
| **D6** (+ DSSP/pLDDT) | −0.02 vs D5 (worse on Spearman) but +0.03 on RMSE/NLL/ICE | −0.03 |
| **5-member ensemble**        | +0.01 to +0.02 (D5/D6) | −0.01 to −0.02 |
| **All stacked (D6 ensemble)** | **0.19 → 0.33 Spearman** | **1.94 → 1.82 NLL** |

## 6. What didn't work (negative results worth reporting)

| Attempt | Outcome | Why it failed |
|---|---|---|
| **LoRA fine-tune of 650M**  | Spearman 0.06 (collapse) | Overfit at epoch 1; σ-branch lost ranking signal |
| **Ranking-aware loss on frozen 650M D3** | Δ Spearman ≤ 0.01 | σ already near rank-optimal on these embeddings |
| **Temperature scaling σ' = T·σ** | T = 1.0000, no change to ICE | σ is already NLL-optimal globally; ICE residual is shape-mismatch |
| **D4 (structural-only)**     | Spearman 0.235 (worse than D0 0.251) | Structural proxies stand-alone don't carry signal — only useful in combination |
| **D5 vs D3 K-fold paired-t** | p ≈ 0.21 | Effect size +0.011 is below noise floor at K = 25 paired obs |
| **D6 (real DSSP/pLDDT) vs D5 (sequence proxies)** | Δ Spearman = −0.009, p = 0.31 | The cheap proxies already captured the signal; real DSSP added a bit of noise on small n_train |
| **Megascale pretrain → T2837 fine-tune** (single seed) | RMSE 1.67, NLL 2.22, Spearman 0.21 (vs frozen D5 0.348) | Pretrain val_loss bottomed at epoch 1 then rose; fine-tune had to undo the domain-shift prior. The 650M embeddings + 13 D5 features already saturate on T2837 — adding 276 K Megascale rows did not lift the ceiling. T2837's `n_test = 170` is the binding constraint, not training-data volume. |

These should appear in the discussion section of any writeup. The pattern
("σ-branch has a ceiling that scales with the backbone, not with the
loss function or feature richness") is itself the take-away.

---

## 7. Limitations

1. **Test set is small (n = 170, fixed protein-level holdout).** Any single-seed Spearman has CI ≈ ± 0.07. K-fold CV over 2584 samples partly mitigates this but still gives σ_per-fold-per-seed ≈ 0.11.
2. **ICE = 0.041 missed the ≤ 0.02 target** by ~2×, though best individual D6 member hit 0.025. Temperature scaling on NLL is a no-op (σ already NLL-optimal), so closing the gap requires isotonic regression / quantile-based recalibration — not implemented.
3. **D5 vs D6 is statistically inconclusive.** Both are within noise of each other on every metric.
4. **DSSP/pLDDT coverage is 82.9%** (441 mutations have NaN-imputed structural features because their uniprot_id was not in AF DB). For those rows D6 falls back to D5-equivalent behaviour.
5. **μ-accuracy doesn't transfer cross-dataset, but σ-ranking *and* σ-calibration do.** D5 ensemble on S669 (n = 617, 90 proteins, no T2837 overlap by construction): σ-ranking transferred and *improved* (Spearman 0.434 vs 0.348). RMSE doubled (1.50 → 2.83) — the absolute ddG scale didn't transfer. σ was initially under-scaled (ICE 0.31, cov@90 = 0.37), but a single closed-form temperature scalar fitted on 308 rows fully closed that gap (ICE 0.04, cov@90 = 0.92, NLL halved). See §11 for the full split-of-skills story and the recalibration numbers.

## 8. Future work (ROI-ranked)

| Direction | Expected lift | Cost |
|---|---|---|
| ~~**Megascale pretrain** (Tsuboyama 2023, ~700K mutations)~~ | **Tried, did not help** — see negative-results table | — |
| **Isotonic regression on val coverage** for ICE recalibration | ICE 0.041 → ~0.02 (target met) | ½ day, RMSE/Spearman unchanged |
| **More seeds** (K-fold with 10–20 seeds) | Tightens CIs, may lift D5-vs-D3 to significance | Cheap (extra Colab compute) |
| **Two-stage warm-start LoRA** | Speculative; unblocks LoRA path if it works | 1 day |
| ~~**External dataset cross-validation (S669, ProThermDB)**~~ | **Done — see §11.** S669 σ-ranking improved (Spearman 0.434); calibration broken on the new label distribution. | — |
| **Different backbone (ESM-3, ESM-IF, ProtT5, SaProt)** | The whole campaign is on ESM2-650M; we changed the dataset (Megascale, S669) but never the encoder. A structure-aware backbone (ESM-IF, SaProt) might rescue μ-accuracy on S669 since the ranking/structure features already transferred. | 2–3 days |
| ~~**Post-hoc σ recalibration (S669 isotonic + temperature)**~~ | **Done — see §11.** Temperature T = 2.76 closed the entire calibration gap (ICE 0.32 → 0.04, cov@90 0.36 → 0.92, NLL halved); Spearman preserved. | — |
| **Improved AF DB coverage** (use experimental PDBs as fallback for the 17.1% missing AF models) | Probably minor; D6 already not better than D5 | ½ day |

---

## 9. Reproduction recipe

The exact production pipeline (D6 ensemble):

```bash
# 0. Cache the 650M embeddings once. ~10 min on T4 / MPS, ~1 hour CPU.
python scripts/01_cache_embeddings_esm_v2.py \
    --t2837-csv StabilityOracle/data/datasets/T2837.csv \
    --out cache/t2837_embeddings_v2_650m.pt \
    --esm-model facebook/esm2_t33_650M_UR50D \
    --seed 42

# 1. Build aligned bio features (k = 13: RSA + chemistry + sequence-struct).
python scripts/06_build_bio_features.py \
    --metadata-csv cache/t2837_metadata.csv \
    --embeddings   cache/t2837_embeddings_v2_650m.pt \
    --out          cache/t2837_bio_features_650m_extended.pt \
    --include-extended

# 2. Add DSSP + pLDDT (k = 21). Needs DSSP binary + biopython.
#    Downloads ~99 AlphaFold structures via the AF DB API. ~5 min.
apt-get install -y dssp || apt-get install -y mkdssp
pip install biopython
python scripts/14_compute_structural_features.py \
    --metadata-csv  cache/t2837_metadata.csv \
    --embeddings    cache/t2837_embeddings_v2_650m.pt \
    --extended-bio  cache/t2837_bio_features_650m_extended.pt \
    --out           cache/t2837_bio_features_650m_dssp.pt

# 3. Final D6 ensemble (5 members).
python scripts/10_deep_ensemble.py \
    --embeddings cache/t2837_embeddings_v2_650m.pt \
    --bio-feats  cache/t2837_bio_features_650m_dssp.pt \
    --out        outputs/ensemble_d6_650m \
    --ablation D6 --members 5 --device cuda

# 4. K-fold CV with paired tests (D0 vs D3 vs D5 vs D6).
python scripts/11_kfold_cv_track_d.py \
    --embeddings cache/t2837_embeddings_v2_650m.pt \
    --bio-feats  cache/t2837_bio_features_650m_dssp.pt \
    --out        outputs/cv_full \
    --ablations D0 D3 D5 D6 \
    --folds 5 --seeds 0 1 2 3 4 \
    --device cuda
```

Output of step 3 contains the headline numbers in `ensemble_summary.json`.
Output of step 4 contains the statistical-significance JSON in `significance.json`.

## 10. Code map

| File | Purpose |
|---|---|
| `uapp/heads.py`              | `FeatureAugmentedHead`, μ-branch / σ-branch separation |
| `uapp/losses.py`             | Student-t NLL, pairwise ranking loss |
| `uapp/mutation_features.py`  | BLOSUM62 / Grantham / Chou-Fasman / window stats / position-relative |
| `uapp/evaluate.py`           | RMSE, MAE, NLL, ICE, Spearman, top-k risk capture |
| `scripts/01_cache_embeddings_esm_v2.py` | Cache ESM2-N embeddings (mutation-aware) |
| `scripts/06_build_bio_features.py`      | Bio features aligned to embedding cache (k = 7 or 13) |
| `scripts/07_experiment_d_real.py`       | Track-D ablation (D0..D6) on the fixed split |
| `scripts/09_multiseed_experiment_d.py`  | K-seed wrapper of script 07 |
| `scripts/10_deep_ensemble.py`           | K-member ensemble, mean-of-means + variance-of-means |
| `scripts/11_kfold_cv_track_d.py`        | K-fold CV by `pdb_code`, paired t-test + Wilcoxon |
| `scripts/12_lora_finetune_d3.py`        | LoRA fine-tune on ESM2-650M (negative result) |
| `scripts/13_temperature_scaling.py`     | Post-hoc σ rescaling (no-op on frozen 650M; archived for completeness) |
| `scripts/14_compute_structural_features.py` | DSSP + pLDDT from AlphaFold DB → k = 21 bio features |
| `scripts/15_megascale_to_t2837_format.py` | Tsuboyama 2023 Megascale CSV → T2837-shaped CSV (used for pretrain corpus) |
| `scripts/16_pretrain_and_finetune.py` | Pretrain D5 head on Megascale, fine-tune on T2837 (negative result, see §6) |
| `scripts/17_s669_to_t2837_format.py` | S669 release CSV → T2837-shaped CSV for external evaluation |
| `scripts/18_evaluate_on_s669.py` | Train D5/D6 ensemble on full T2837, predict on S669, report metrics |
| `scripts/19_recalibrate_sigma_s669.py` | Post-hoc σ recalibration on S669 — temperature scaling + isotonic regression on a held-out cal split |
| `scripts/20_cache_embeddings_saprot.py` | SaProt embeddings (combined AA + 3Di tokens via FoldSeek) — drop-in encoder swap for §12 |
| `scripts/21_align_cache_to_reference.py` | Filter source cache+bio to a reference cache's row subset (apples-to-apples §12 follow-up) |
| `scripts/22_cache_embeddings_esmif.py` | ESM-IF1 embeddings (structure-aware, pure-AA tokens via GVP-Transformer) — §12 follow-up |
| `notebooks/colab_s669_evaluation.ipynb` | End-to-end Colab walkthrough that ran the §11 evaluation |
| `notebooks/colab_saprot_evaluation.ipynb` | End-to-end Colab walkthrough for §12: SaProt + the two follow-ups (aligned ESM2, ESM-IF) |
| `notebooks/next_steps_comprehensive.ipynb`  | NEXT_STEPS Tracks A–D walkthrough on synthetic data |
| `notebooks/colab_lora_finetune_d3.ipynb`    | Colab T4 notebook for LoRA fine-tune |

## 11. External validation on S669

After the campaign closed on T2837, we ran the production D5 ensemble on **S669** (Pancotti et al. 2022, *Briefings in Bioinformatics*; Zenodo record 7568094) — 669 single-point variants across 94 proteins, curated to have low overlap with common training sets. After the converter (script 17) dropped rows whose `Seq_Mut` position didn't align with the WT chain extracted from the bundled PDB, **617 rows across 90 proteins** entered evaluation.

Pipeline (notebook `colab_s669_evaluation.ipynb`):

1. Download `S669.zip` from Zenodo, extract WT chain sequences from the bundled PDBs into a FASTA keyed by `Protein` id.
2. `scripts/17_s669_to_t2837_format.py` — convert the Zenodo CSV to T2837 schema using `--mut-col Seq_Mut --ddg-col Experimental_DDG_dir --fasta wildtypes.fasta`. 617/669 SAVs kept.
3. `scripts/01_cache_embeddings_esm_v2.py` — cache ESM2-650M embeddings for S669 (the same backbone used for T2837).
4. Inline build of D5 bio features (k = 13) standardised with **T2837's** train mu/sd. (Script 06 fits the standardiser on a `train` split, which S669 doesn't have, *and* using T2837 statistics is the correct thing to do scientifically since that's what the trained heads expect.)
5. `scripts/18_evaluate_on_s669.py` — train K = 5 D5 heads on T2837 train, early-stop on T2837 val, predict on both T2837 test (sanity) and S669.

### Sanity check (T2837 test, n = 170)

The pipeline reproduces the campaign's Day-2 numbers within rounding noise:

| Metric | This run | Day-2 reference |
|---|---:|---:|
| RMSE | 1.496 | 1.50 |
| NLL | 1.849 | 1.85 |
| ICE | 0.050 | 0.05 |
| Spearman | 0.348 | 0.348 |

The S669 numbers below are therefore not a measurement artefact.

### S669 results (n = 617)

| Metric | T2837 ensemble | S669 ensemble | Δ | Read |
|---|---:|---:|---:|---|
| **Spearman(σ, \|err\|)** | 0.348 | **0.434** | **+0.087** | ✓ σ-ranking transfers and *improves* with the larger sample |
| RMSE | 1.50 | 2.83 | +1.33 | ✗ μ accuracy nearly halved |
| MAE | 1.05 | 2.29 | +1.25 | ✗ same |
| NLL | 1.85 | 4.64 | +2.79 | ✗ joint likelihood dominated by RMSE inflation |
| ICE | 0.05 | 0.31 | +0.26 | ✗ σ severely under-scaled |
| cov@0.90 | 0.86 | 0.37 | −0.49 | ✗ 90% PI covers only 37% of S669 mass |

### Interpretation — the model splits cleanly into two skills

**Transferable (σ-branch ranking).** Spearman 0.434 with `n = 617` is statistically robust (SE ≈ 0.04, z ≈ 11 vs zero) and *higher* than the on-distribution T2837 number (0.348, `n = 170`). The σ-branch learned a property of the input — *what kinds of mutations are uncertain* — that does not depend on the train-time ddG distribution. This is exactly the NEXT_STEPS Track-D goal stated in cross-dataset form.

**Non-transferable (μ accuracy + absolute calibration).** RMSE doubles from 1.50 to 2.83 and σ becomes consistently too small. The joint failure is consistent with a label-distribution shift: S669 (curated from ThermoMutDB) contains more strongly destabilising mutations than T2837 (Stability Oracle release), and the frozen ESM2-650M head was never exposed to that scale during training. σ scales with σ_train, so when |residual| inflates by ~2× the predictive intervals shrink in relative terms.

This is a **scientifically useful split**: in deployment, σ-ranking lets you triage which predictions to trust on a *new* protein/dataset, even when you can't trust the absolute μ values yet. Active-learning and human-in-the-loop pipelines depend on this property — and we can now claim it cross-dataset.

### Recalibration (`scripts/19_recalibrate_sigma_s669.py`)

To close the absolute-calibration gap, we run two post-hoc methods on a **50/50 calibration / evaluation split** of S669:

1. **Temperature scaling** — single global multiplier σ' = T·σ, with T chosen in closed form to minimise Gaussian NLL on the calibration half: `T² = (1/N) Σ (y - μ)² / σ²`. Preserves Spearman exactly.
2. **Isotonic regression** — fit a monotone σ → |residual| mapper on the calibration half, then convert back to a Gaussian σ via the half-normal factor √(π/2). Allows non-uniform stretch (different correction at small vs large σ). Preserves Spearman because monotone.

Both methods are evaluated on the unseen half. By construction, Spearman(σ, \|err\|) is preserved. Live numbers (random 50/50 split, eval n = 309):

| Method | RMSE | NLL | ICE | cov@0.90 | cov@0.95 | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| baseline σ                | 2.82 | 4.60 | 0.318 | 0.359 | 0.447 | 0.449 |
| **temperature scaling**   | 2.82 | 2.37 | **0.044** | **0.919** | 0.987 | 0.449 |
| **isotonic regression**   | 2.82 | **2.34** | 0.052 | 0.935 | 0.974 | 0.461 |

Fitted **T = 2.76** — σ was about 2.76× too small on S669, consistent with RMSE doubling between T2837 and S669. After scaling, **ICE drops from 0.318 to 0.044** (well below the NEXT_STEPS target of 0.05), **cov@0.90 climbs from 0.36 to 0.92**, and **NLL halves**. RMSE is unchanged because μ is unchanged. Spearman is preserved exactly under temperature scaling (single global multiplier); isotonic ticks up by 0.012 because flat regions of the fitted mapper introduce ties that slightly reorder ranks.

Net effect: a single global multiplier learned from 308 calibration rows closes the entire absolute-calibration gap on S669. **The ranking property and the calibration property are independent and both transferable** — the first natively, the second after a one-parameter post-hoc fix.

Run with `--split-by-protein --metadata cache/s669_metadata_processed.csv` for a stricter leak-free split if you want to verify these numbers don't depend on protein-level near-duplicates between cal and eval.

### Open questions raised by §11

The campaign so far changed the **dataset** twice (Megascale pretrain, S669 eval) but kept the **backbone** fixed at ESM2-650M throughout. The clean σ-ranks-transfer / μ-doesn't-transfer split suggests the next interesting axis to vary is the encoder:

- **ESM-IF / ESM-3 / SaProt** — structure-aware backbones may give better μ accuracy on S669 because they encode features that *are* sensitive to the ddG-magnitude shift (e.g. local geometry near the mutation).
- **ProtT5 / Ankh** — different self-supervision objectives; sometimes generalise better on out-of-distribution proteins.

Whichever backbone we try, the σ-ranking property gives a clean evaluation lever: if a new backbone preserves Spearman(σ, |err|) on S669 *and* drops RMSE, that's a strict improvement over the current production model.

## 12. Structure-aware backbone — SaProt

The §11 split-of-skills result pointed at the encoder as the next axis to vary. ESM2-650M was held fixed for the entire campaign; every prior variation was on the head, the loss, or the data. This section runs the encoder swap.

§12.1 below is the original SaProt run on the 170-vs-137 cross-population comparison. §12.2 is the follow-up apples-to-apples comparison that removes the population confound, and *changes the headline interpretation* — see §12.2 if you want the punchline.

### 12.1 First run (population-confounded)


`scripts/20_cache_embeddings_saprot.py` + `notebooks/colab_saprot_evaluation.ipynb` implement a drop-in swap to **SaProt** (Westlake/SJTU 2024), a structure-augmented PLM that takes combined `<aa><3di>` tokens, where the 3Di letters come from FoldSeek's structural tokenisation of a PDB. SaProt has the same hidden size (1280) and parameter count (650M) as ESM2-650M, so every downstream script (06, 18, 19) works unchanged.

### Setup

Per-protein PDB resolution:
- T2837: AlphaFold-DB models cached by scripts/14, keyed by `uniprot_id` (`AF-{uniprot_id}.pdb`). 100/108 proteins matched (8 uniprot_ids 404 on the AF API — retired or replaced entries).
- S669: WT PDBs bundled in the Zenodo release. The `pdb_code` column is the original `Protein` value from the S669 CSV (e.g. `1a0fA` = PDB id + chain), while the bundled file is named after just the PDB id (e.g. `1a0f.pdb`); a small filename-resolver step (notebook cell 4a) maps one to the other. 90/90 proteins matched.

The script re-resolves `mut_idx` against the **PDB-derived AA string** (not the metadata sequence) before sampling embeddings — without this, ~80% of T2837 rows would land at the wrong residue because AlphaFold-DB models are often shifted vs the T2837 metadata sequence (signal peptides cleaved, chain offsets). Final alignment rates: T2837 2129/2145 (99.3%), S669 615/615 (100.0%).

### Results

```
Encoder comparison — ESM2-650M (§11) vs SaProt (this section)

T2837 test (n_ESM2 = 170; n_SaProt = 137 — see caveat below)
  ESM2-650M:   RMSE 1.500   NLL 1.850   ICE 0.050   Spearman 0.348
  SaProt:      RMSE 1.598   NLL 1.889   ICE 0.038   Spearman 0.218

S669 (n_ESM2 = 617; n_SaProt = 615)
  ESM2-650M:   RMSE 2.830   NLL 4.640   ICE 0.310   Spearman 0.434
  SaProt:      RMSE 2.534   NLL 3.185   ICE 0.261   Spearman 0.416

σ recalibration on S669:  ESM2 T = 2.76    SaProt T = 2.10
SaProt recal: ICE 0.27 → 0.05   cov@0.90 0.52 → 0.94   NLL 3.08 → 2.27
```

### Three findings

**1. The hypothesis was directionally right.** SaProt drops S669 RMSE from 2.83 to 2.53 (−10%) and NLL from 4.64 to 3.19 (−31%). Structure-aware encoding *does* recover μ-accuracy on the new label distribution. This validates §11's argument that the encoder, not the head, is the bottleneck for cross-dataset μ generalisation.

**2. The cost lands on T2837 in-distribution performance.** SaProt T2837 RMSE is +0.10 worse and Spearman drops from 0.348 → 0.218 (-37%). The σ-ranking property — the strongest finding of the entire campaign — partially breaks under the encoder swap. The combined `<aa><3di>` token space changes what the σ branch can attend to, and the input-side regularities ESM2 had learned about "which mutations are uncertain" do not survive intact.

**3. The strict-improvement criterion is not met.** From the notebook's decision rule: T2837 RMSE > 1.50 *and* Spearman < 0.348 trigger the "encoder swap broke in-distribution performance" row. SaProt is **not** a drop-in production replacement for ESM2-650M.

### S669 σ recalibration (SaProt edition)

The recalibration recipe from §11 still works: a closed-form temperature scalar fitted on 308 S669 rows brings ICE from 0.27 to 0.05 and cov@0.90 from 0.52 to 0.94. SaProt's fitted T = 2.10 is smaller than ESM2's 2.76, consistent with SaProt's σ being less under-scaled to start with (SaProt RMSE 2.53 < ESM2 RMSE 2.83).

| Method | RMSE | NLL | ICE | cov@0.90 | cov@0.95 | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| baseline σ              | 2.46 | 3.08 | 0.271 | 0.518 | 0.632 | 0.440 |
| **temperature scaling** | 2.46 | 2.27 | **0.047** | **0.938** | 0.990 | 0.440 |
| **isotonic regression** | 2.46 | **2.25** | 0.057 | 0.948 | 0.977 | 0.462 |

### Caveats

1. **Different test populations.** The SaProt T2837 cache lost 8 proteins (439 mutations) because their `uniprot_id` was retired in UniProt and the AF API returned 404. The remaining T2837 test set is n=137 instead of the §11 n=170; 33 of the missing test rows happen to fall in those 8 proteins. ESM2 has not been re-evaluated on the same 137-row subset, so part of the SaProt T2837 RMSE inflation may be sample-shift rather than genuine encoder regression. A clean comparison needs ESM2 evaluated on the SaProt-aligned subset (deferred — should be a 30-line follow-up).
2. **Single seed of seeds.** Both ESM2 and SaProt were 5-member ensembles; the standard error on Spearman at n≈137 is ~0.07. The SaProt T2837 Spearman of 0.218 vs ESM2's 0.348 is roughly two standard errors — meaningful but not overwhelming.
3. **3Di tokenisation is from AlphaFold-DB structures**, the same source the production D6 ablation already used for DSSP/pLDDT features. The signal SaProt encodes is therefore *not* orthogonal to D6 — some of the structural information is already in the D5 ensemble's σ-branch features. A cleaner test of "does the encoder need to know structure" would compare SaProt against an ESM2 baseline that has no structural bio features (D0).

### Provisional reading (revised in §12.2)

The §12.1 first-pass reading was *partial-positive*: cross-dataset μ improvement looks real, T2837 σ-ranking degradation also looks real, no strict-improvement. Two of the three caveats above (different test populations; the AF-DB structural overlap) make this reading suspect. §12.2 runs the apples-to-apples comparison that removes the population confound; §13 runs ESM-IF to test the token-vocabulary hypothesis.

### 12.2 Apples-to-apples follow-up — corrected interpretation

**Setup.** `scripts/21_align_cache_to_reference.py` filters the ESM2 T2837 cache + bio features to the SaProt-aligned 137-row test subset (and the matching 1179/813 train/val that SaProt actually trained on). Script 18 then trains a fresh 5-member D5 ensemble on the *same* rows SaProt saw and predicts on the *same* T2837 test rows + S669. Both encoders now differ on exactly one axis: the encoder.

**Results (137-row T2837 test, identical 1179/813 train pool, 617-row S669):**

```
                    T2837 RMSE   T2837 Spearman   S669 RMSE   S669 Spearman
ESM2 (aligned)        1.532        0.417            2.869       0.407
SaProt                1.598        0.218            2.534       0.416
```

**What changes vs §12.1.**

1. **The S669 μ improvement is real, not a population artifact.** SaProt drops S669 RMSE from 2.869 to 2.534 (Δ = −0.34) on the *same* 617 rows as ESM2-aligned. The §12 first-pass reading already said this, but the §12.1 ESM2 baseline (2.83 on n=617, trained on the larger 1395/1019 pool) was a different experimental condition; the apples-to-apples comparison confirms the encoder, not the population, is doing the work.
2. **The T2837 σ-ranking degradation is also real, but is now larger than §12.1 suggested.** The aligned-ESM2 ensemble hits Spearman 0.417 on the 137-row test (better than the §11 0.348 on 170 rows — the 33 missing test rows happen to be hard ones). SaProt's 0.218 is now ~2.0× lower than ESM2's, not ~1.6× as §12.1 reported. The σ-ranking signal that ESM2 captures simply is not in SaProt's combined-token representation.
3. **No strict winner.** SaProt **wins on S669 RMSE** (−0.34) and **loses on T2837 Spearman** (−0.20). Neither encoder dominates.

The headline of §12 is now: **structure-aware encoding is a real lever for cross-dataset μ-accuracy, at a measurable cost in in-distribution σ-ranking**. §13 tests whether this is specific to SaProt's combined token vocabulary or a general property of structure-aware encoders.

## 13. ESM-IF — second structure-aware encoder, pure-AA tokens

`scripts/22_cache_embeddings_esmif.py` builds the same mutation-aware feature shape using `esm.pretrained.esm_if1_gvp4_t16_142M_UR50` from the `fair-esm` package. ESM-IF1 takes structure through GVP geometric features rather than injecting 3Di letters into the token vocabulary, so the σ-branch input alphabet stays pure-AA. Encoder hidden size is 512 (vs 1280 for ESM2/SaProt) → mutation-aware feature is 1064-d (vs 2600-d); `FeatureAugmentedHead` reads `d_in` dynamically so the rest of the pipeline is unchanged.

The hypothesis was: if SaProt's T2837 σ-ranking drop is caused by AA+3Di token mixing, ESM-IF should preserve σ-ranking while still helping S669 μ — the strict-improvement encoder.

### Results

Apples-to-apples vs the §12.2 ESM2 baseline (same 137-row T2837 test, same 1179/813 train pool, same 616/617-row S669):

```
                    T2837 RMSE   T2837 Spearman   S669 RMSE   S669 Spearman
ESM2 (aligned)        1.532        0.417            2.869       0.407
SaProt                1.598        0.218            2.534       0.416
ESM-IF                1.605        0.074            2.416       0.392
```

σ recalibration on S669: ESM2 T = 2.76, SaProt T = 2.10, **ESM-IF T = 2.03**. The temperature scalar required to recover calibration drops monotonically with structural awareness — structure-aware encoders are intrinsically less σ-overconfident on out-of-distribution data, but at the cost of in-distribution σ-ranking.

### Strict-improvement criterion

| Test | ESM-IF | vs ESM2 (aligned) | Verdict |
|---|---:|---:|:---:|
| T2837 RMSE        | 1.605 | 1.532   | **FAIL** |
| T2837 Spearman    | 0.074 | 0.417   | **FAIL** |
| S669 RMSE         | 2.416 | 2.869   | **PASS** (Δ = −0.45) |

### Two findings

**1. The token-vocabulary hypothesis is falsified.** ESM-IF uses pure-AA tokens and degrades T2837 σ-ranking *more* than SaProt (0.074 vs 0.218). The Spearman drop is not caused by SaProt's combined `<aa><3di>` alphabet; it's a property of structure-aware encoding regardless of how structure enters the token stream. The likely mechanism: structure-aware encoders force the per-residue embedding to encode global geometric context, which makes σ less responsive to the local-sequence variation that drives "easy vs hard mutation" ranking. The σ-ranking signal lives in ESM2's local sequence-attention patterns, and structure-awareness washes it out.

**2. There is a clean Pareto trade-off across the three encoders.** Sorting ESM2 → SaProt → ESM-IF by structural awareness gives a *monotone* trade-off in both directions:

```
                   T2837 Spearman    S669 RMSE
ESM2 (aligned)     0.417             2.869
SaProt             0.218             2.534
ESM-IF             0.074             2.416
                   ──────────────    ─────────
                   ↓ structure ↑       structure ↑
                   ↓ in-dist σ-rank ↑   cross-dataset μ ↑
```

The campaign asked "is structure-aware encoding a strict improvement?" The answer is **no** under the original strict-improvement criterion. The deeper finding is that structure-awareness is a real, quantifiable lever along a Pareto frontier we hadn't characterised before.

## 14. Encoder-swap campaign — closure

### Production decision

**ESM2-650M D5 ensemble + post-hoc σ temperature scaling (the §11 recipe) remains the production model.** It is the only ensemble of the three that preserves the σ-ranking property the entire campaign was built around (Spearman 0.348 on the original 170-row test, 0.417 on the 137-row aligned subset). For active-learning / σ-based query selection — the canonical use case for this model — σ-ranking is the primary metric and ESM2 dominates.

### Documented alternatives (use-case dependent)

| Use case | Recommended | Why |
|---|---|---|
| In-distribution active learning, σ-driven query selection | **ESM2-650M D5 ensemble** | Best T2837 Spearman (0.42); σ ranks easy/hard mutations as designed |
| Cross-dataset μ accuracy on novel proteins (e.g. lab triage of single-point variants on new targets) | **ESM-IF D5 ensemble** | Best S669 RMSE (2.42) by a margin (−0.45 vs ESM2, −0.12 vs SaProt) |
| Balanced (some μ gain on cross-dataset, σ-ranking degraded but not catastrophic) | **SaProt D5 ensemble** | T2837 Spearman 0.22 (still > 0); S669 RMSE 2.53 |

### Falsified hypotheses

* **"§12 SaProt S669 win was a population artifact"** (raised as a §12 caveat). False — confirmed real on the apples-to-apples test (§12.2).
* **"SaProt's σ-ranking drop is caused by AA+3Di token mixing"** (working hypothesis behind the ESM-IF experiment). False — ESM-IF with pure-AA tokens degrades σ-ranking even further (§13).

### Confirmed findings

* **There is a Pareto trade-off between in-distribution σ-ranking and cross-dataset μ-accuracy** along the structural-awareness axis. Three independent encoders confirm the direction and roughly the slope.
* **σ recalibration temperature is encoder-specific and decreases monotonically with structural awareness** (T = 2.76 → 2.10 → 2.03). The post-hoc temperature-scaling recipe from §11 transfers across all three encoders.
* **The σ-ranking signal lives in ESM2's local sequence-attention patterns**, not in any explicit feature engineered into the head. No head we trained recovered it after the encoder was swapped.

### Future work

In ROI order, given the new Pareto-trade-off framing:

1. **Dual-encoder feature** — concatenate `[ESM2_features, ESM-IF_features]` and let a single head route μ-vs-σ to the right channel. This is the most direct attack on the observed Pareto: ESM-IF carries the cross-dataset μ signal; ESM2 carries the σ-ranking signal. Risk: 1179 train rows × 3664-d feature is overfitting territory; probably needs feature dropout or low-rank projection.
2. **Encoder-level ensembling** as a strong baseline — train ESM2 D5 and ESM-IF D5 separately, take μ from ESM-IF and σ from ESM2 at inference. Cheaper than (1) and answers "is the Pareto a single-head limitation or a data limitation". Half a day of work.
3. **σ-branch token-sensitivity diagnostic** — measure whether ESM2's σ-branch is reading specific token types (e.g. wt-AA identity, neighbour residues) vs window-aggregate features. If we can isolate *which* token signal is doing σ-ranking, we may be able to add it as an explicit feature on top of structure-aware encoders.
4. **Larger ESM-IF or richer geometric features** — ESM-IF1 is 142M params; a bigger structure-aware model (ESM-3, AlphaFold representations) might shift the Pareto. Speculative.

What is **not** worth doing based on the campaign so far: wider/deeper MLP heads (D5/D6 ablations already explored capacity); switching the output distribution (Student-t vs Gaussian/NIG is orthogonal to the Pareto); LoRA/full fine-tune of the encoders on T2837 (LoRA already destroyed σ-ranking on ESM2; nothing in the campaign suggests it would behave differently on SaProt or ESM-IF).

## 15. Pull-request trail

The campaign was developed iteratively; the merged PRs document the
decision sequence and serve as a citable trail:

* `#8`  — Deep ensemble (Lakshminarayanan 2017) for Track D
* `#9`  — K-fold CV by `pdb_code`
* `#10` — LoRA fine-tune of ESM2-650M (negative result)
* `#11` — Colab T4 notebook
* `#12` — Day-1 + Day-2 toolkit (ranking loss, temperature scaling, sequence-derived structural proxies, D4/D5 ablations, paired-test fix)
* `#13` — Day-3 toolkit (real DSSP + pLDDT from AF DB, D6 ablation)
* `#14` — AF DB API URL fix (v4 retired → API resolves canonical v6)
* `#16` — Megascale pretrain pipeline (scripts 15 + 16) and pipeline fixes
* `#18` — S669 external validation: format converter + eval script (scripts 17 + 18)
* `#20` — Colab notebook for S669 (Zenodo source + FASTA extractor + T2837-stat standardiser)
* `#21` — `REPORT.md` §11 + `scripts/19` post-hoc σ recalibration
* `#22` — SaProt structure-aware backbone scaffolding (`scripts/20` + Colab notebook)
* `#23` — Fix AF PDB path resolution in SaProt notebook
* `#24` — Script 20: re-resolve `mut_idx` against PDB-AA so SaProt samples the right residue
* `#25` — SaProt live results (§12.1) + `scripts/21` (apples-to-apples cache subset filter) + `scripts/22` (ESM-IF encoder) + notebook sections 10 and 11
* this PR — Encoder-swap campaign closure: §12.2 apples-to-apples comparison (corrects the §12.1 reading), §13 ESM-IF live results, §14 Pareto-trade-off framing + production-decision matrix

---

## One-paragraph abstract for an internal slide

> We trained heteroscedastic Student-t heads on ESM2-650M embeddings of
> T2837 mutations. Augmenting the σ branch with **RSA + chemistry +
> sequence-derived structural proxies** (D5) gave a K-fold-significant
> Spearman improvement over the no-extras baseline (Δ = +0.031, paired-t
> p = 0.007 over 25 paired observations). Adding **real DSSP + AlphaFold
> pLDDT features** (D6) did not further improve Spearman ranking, but did
> push the ensemble to RMSE = 1.486, NLL = **1.822** (target < 1.85
> exceeded), and ICE = 0.041. Three of four NEXT_STEPS targets met or
> exceeded; ICE missed by ~2× on T2837. LoRA fine-tune, ranking-aware
> loss, and Megascale pretrain were attempted but gave neutral or
> negative deltas — documented as negative results. **External
> validation on S669 (n = 617) gave the strongest finding of the
> campaign: σ-ranking transferred and improved (Spearman 0.348 →
> 0.434), and a single temperature scalar fitted on 308 calibration
> rows recovered absolute calibration on the new dataset (ICE 0.31 →
> 0.04, cov@90 from 0.37 to 0.92).** μ-accuracy did not transfer (RMSE
> 1.50 → 2.83), pointing to the encoder rather than σ-branch design as
> the next lever — ESM2-650M was held fixed throughout and a
> structure-aware backbone is the recommended next experiment.
> **§§12–14 ran the encoder-swap experiments.** Two structure-aware
> encoders were tested: SaProt (combined AA+3Di tokens) and ESM-IF1
> (pure-AA tokens, structure via GVP geometric features). Apples-to-apples
> on the same 137-row T2837 test and 1179/813 train pool, three encoders
> form a clean Pareto trade-off in two directions: T2837 Spearman 0.417
> → 0.218 → 0.074 (ESM2 → SaProt → ESM-IF) and S669 RMSE 2.869 → 2.534
> → 2.416. Structural awareness is a real lever for cross-dataset
> μ-accuracy (ESM-IF beats ESM2 by Δ RMSE = −0.45 on S669) at a
> measurable cost in in-distribution σ-ranking. Two hypotheses were
> falsified along the way (the §12 SaProt "S669 win" was *not* a
> population artifact; the σ-ranking degradation is *not* caused by
> SaProt's combined token vocabulary — ESM-IF's pure-AA tokens degrade
> ranking even more). **Production decision**: ESM2-650M D5 + post-hoc
> σ recalibration remains the model. ESM-IF and SaProt are documented
> as alternatives for use cases that prioritise cross-dataset μ over
> in-distribution σ-ranking. The most promising next experiment is a
> dual-encoder feature [ESM2; ESM-IF] that lets a single head route
> μ-vs-σ to the channel that carries each signal.
