# Suggested Next Research Directions from Notebook Results

Based on the executed notebook outputs and summary tables in `figures/`, the strongest immediate direction is to improve **calibration while preserving (or improving) RMSE**.

## 1) Priority direction: robust heteroscedastic likelihood + post-hoc calibration

- `C_StudentT_v3` currently gives the best calibration (ICE `0.0209`) with competitive NLL (`1.8529`) and good RMSE (`1.4909`).
- `C_Laplace` slightly improves RMSE (`1.4859`) but worsens ICE (`0.0482`) versus `C_StudentT_v3`.
- The next step is to combine robust likelihood modeling with a light post-hoc recalibration stage to target both metrics.

### Proposed experiment

1. Train Student-t heads (`nu` in {3, 5, 8, learned}).
2. On validation only, fit one-parameter and two-parameter scaling of predictive variance:
   - `sigma' = a * sigma`
   - `sigma' = a * sigma + b`
3. Evaluate on held-out test split:
   - RMSE/MAE (must stay within +1% of base model)
   - NLL
   - ICE and coverage at {50, 80, 90, 95}%


## 2) Mentor-suggested direction: fixed variance, calibrate error through mean learning

- Try a constrained setup where predictive variance is held fixed (global constant or per-batch scheduled constant) and the model focuses on learning a mean that better tracks expected error patterns.
- This tests whether current calibration gains are mostly coming from variance flexibility versus better point estimates.

### Proposed experiment

1. Freeze uncertainty to a fixed value:
   - Variant A: global `sigma = c` tuned on validation NLL.
   - Variant B: weakly scheduled `sigma_t` (epoch-dependent, not input-dependent).
2. Retrain mean head with calibration-aware objective:
   - Base loss: MSE or Huber on `mu`.
   - Add residual-shape regularizer on validation trend (e.g., penalize bias across error bins or residue/mutation groups).
3. Compare to heteroscedastic Student-t baseline:
   - RMSE/MAE and signed-bias by subgroup
   - NLL/ICE and coverage
   - Whether fixed-variance models can match calibration after simple post-hoc scaling

## 3) Secondary direction: improve uncertainty ranking quality

- Deep ensemble members and the final ensemble are stable, but uncertainty-error correlation is near zero in notebook output (`corr(σ_ens, |error|) ≈ 0.0012`).
- This suggests the model can be calibrated globally yet still weak at ranking hard vs easy mutations.

### Proposed experiment

1. Add an auxiliary loss encouraging monotonic relation between predicted uncertainty and absolute residual (pairwise ranking or soft Spearman surrogate).
2. Compare against baseline Student-t using:
   - Spearman(`sigma`, `|error|`)
   - top-k risk capture (fraction of largest errors inside top-k uncertain predictions)
   - AURC / selective prediction curves.

## 4) Tertiary direction: structure-aware and mutation-aware uncertainty features

- Current notebooks include mutation-aware variants and RSA metadata analyses; next gain is likely from richer context features that modulate uncertainty, not only mean prediction.

### Proposed experiment

1. Add lightweight features to variance branch only:
   - RSA binning / solvent exposure
   - mutation type class (polar↔hydrophobic, charge flips)
   - local sequence context window statistics.
2. Keep mean branch unchanged to isolate uncertainty effect.
3. Ablate feature groups and report ICE/NLL deltas.

## Minimal, high-value 1-week plan

- **Day 1–2:** Student-t + variance scaling sweep.
- **Day 3:** fixed-variance + mean-calibration pilot (mentor direction).
- **Day 4:** ranking-aware auxiliary loss pilot.
- **Day 5:** uncertainty-feature ablation on best setting.
- **Deliverable:** one table with RMSE, NLL, ICE, coverage, bias-by-group, Spearman, top-k risk capture.

## Success criteria for next milestone

- ICE ≤ `0.02` (match or beat current best).
- NLL < `1.85` without RMSE degradation >1%.
- Spearman(`sigma`, `|error|`) significantly > 0.
