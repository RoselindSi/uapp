# UAPP Results Deliverable (Notebook Roundup + Action Plan)

## 1) Executive summary

Yes — your Student-t results are in the artifacts, and they are currently the strongest calibration result.

- **Best calibration overall:** `C_StudentT_v3` with **ICE = 0.0209**, **NLL = 1.8529**, **RMSE = 1.4909**, **90% coverage = 0.900**.
- **Best RMSE in probabilistic block:** `C_Laplace` at **RMSE = 1.4859**, but worse calibration (**ICE = 0.0482**) than Student-t.
- **Deep ensemble is stable** but does not beat Student-t on ICE/NLL (ensemble ICE `0.0303`, NLL `1.8592`).
- **Density-aware variants improve NLL** in that notebook table, but calibration worsens substantially (ICE around `0.088`), so they are not first choice for calibrated uncertainty.

Bottom line: keep **Student-t (nu=3)** as the reference model for next implementation.

---

## 2) Consolidated leaderboard (from saved result tables)

### Main ablation leaderboard (selected rows)

| Model | RMSE | MAE | NLL | ICE | cov@90 | Takeaway |
|---|---:|---:|---:|---:|---:|---|
| C_StudentT_v3 | 1.4909 | 1.0844 | 1.8529 | **0.0209** | 0.900 | Best calibrated and strong overall |
| C_Laplace | **1.4859** | 1.0841 | 1.8533 | 0.0482 | 0.924 | Slightly better RMSE, worse calibration |
| C_StudentT_v5 | 1.5463 | 1.1689 | 1.8780 | 0.0522 | 0.929 | Worse than v3 on all major metrics |
| B_TwoStage | **1.4248** | **0.9824** | **1.8359** | 0.1184 | 0.929 | Great point metrics but poorly calibrated |
| B_SingleHead | 1.5630 | 1.1913 | 1.9274 | 0.0868 | 0.953 | Over-coverage and weak calibration |

### Deep ensemble summary

| Model | RMSE | MAE | NLL | ICE | cov@90 |
|---|---:|---:|---:|---:|---:|
| ENSEMBLE (M=5) | 1.5116 | 1.1254 | 1.8592 | 0.0303 | 0.918 |

### Density/Laplace comparison snapshot

| Config | RMSE | NLL | ICE | cov@90 |
|---|---:|---:|---:|---:|
| sig_alea only | 1.393 | 2.2073 | **0.0351** | 0.829 |
| sig_alea + density_v2 | 1.393 | 1.8107 | 0.0878 | 0.906 |
| sig_alea + Laplace | 1.393 | 2.1865 | **0.0349** | 0.829 |
| sig_alea + Lap + density_v2 | 1.393 | 1.8101 | 0.0884 | 0.906 |

Interpretation: density scaling helps NLL but hurts ICE; this is a calibration tradeoff we should control explicitly.

---

## 3) What the current results mean

1. If your objective is **calibrated uncertainty**, `C_StudentT_v3` is the best current reference.
2. If your objective is only **point accuracy**, `B_TwoStage` is strongest but under-delivers on calibration.
3. Ensemble adds robustness but does not outperform the best Student-t config on calibration quality.
4. We should run upcoming experiments against a fixed baseline pair:
   - **Calibration baseline:** `C_StudentT_v3`
   - **Point baseline:** `B_TwoStage`

---

## 4) Implementation queue ("let's implement all and try all")

## Track A — Student-t reference sweeps (start here)

- Reconfirm `nu` sweep (`3, 5, 8, learned`) with 3 seeds.
- Add post-hoc variance scaling on validation:
  - `sigma' = a*sigma`
  - `sigma' = a*sigma + b`
- Keep `C_StudentT_v3` as target to beat on ICE.

## Track B — Mentor direction (fixed variance, mean calibrates error)

- Freeze uncertainty while training mean branch:
  - A: `sigma = c` (global constant tuned on val)
  - B: scheduled `sigma_t` (epoch-dependent only)
- Train mean with MSE/Huber + bias regularizer across mutation/RSA groups.
- Evaluate whether calibration can be recovered with fixed variance + simple scaling.

## Track C — Uncertainty ranking quality

- Add ranking-aware residual objective (pairwise or soft Spearman).
- New metrics: Spearman(`sigma`, `|error|`), top-k risk capture, AURC.

## Track D — Structure-aware uncertainty features

- Add RSA/mutation-context features to variance head only.
- Ablate feature groups to isolate which features move ICE and NLL.

---

## 5) Acceptance criteria for next round

- **Calibration target:** ICE <= 0.0209 (beat current Student-t).
- **NLL target:** < 1.8529.
- **Point-metric guardrail:** RMSE no worse than +1% versus Student-t reference.
- **Ranking target:** Spearman(`sigma`, `|error|`) materially > 0 with improved top-k risk capture.

---

## 6) Immediate recommendation

For your next coding pass, implement in this exact order:

1. Reproduce Student-t v3 numbers with 3 seeds.
2. Add variance scaling (a, b) and rerun.
3. Add fixed-variance mentor track.
4. Add ranking-aware loss.
5. Add variance-head feature ablation.

This order gives fastest feedback while preserving a strong calibrated baseline.
