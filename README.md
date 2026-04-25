# UAPP — Uncertainty-Aware Protein Property Prediction

Course project starter code. A lightweight probabilistic head on top of a
frozen Stability Oracle backbone, trained with Gaussian NLL on T2837.

## What this repo does

1. Loads the Stability Oracle pretrained backbone and **freezes** it.
2. Runs a one-time forward pass over T2837 and **caches** the graph-level
   embeddings `h_G` to disk as a `.pt` file.
3. Trains three small MLP heads on the cached embeddings:
   - `mse` — deterministic baseline, MSE loss.
   - `two_head_nll` — separate MLPs for µ and σ, Gaussian NLL loss.
   - `single_head_nll` — one MLP outputs [µ, s], Gaussian NLL loss.
4. Evaluates each on RMSE, MAE, held-out NLL, and Interval Calibration
   Error (ICE), plus a reliability diagram PNG.
5. Produces a results table comparing all three heads.

The key design choice: the backbone is frozen and its outputs are cached
*once*. All head training operates on cached vectors, so training runs in
seconds per epoch on CPU.

## Repo layout

```
uapp/
├── uapp/                    # library code
│   ├── __init__.py
│   ├── data.py              # T2837 loading + cached embedding dataset
│   ├── backbone.py          # Stability Oracle wrapper (YOU FILL IN TWO FUNCTIONS)
│   ├── heads.py             # MSE, two-head NLL, single-head NLL
│   ├── losses.py            # Gaussian NLL
│   ├── train.py             # training loop with early stopping
│   ├── evaluate.py          # RMSE/MAE/NLL/ICE + reliability diagram
│   └── utils.py             # seeding, logging, device detection
├── scripts/
│   ├── 01_cache_embeddings.py   # run backbone once, save h_G vectors
│   ├── 02_train_head.py         # train one head
│   ├── 03_evaluate.py           # evaluate a trained head
│   └── 04_run_all.py            # end-to-end: train + eval all three heads
├── configs/
│   └── default.yaml         # hyperparameters
├── tests/
│   └── test_smoke.py        # runs the whole pipeline on synthetic data
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then clone the Stability Oracle repo somewhere accessible and note the path.
You'll need it in the next step.

```bash
git clone https://github.com/danny305/StabilityOracle.git ../StabilityOracle
# Download pretrained weights and graphs from the Zenodo link in their README.
```

## Step 1 — Verify the pipeline on synthetic data FIRST

Before touching the real backbone, run the smoke test. It uses randomly
generated "embeddings" to exercise every piece of the pipeline. If this
passes, you know heads, training, and evaluation all work.

```bash
python -m pytest tests/test_smoke.py -v
```

Expected output: all tests pass in under a minute. A reliability diagram
appears at `outputs/smoke/reliability_diagram.png`.

**Do not skip this.** It's the cheapest way to catch bugs before you spend
time on the real backbone.

## Step 2 — Wire up the real backbone

Open `uapp/backbone.py`. There are exactly **two functions** marked with
`# TODO(you)` that you need to fill in:

1. `load_stability_oracle(checkpoint_path)` — load the pretrained model.
2. `extract_graph_embedding(model, protein_input)` — run a forward pass and
   return the pooled graph-level vector `h_G`.

Everything else in the repo is already written and tested. The file has
extensive comments telling you what to look for in the Stability Oracle
codebase. If you get stuck, paste the relevant Stability Oracle files into
chat and I'll help you figure it out.

## Step 3 — Cache embeddings

```bash
python scripts/01_cache_embeddings.py \
    --t2837-path /path/to/T2837 \
    --backbone-checkpoint /path/to/stability_oracle.pt \
    --out cache/t2837_embeddings.pt
```

This runs the frozen backbone over all 2,837 mutations once. Expect 5–20
minutes on CPU depending on your machine. The output file is small
(a few MB) and is all you need for the rest of the project.

## Step 4 — Train and evaluate all three heads

```bash
python scripts/04_run_all.py \
    --embeddings cache/t2837_embeddings.pt \
    --out outputs/real
```

This trains all three heads, evaluates each, writes results to
`outputs/real/results.json`, and saves a reliability diagram PNG.
Expected runtime: a few minutes on CPU.

## Step 5 — Read the results

```bash
cat outputs/real/results.json
open outputs/real/reliability_diagram.png
```

You'll see a table comparing the three heads on RMSE, MAE, NLL, and ICE.
The reliability diagram is what goes in your final report.

## Interpreting results

- **MSE baseline vs. NLL heads on RMSE/MAE**: should be comparable. The
  probabilistic head is not supposed to *improve* point accuracy; it adds
  calibrated uncertainty on top.
- **Held-out NLL**: lower is better. NLL heads should beat the MSE baseline
  because the baseline assumes constant variance.
- **ICE**: lower is better. A well-calibrated model has ICE near 0 — a
  nominal 90% interval actually contains ~90% of test points.
- **Reliability diagram**: the closer the curve is to the diagonal, the
  better calibrated. Systematic deviation above the diagonal = overconfident;
  below = underconfident.

## Gotchas

- **Variance collapse**: if σ collapses to the floor for all inputs, your
  NLL head is behaving like a deterministic model. Check `outputs/.../sigma_stats.json`
  after training. If the mean σ is ~1e-6, try reducing learning rate or
  initializing the variance head bias to a positive number.
- **Dataset split**: use Stability Oracle's existing train/test split. Do
  NOT shuffle across it — that causes data leakage through homologous
  proteins.
- **Random seeds**: all scripts accept `--seed`. Report results averaged
  over 3 seeds in your final report.


## Research tracks runner (Student-t, fixed-sigma, ranking-aware)

After you have cached embeddings with train/val/test splits (for example `cache/t2837_embeddings_v2.pt`), run:

```bash
python scripts/05_run_research_tracks.py \
    --embeddings cache/t2837_embeddings_v2.pt \
    --out outputs/research_tracks \
    --seed 42
```

Outputs:

- `outputs/research_tracks/research_track_summary.csv` — all implemented track results
- `outputs/research_tracks/best_models.json` — best-by-ICE and best-by-NLL snapshots

This script implements the planned tracks:

- Student-t `nu` sweep + post-hoc variance scaling (`sigma' = a*sigma + b`)
- Mentor direction with fixed-sigma probabilistic head
- Ranking-aware auxiliary uncertainty loss
