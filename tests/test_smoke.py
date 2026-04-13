"""End-to-end smoke test on synthetic data.

This exercises everything EXCEPT the real Stability Oracle backbone:
    - data loading / caching (via synthetic embeddings)
    - all three heads
    - training loop with early stopping
    - evaluation (RMSE, MAE, NLL, ICE)
    - reliability-diagram plotting

If these tests pass, you know the downstream pipeline is correct. The
only remaining step is wiring up `uapp/backbone.py`.

Run with:
    python -m pytest tests/test_smoke.py -v
or:
    python tests/test_smoke.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uapp.data import load_cached_embeddings, make_loader, save_cached_embeddings
from uapp.evaluate import (
    compute_coverage_curve,
    compute_gaussian_nll,
    compute_ice,
    compute_mae,
    compute_rmse,
    evaluate_head,
    gather_probabilistic_predictions,
    plot_reliability_diagram,
    save_results_json,
)
from uapp.heads import MSEHead, SingleHeadNLL, TwoHeadNLL, build_head, is_probabilistic
from uapp.losses import gaussian_nll_loss, mse_loss
from uapp.train import TrainConfig, train_head
from uapp.utils import set_seed


# ---------------------------------------------------------------------------
# Synthetic data: a toy regression with heteroscedastic noise
# ---------------------------------------------------------------------------

def make_synthetic(
    n: int,
    d: int,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create (X, y) where y depends on X with input-dependent noise.

    Ground truth:
        y = w^T x + noise(x)
    where noise(x) has std depending on the first coordinate of x — so a
    well-trained probabilistic head should learn higher sigma for points
    with large x[0].
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    w = rng.normal(size=(d,)).astype(np.float32) * 0.3
    signal = X @ w
    # Noise std scales with |x[0]| so the model has something to learn
    noise_std = 0.1 + 0.8 * np.abs(X[:, 0])
    noise = rng.normal(size=(n,)).astype(np.float32) * noise_std
    y = signal + noise
    return torch.from_numpy(X), torch.from_numpy(y)


@pytest.fixture(scope="module")
def synth_splits():
    """Train/val/test synthetic splits."""
    set_seed(0)
    d = 16
    X_tr, y_tr = make_synthetic(800, d, seed=1)
    X_va, y_va = make_synthetic(200, d, seed=2)
    X_te, y_te = make_synthetic(400, d, seed=3)
    return {
        "train": (X_tr, y_tr),
        "val": (X_va, y_va),
        "test": (X_te, y_te),
    }


@pytest.fixture(scope="module")
def tmp_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke_out")
    yield out
    shutil.rmtree(out, ignore_errors=True)


# ---------------------------------------------------------------------------
# Unit-level tests
# ---------------------------------------------------------------------------

class TestLosses:
    def test_mse_zero_when_pred_equals_target(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        tgt = torch.tensor([1.0, 2.0, 3.0])
        assert mse_loss(pred, tgt).item() == pytest.approx(0.0)

    def test_gaussian_nll_finite(self):
        mu = torch.tensor([0.0, 0.0, 0.0])
        sigma = torch.tensor([1.0, 1.0, 1.0])
        y = torch.tensor([0.5, -0.3, 0.1])
        loss = gaussian_nll_loss(mu, sigma, y)
        assert torch.isfinite(loss)

    def test_gaussian_nll_penalizes_overconfidence(self):
        """A confident wrong prediction should have higher NLL than
        a less confident wrong prediction with the same mean error."""
        mu = torch.tensor([0.0])
        y = torch.tensor([2.0])
        confident = gaussian_nll_loss(mu, torch.tensor([0.1]), y)
        humble = gaussian_nll_loss(mu, torch.tensor([2.0]), y)
        assert confident.item() > humble.item()


class TestHeads:
    def test_mse_head_output_shape(self):
        head = MSEHead(d_in=16)
        X = torch.randn(5, 16)
        out = head(X)
        assert out.shape == (5,)
        assert not is_probabilistic(head)

    def test_two_head_output_shapes(self):
        head = TwoHeadNLL(d_in=16)
        X = torch.randn(5, 16)
        mu, sigma = head(X)
        assert mu.shape == (5,)
        assert sigma.shape == (5,)
        assert (sigma > 0).all(), "sigma must be strictly positive"
        assert is_probabilistic(head)

    def test_single_head_output_shapes(self):
        head = SingleHeadNLL(d_in=16)
        X = torch.randn(5, 16)
        mu, sigma = head(X)
        assert mu.shape == (5,)
        assert sigma.shape == (5,)
        assert (sigma > 0).all()
        assert is_probabilistic(head)

    def test_build_head_factory(self):
        assert isinstance(build_head("mse", 16), MSEHead)
        assert isinstance(build_head("two_head_nll", 16), TwoHeadNLL)
        assert isinstance(build_head("single_head_nll", 16), SingleHeadNLL)
        with pytest.raises(ValueError):
            build_head("nonsense", 16)


class TestMetrics:
    def test_rmse_mae_basic(self):
        mu = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 4.0])
        # errors are [0, 0, 1] -> RMSE = sqrt(1/3), MAE = 1/3
        assert compute_rmse(mu, y) == pytest.approx(np.sqrt(1 / 3))
        assert compute_mae(mu, y) == pytest.approx(1 / 3)

    def test_gaussian_nll_numeric(self):
        mu = np.array([0.0])
        sigma = np.array([1.0])
        y = np.array([0.0])
        # For N(0,1) at y=0: 0.5 * log(2*pi) ~ 0.9189
        nll = compute_gaussian_nll(mu, sigma, y)
        assert nll == pytest.approx(0.5 * np.log(2 * np.pi), abs=1e-6)

    def test_perfect_calibration_has_zero_ice(self):
        """If we sample y from N(mu, sigma^2), empirical coverage should
        match nominal levels. ICE should be small."""
        rng = np.random.default_rng(42)
        n = 5000
        mu = np.zeros(n)
        sigma = np.ones(n)
        y = rng.normal(mu, sigma)
        ice, coverage = compute_ice(mu, sigma, y)
        assert ice < 0.05, f"ICE should be near 0 for perfectly calibrated, got {ice}"
        # 90% interval should contain ~90% of points
        assert 0.87 < coverage["0.90"] < 0.93

    def test_overconfidence_has_high_ice(self):
        """If we predict sigma that is too small, empirical coverage
        will be well below nominal -> high ICE."""
        rng = np.random.default_rng(42)
        n = 5000
        mu = np.zeros(n)
        true_sigma = np.ones(n)
        y = rng.normal(mu, true_sigma)
        # Predict sigma 10x too small -> overconfident
        pred_sigma = np.ones(n) * 0.1
        ice, _ = compute_ice(mu, pred_sigma, y)
        assert ice > 0.3, f"overconfident model should have high ICE, got {ice}"

    def test_coverage_curve_monotonic(self):
        """compute_coverage_curve iterates alphas from low to high, so
        nominal = 1 - alpha is monotonically *decreasing*. Empirical
        coverage should track it (also decreasing) on calibrated data."""
        rng = np.random.default_rng(0)
        n = 2000
        mu = np.zeros(n)
        sigma = np.ones(n)
        y = rng.normal(mu, sigma)
        nominal, empirical = compute_coverage_curve(mu, sigma, y)
        # Nominal coverage is monotonically non-increasing by construction
        assert np.all(np.diff(nominal) <= 0)
        # Empirical should track it (also non-increasing, allow small noise)
        violations = np.sum(np.diff(empirical) > 0.02)
        assert violations <= 2


# ---------------------------------------------------------------------------
# Integration test: the full pipeline on synthetic data
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_cached_embedding_roundtrip(self, synth_splits, tmp_out):
        cache_path = tmp_out / "synth_cache.pt"
        save_cached_embeddings(cache_path, synth_splits, meta={"test": True})
        assert cache_path.exists()
        loaded, meta = load_cached_embeddings(cache_path)
        assert set(loaded.keys()) == {"train", "val", "test"}
        assert meta["d"] == 16
        torch.testing.assert_close(loaded["train"][0], synth_splits["train"][0])

    def test_train_mse_head(self, synth_splits):
        """Training the MSE head should reduce training loss and produce
        a finite best-val-loss checkpoint."""
        set_seed(0)
        X_tr, y_tr = synth_splits["train"]
        X_va, y_va = synth_splits["val"]
        head = build_head("mse", d_in=16, d_hidden=32)
        cfg = TrainConfig(max_epochs=30, patience=10, log_every=100)
        head, hist = train_head(
            head,
            make_loader(X_tr, y_tr, 64, shuffle=True),
            make_loader(X_va, y_va, 64, shuffle=False),
            cfg,
            torch.device("cpu"),
        )
        # Training loss must decrease from its initial value
        assert hist.train_loss[-1] < hist.train_loss[0], \
            "training loss did not decrease"
        # A best-val checkpoint must have been selected (finite value)
        assert np.isfinite(hist.best_val_loss)
        assert hist.best_epoch >= 1

    def test_train_and_eval_all_three_heads(self, synth_splits, tmp_out):
        """Full train-and-evaluate cycle for every head + reliability plot."""
        set_seed(0)
        X_tr, y_tr = synth_splits["train"]
        X_va, y_va = synth_splits["val"]
        X_te, y_te = synth_splits["test"]

        train_loader = make_loader(X_tr, y_tr, 64, shuffle=True)
        val_loader = make_loader(X_va, y_va, 64, shuffle=False)
        test_loader = make_loader(X_te, y_te, 64, shuffle=False)

        results = []
        reliability = {}
        cfg = TrainConfig(max_epochs=60, patience=15, log_every=100)
        for name in ("mse", "two_head_nll", "single_head_nll"):
            head_kwargs = {"d_hidden": 32}
            if name != "mse":
                head_kwargs["init_sigma_bias"] = 0.5
            head = build_head(name, d_in=16, **head_kwargs)
            head, _ = train_head(head, train_loader, val_loader, cfg, torch.device("cpu"))
            result = evaluate_head(head, test_loader, torch.device("cpu"), name)
            results.append(result)
            if is_probabilistic(head):
                reliability[name] = gather_probabilistic_predictions(
                    head, test_loader, torch.device("cpu")
                )

        # Sanity checks on numerical results
        for r in results:
            assert np.isfinite(r.rmse) and r.rmse > 0
            assert np.isfinite(r.mae) and r.mae > 0
            if r.nll is not None:
                assert np.isfinite(r.nll)
            if r.ice is not None:
                assert 0.0 <= r.ice <= 1.0

        # MSE baseline's point accuracy should be in the same ballpark as NLL
        # heads (probabilistic heads aren't supposed to improve RMSE/MAE much).
        rmses = [r.rmse for r in results]
        assert max(rmses) / min(rmses) < 2.5, "heads diverge wildly on point accuracy"

        # Save results + reliability diagram
        save_results_json(results, tmp_out / "results.json")
        plot_reliability_diagram(reliability, tmp_out / "reliability_diagram.png")
        assert (tmp_out / "results.json").exists()
        assert (tmp_out / "reliability_diagram.png").exists()
        assert (tmp_out / "reliability_diagram.png").stat().st_size > 1000


# ---------------------------------------------------------------------------
# Allow running as `python tests/test_smoke.py` without pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"], check=False)
