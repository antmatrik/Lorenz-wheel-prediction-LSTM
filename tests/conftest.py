"""Shared pytest fixtures for the Lorenz water-wheel model tests.

``src/`` is placed on the import path via ``pythonpath = src`` in pytest.ini, so tests
import the model packages directly, e.g. ``from tcn import tcn_forecast`` /
``from common import evaluate`` / ``from lstm import lstm_train_predict``.
"""
import numpy as np
import pytest


def make_series(n: int = 400, dt: float = 0.033, t0: float = 1_700_000_000.0, seed: int = 0) -> np.ndarray:
    """Synthetic water-wheel-like array: columns [angular_velocity, sin, cos, epoch].

    Angular velocity varies smoothly (so it is autoregressible) with a little noise;
    sin/cos are derived from the integrated wheel angle, exactly like the real data.
    """
    rng = np.random.default_rng(seed)
    steps = np.arange(n)
    w = 5.0 * np.sin(0.02 * steps) + rng.normal(0.0, 0.1, size=n)
    theta = np.cumsum(w) * dt
    return np.column_stack([w, np.sin(theta), np.cos(theta), t0 + steps * dt]).astype(float)


@pytest.fixture
def synthetic_series() -> np.ndarray:
    return make_series(n=400, seed=1)


def _write_csv(path, arr: np.ndarray) -> None:
    np.savetxt(str(path), arr, delimiter=",", fmt="%.10f")


@pytest.fixture
def tiny_test_dataset(tmp_path):
    """A temp <NN>_in.csv / <NN>_out.csv dataset (2 pairs) shaped like data/test-dataset."""
    d = tmp_path / "test-dataset"
    d.mkdir()
    for i in range(1, 3):
        full = make_series(n=360, seed=10 + i)
        _write_csv(d / f"{i:02d}_in.csv", full[:300])
        _write_csv(d / f"{i:02d}_out.csv", full[300:])
    return d


@pytest.fixture
def tiny_tcn_checkpoint(tmp_path, synthetic_series):
    """A small, randomly-initialised TCN checkpoint saved to disk; returns its path.

    Training is not needed to exercise the rollout/eval plumbing, so this keeps the
    fixture fast. Uses a tiny window so it works with short test files.
    """
    pytest.importorskip("torch")  # skip (not error) tests using this fixture when torch is absent
    from tcn import tcn_forecast

    window, num_channels, kernel_size = 16, (8, 8), 2
    model = tcn_forecast.TCNPredictor(
        n_features=3, num_channels=num_channels, kernel_size=kernel_size, dropout=0.0
    )
    scaler = tcn_forecast.fit_global_scaler([synthetic_series])
    path = tmp_path / "tcn_tiny.pt"
    tcn_forecast.save_checkpoint(model, scaler, window, num_channels, kernel_size, path=str(path))
    return path
