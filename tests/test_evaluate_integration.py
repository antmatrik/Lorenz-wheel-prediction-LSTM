"""Integration tests for the model-agnostic evaluator (src/common/evaluate.py).

Confirms the TCN backend (native + physics) runs end-to-end through the shared
metric/aggregation/horizon-sweep code, and that the refactored LSTM backend still works
(skipped when TensorFlow isn't installed). ``import evaluate`` itself needs neither torch
nor TF, so the pure-metric test always runs.
"""
import numpy as np
import pytest

from common import evaluate

_METRIC_KEYS = ["rmse", "mae", "rmse_abs", "mae_abs", "rmse_mirror", "corr"]


def test_compute_metrics_perfect_prediction():
    actual = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
    m = evaluate.compute_metrics(actual.copy(), actual.copy())
    assert m["rmse"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse_abs"] == pytest.approx(0.0)
    assert m["corr"] == pytest.approx(1.0)


def test_compute_metrics_mirror_forgives_global_flip():
    actual = np.array([1.0, -2.0, 3.0, -4.0])
    m = evaluate.compute_metrics(-actual, actual)  # exact sign flip
    assert m["rmse"] > 0.0
    assert m["rmse_mirror"] == pytest.approx(0.0)  # mirror forgives the flip


def _tcn_backend(checkpoint_path, physics=False):
    from tcn import tcn_forecast
    model, scaler, window = tcn_forecast.load_checkpoint(str(checkpoint_path))
    return evaluate.TCNBackend(model, scaler, window, physics=physics)


def test_tcn_backend_native_eval(tiny_tcn_checkpoint, tiny_test_dataset):
    backend = _tcn_backend(tiny_tcn_checkpoint)
    rows, aggregate = evaluate.evaluate_dataset(
        backend, tiny_test_dataset, input_rows=0, max_steps=30, limit=2
    )
    assert len(rows) == 2
    for k in _METRIC_KEYS:
        assert k in aggregate
    assert np.isfinite(aggregate["rmse_abs"])  # primary comparison metric


def test_tcn_backend_physics_eval(tiny_tcn_checkpoint, tiny_test_dataset):
    backend = _tcn_backend(tiny_tcn_checkpoint, physics=True)
    rows, aggregate = evaluate.evaluate_dataset(
        backend, tiny_test_dataset, input_rows=0, max_steps=30, limit=2
    )
    assert len(rows) == 2
    assert np.isfinite(aggregate["rmse_abs"])


def test_tcn_horizon_sweep(tiny_tcn_checkpoint, tiny_test_dataset):
    backend = _tcn_backend(tiny_tcn_checkpoint)
    summary = evaluate.evaluate_horizons(
        backend, tiny_test_dataset, input_rows=0, limit=2, horizons=[10, 20]
    )
    assert [s["steps"] for s in summary] == [10, 20]
    for s in summary:
        assert {"model", "base0", "persist"} <= set(s)
        assert np.isfinite(s["base0"]["rmse_abs"])  # naive baseline always defined


def test_render_horizon_plots_writes_one_png_per_horizon(tmp_path):
    """--plots renders one PNG per horizon, named by model + horizon, from given arrays."""
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(0)
    n, full = 3, 120
    preds_w = rng.standard_normal((n, full))
    loaded = [
        {"name": f"{i:02d}", "actual_w": rng.standard_normal(full), "horizon": full, "dt": 0.033}
        for i in range(n)
    ]
    paths = evaluate.render_horizon_plots(preds_w, loaded, "lstm", tmp_path, [50, 100])
    assert {p.name for p in paths} == {"eval_plots_lstm_h50.png", "eval_plots_lstm_h100.png"}
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_render_horizon_plots_clamps_and_dedupes_horizons(tmp_path):
    """Horizons beyond the rollout length clamp to it and collapse to one PNG."""
    pytest.importorskip("matplotlib")
    preds_w = np.zeros((2, 40))
    loaded = [
        {"name": f"{i:02d}", "actual_w": np.zeros(40), "horizon": 40, "dt": 0.033}
        for i in range(2)
    ]
    paths = evaluate.render_horizon_plots(preds_w, loaded, "tcn(physics)", tmp_path, [100, 300])
    assert [p.name for p in paths] == ["eval_plots_tcn_physics_h40.png"]  # clamped + slugged


def test_lstm_backend_regression(tiny_test_dataset):
    """The refactored LSTM path still runs (random weights are fine for a plumbing test)."""
    pytest.importorskip("tensorflow")
    from lstm import lstm_train_predict

    _, model = lstm_train_predict.build_lstm_model(seq_len=8, units=4, learning_rate=1e-3)
    mean = np.array([0.0, 0.0, 0.0])
    std = np.array([5.0, 1.0, 1.0])
    backend = evaluate.LSTMBackend(model, mean, std, seq_len=8)
    rows, aggregate = evaluate.evaluate_dataset(
        backend, tiny_test_dataset, input_rows=0, max_steps=20, limit=2
    )
    assert len(rows) == 2
    assert np.isfinite(aggregate["rmse_abs"])
