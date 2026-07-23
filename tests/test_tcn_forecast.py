"""Unit tests for the PyTorch TCN forecaster (src/tcn/tcn_forecast.py)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tcn import tcn_forecast  # noqa: E402  (after importorskip so a torch-less env skips cleanly)


def test_forward_shape():
    model = tcn_forecast.TCNPredictor(n_features=3, num_channels=(8, 8), kernel_size=3, dropout=0.0)
    x = torch.randn(4, 16, 3)
    out = model(x)
    assert out.shape == (4, 3)


def test_native_rollout_shape_and_finite(synthetic_series):
    model = tcn_forecast.TCNPredictor(n_features=3, num_channels=(8, 8), kernel_size=2, dropout=0.0)
    scaler = tcn_forecast.fit_global_scaler([synthetic_series])
    window, horizon, n = 16, 7, 5
    seeds = np.zeros((n, window, 3), dtype=np.float32)
    preds = tcn_forecast.rollout_native_batched(model, seeds, horizon, scaler)
    assert preds.shape == (n, horizon, 3)
    assert np.isfinite(preds).all()


def test_physics_rollout_shape_and_finite(synthetic_series):
    model = tcn_forecast.TCNPredictor(n_features=3, num_channels=(8, 8), kernel_size=2, dropout=0.0)
    scaler = tcn_forecast.fit_global_scaler([synthetic_series])
    window, horizon, n = 16, 7, 5
    seeds = np.zeros((n, window, 3), dtype=np.float32)
    preds_w = tcn_forecast.rollout_physics_batched(
        model, seeds,
        theta0=np.zeros(n), w_prev=np.zeros(n), dt=np.full(n, 0.033),
        horizon=horizon, scaler=scaler,
    )
    assert preds_w.shape == (n, horizon)
    assert np.isfinite(preds_w).all()


def test_checkpoint_roundtrip(tmp_path, synthetic_series):
    window, num_channels, kernel_size = 16, (8, 8), 2
    model = tcn_forecast.TCNPredictor(
        n_features=3, num_channels=num_channels, kernel_size=kernel_size, dropout=0.0
    )
    model.eval()
    scaler = tcn_forecast.fit_global_scaler([synthetic_series])
    path = tmp_path / "ckpt.pt"
    tcn_forecast.save_checkpoint(model, scaler, window, num_channels, kernel_size, path=str(path))

    reloaded, scaler2, window2 = tcn_forecast.load_checkpoint(str(path))
    assert window2 == window
    np.testing.assert_allclose(scaler2["mean"], scaler["mean"])
    np.testing.assert_allclose(scaler2["std"], scaler["std"])

    x = torch.randn(3, window, 3)
    with torch.no_grad():
        np.testing.assert_allclose(
            model(x).numpy(), reloaded(x).numpy(), rtol=1e-5, atol=1e-6
        )


def test_csv_loader_and_dt(tmp_path, synthetic_series):
    path = tmp_path / "series.csv"
    np.savetxt(str(path), synthetic_series, delimiter=",", fmt="%.10f")
    loaded = tcn_forecast.load_lorenz_waterwheel_csv(str(path))
    assert loaded.shape == synthetic_series.shape
    dt = tcn_forecast.infer_dt(loaded[:, 3])
    assert dt == pytest.approx(0.033, abs=1e-4)


def test_csv_loader_rejects_wrong_columns(tmp_path):
    path = tmp_path / "bad.csv"
    np.savetxt(str(path), np.zeros((50, 3)), delimiter=",")
    with pytest.raises(ValueError):
        tcn_forecast.load_lorenz_waterwheel_csv(str(path))


def test_train_one_epoch_runs(tmp_path, synthetic_series):
    window, horizon = 16, 3
    scaler = tcn_forecast.fit_global_scaler([synthetic_series])
    train_dss, val_dss = tcn_forecast.build_file_datasets(
        [synthetic_series], scaler, window, horizon, val_fraction=0.2
    )
    from torch.utils.data import ConcatDataset

    model = tcn_forecast.TCNPredictor(n_features=3, num_channels=(8, 8), kernel_size=2, dropout=0.0)
    out_path = tmp_path / "trained.pt"
    trained, best_val = tcn_forecast.train(
        model,
        ConcatDataset(train_dss),
        ConcatDataset(val_dss),
        scaler,
        window=window,
        num_channels=(8, 8),
        kernel_size=2,
        epochs=1,
        batch_size=64,
        device="cpu",
        out_path=str(out_path),
    )
    assert np.isfinite(best_val)
    assert out_path.exists()
