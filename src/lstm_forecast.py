#!/usr/bin/env python3
"""Standalone single-file LSTM forecaster for Lorenz water-wheel CSVs.

Trains a small LSTM on sliding windows of [angular_velocity, sin, cos] from one
input file, then forecasts future points via autoregressive rollout. For the
multi-file training pipeline with checkpoints, see lstm_train_predict.py.

Example:
    python src/lstm_forecast.py --input data/samples/water_wheel.csv \\
        --steps 300 --output outputs/forecast_lstm.csv
"""
import argparse
import numpy as np


def load_lorenz_waterwheel_csv(path: str):
    data = np.genfromtxt(path, delimiter=",", dtype=float)
    if data.size == 0:
        raise ValueError("CSV is empty or unreadable.")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 4:
        raise ValueError(
            f"Expected 4 columns in order [angular_velocity, sin, cos, epoch], got {data.shape[1]}."
        )

    w = np.asarray(data[:, 0], dtype=float).reshape(-1)
    s = np.asarray(data[:, 1], dtype=float).reshape(-1)
    c = np.asarray(data[:, 2], dtype=float).reshape(-1)
    t = np.asarray(data[:, 3], dtype=float).reshape(-1)

    if len(t) < 30:
        raise ValueError("Need at least 30 rows for LSTM training.")

    x = np.column_stack([w, s, c, t])
    return x


def build_windows(series_wsc: np.ndarray, seq_len: int):
    # Inputs are [w, sin, cos] windows, target is next [w, sin, cos].
    x, y = [], []
    for i in range(len(series_wsc) - seq_len):
        x.append(series_wsc[i : i + seq_len])
        y.append(series_wsc[i + seq_len])
    if not x:
        raise ValueError("Not enough points for chosen sequence length.")
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def standardize_fit(train_2d: np.ndarray):
    mean = train_2d.mean(axis=0)
    std = train_2d.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def standardize_apply(values: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (values - mean) / std


def standardize_inverse(values: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return values * std + mean


def main():
    p = argparse.ArgumentParser(
        description="LSTM forecast for Lorenz water wheel CSV."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Input CSV with [angular_velocity, sin, cos, epoch] (no header)",
    )
    p.add_argument("--output", default="forecast_lstm.csv", help="Output CSV for predicted next points")
    p.add_argument("--steps", type=int, default=200, help="Number of future points to forecast")
    p.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of series used for training")
    p.add_argument("--seq-len", type=int, default=30, help="LSTM input sequence length")
    p.add_argument("--units", type=int, default=64, help="LSTM hidden units")
    p.add_argument("--epochs", type=int, default=60, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    p.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()

    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required for this script. Install with: pip install tensorflow"
        ) from exc

    tf.keras.utils.set_random_seed(args.seed)

    x_all = load_lorenz_waterwheel_csv(args.input)
    wsc = x_all[:, :3]  # [angular_velocity, sin, cos]
    n = len(wsc)

    cut = int(n * args.train_ratio)
    cut = max(args.seq_len + 5, min(cut, n - 5))

    train_wsc = wsc[:cut]
    mean, std = standardize_fit(train_wsc)
    wsc_scaled = standardize_apply(wsc, mean, std)

    x_train, y_train = build_windows(wsc_scaled[:cut], args.seq_len)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(args.seq_len, 3)),
            tf.keras.layers.LSTM(args.units),
            tf.keras.layers.Dense(3),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="mse",
    )
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath="lstm_weights.weights.h5",
        save_weights_only=True,
        save_freq="epoch",
        verbose=0,
    )
    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
        callbacks=[ckpt_cb],
    )

    dt = np.median(np.diff(x_all[:, 3]))
    if not np.isfinite(dt) or dt == 0:
        raise ValueError("Could not infer a valid time step from epoch column.")

    # Start autoregressive rollout from the last observed seq_len points.
    rollout_seq = wsc_scaled[-args.seq_len:].copy()
    last_t = x_all[-1, 3]
    preds = np.zeros((args.steps, 4), dtype=float)

    for i in range(args.steps):
        inp = rollout_seq.reshape(1, args.seq_len, 3)
        y_next_scaled = model.predict(inp, verbose=0)[0]
        y_next = standardize_inverse(y_next_scaled, mean, std)
        next_t = last_t + dt
        preds[i] = np.array([y_next[0], y_next[1], y_next[2], next_t], dtype=float)

        # Push normalized prediction to sequence for next step.
        rollout_seq = np.vstack([rollout_seq[1:], y_next_scaled])
        last_t = next_t

    header = "angular_velocity,sin,cos,epoch"
    np.savetxt(args.output, preds, delimiter=",", header=header, comments="")
    print(f"Wrote {args.steps} forecast points to {args.output}")


if __name__ == "__main__":
    main()
