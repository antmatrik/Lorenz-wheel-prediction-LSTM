#!/usr/bin/env python3
"""Train an LSTM and an ESN on the same CSV and plot their forecasts vs actual.

Both models are trained on the first part of the series and evaluated on the
held-out tail, then overlaid on a single comparison figure.

Example:
    python src/compare_lstm_esn.py --input data/samples/water_wheel.csv
"""
import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


def load_lorenz_waterwheel_csv(path: str) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", dtype=float)
    if data.size == 0:
        raise ValueError("CSV is empty or unreadable.")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 4:
        raise ValueError(
            f"Expected 4 columns [angular_velocity, sin, cos, epoch], got {data.shape[1]}."
        )
    if len(data) < 50:
        raise ValueError("Need at least 50 rows for LSTM + ESN comparison.")
    return np.asarray(data, dtype=float)


def standardize_fit(train_2d: np.ndarray):
    mean = train_2d.mean(axis=0)
    std = train_2d.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def standardize_apply(values: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (values - mean) / std


def standardize_inverse(values: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return values * std + mean


def build_windows(series_wsc: np.ndarray, seq_len: int):
    x, y = [], []
    for i in range(len(series_wsc) - seq_len):
        x.append(series_wsc[i : i + seq_len])
        y.append(series_wsc[i + seq_len])
    if not x:
        raise ValueError("Not enough points for chosen sequence length.")
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def train_and_predict_lstm(
    x_all: np.ndarray,
    cut: int,
    seq_len: int,
    units: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    log_every_epochs: int,
) -> np.ndarray:
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required for LSTM. Install with: pip install tensorflow"
        ) from exc

    tf.keras.utils.set_random_seed(seed)

    wsc = x_all[:, :3]
    train_wsc = wsc[:cut]
    mean, std = standardize_fit(train_wsc)
    wsc_scaled = standardize_apply(wsc, mean, std)
    x_train, y_train = build_windows(wsc_scaled[:cut], seq_len)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(seq_len, 3)),
            tf.keras.layers.LSTM(units),
            tf.keras.layers.Dense(3),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )

    class LSTMProgress(tf.keras.callbacks.Callback):
        def __init__(self, total_epochs: int, every_epochs: int):
            super().__init__()
            self.total_epochs = total_epochs
            self.every_epochs = max(1, every_epochs)

        def on_epoch_end(self, epoch, logs=None):
            current = epoch + 1
            if current % self.every_epochs == 0 or current == self.total_epochs:
                loss = None if logs is None else logs.get("loss")
                pct = int(round((current / self.total_epochs) * 100))
                if loss is None:
                    print(f"[LSTM] epoch {current}/{self.total_epochs} ({pct}%)", flush=True)
                else:
                    print(
                        f"[LSTM] epoch {current}/{self.total_epochs} ({pct}%) loss={loss:.6f}",
                        flush=True,
                    )

    print(f"[LSTM] training started: epochs={epochs}, samples={len(x_train)}", flush=True)
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[LSTMProgress(epochs, log_every_epochs)],
    )
    print("[LSTM] training completed", flush=True)

    test_len = len(x_all) - cut
    rollout_seq = wsc_scaled[cut - seq_len : cut].copy()
    preds = np.zeros((test_len, 3), dtype=float)
    for i in range(test_len):
        inp = rollout_seq.reshape(1, seq_len, 3)
        y_next_scaled = model.predict(inp, verbose=0)[0]
        y_next = standardize_inverse(y_next_scaled, mean, std)
        preds[i] = y_next
        rollout_seq = np.vstack([rollout_seq[1:], y_next_scaled])
    return preds


@dataclass
class ESNConfig:
    input_dim: int
    reservoir_dim: int = 400
    output_dim: int = 3
    spectral_radius: float = 0.9
    sparsity: float = 0.95
    input_scale: float = 0.4
    leak_rate: float = 0.3
    ridge: float = 1e-6
    random_seed: int = 42


class EchoStateNetwork:
    def __init__(self, cfg: ESNConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.random_seed)
        self.w_in = rng.uniform(
            low=-cfg.input_scale,
            high=cfg.input_scale,
            size=(cfg.reservoir_dim, cfg.input_dim + 1),
        )
        w = rng.uniform(-1.0, 1.0, size=(cfg.reservoir_dim, cfg.reservoir_dim))
        mask = rng.random((cfg.reservoir_dim, cfg.reservoir_dim)) < cfg.sparsity
        w[mask] = 0.0
        eigvals = np.linalg.eigvals(w)
        radius = np.max(np.abs(eigvals))
        if radius == 0:
            raise ValueError("Reservoir spectral radius is zero; adjust sparsity/seed.")
        self.w = w * (cfg.spectral_radius / radius)
        self.w_out = None
        self.state = np.zeros(cfg.reservoir_dim, dtype=float)

    def reset_state(self):
        self.state = np.zeros(self.cfg.reservoir_dim, dtype=float)

    def _step(self, u: np.ndarray):
        ext_u = np.concatenate(([1.0], u))
        pre = self.w_in @ ext_u + self.w @ self.state
        x_new = np.tanh(pre)
        self.state = (1.0 - self.cfg.leak_rate) * self.state + self.cfg.leak_rate * x_new

    def fit(self, x: np.ndarray, y: np.ndarray, washout: int, log_every_steps: int):
        self.reset_state()
        n = x.shape[0]
        if n <= washout + 1:
            raise ValueError("Not enough training data after ESN washout.")

        states = []
        targets = []
        log_every_steps = max(1, log_every_steps)
        print(
            f"[ESN] training started: steps={n}, washout={washout}, reservoir={self.cfg.reservoir_dim}",
            flush=True,
        )
        for i in range(n):
            self._step(x[i])
            if i >= washout:
                states.append(np.concatenate(([1.0], x[i], self.state)))
                targets.append(y[i])
            current = i + 1
            if current % log_every_steps == 0 or current == n:
                pct = int(round((current / n) * 100))
                print(f"[ESN] step {current}/{n} ({pct}%)", flush=True)

        s = np.asarray(states)
        t = np.asarray(targets)
        reg = self.cfg.ridge * np.eye(s.shape[1])
        self.w_out = np.linalg.solve(s.T @ s + reg, s.T @ t)
        print("[ESN] training completed", flush=True)

    def predict_one(self, u: np.ndarray) -> np.ndarray:
        if self.w_out is None:
            raise RuntimeError("ESN is not trained. Call fit() first.")
        self._step(u)
        ext = np.concatenate(([1.0], u, self.state))
        return ext @ self.w_out

    def forecast(self, last_input: np.ndarray, steps: int, dt: float) -> np.ndarray:
        pred = np.zeros((steps, 4), dtype=float)
        cur = last_input.copy()
        for i in range(steps):
            y = self.predict_one(cur)
            next_t = cur[3] + dt
            pred[i] = np.array([y[0], y[1], y[2], next_t], dtype=float)
            cur = pred[i]
        return pred


def train_and_predict_esn(
    x_all: np.ndarray,
    cut: int,
    reservoir: int,
    spectral_radius: float,
    leak_rate: float,
    sparsity: float,
    ridge: float,
    washout: int,
    seed: int,
    log_every_steps: int,
) -> np.ndarray:
    dt = np.median(np.diff(x_all[:, 3]))
    if not np.isfinite(dt) or dt == 0:
        raise ValueError("Could not infer a valid time step from epoch column.")

    x_train = x_all[:cut]
    x_in = x_train[:-1]
    y_out = x_train[1:, :3]
    cfg = ESNConfig(
        input_dim=4,
        output_dim=3,
        reservoir_dim=reservoir,
        spectral_radius=spectral_radius,
        sparsity=sparsity,
        leak_rate=leak_rate,
        ridge=ridge,
        random_seed=seed,
    )
    esn = EchoStateNetwork(cfg)
    esn.fit(x_in, y_out, washout=washout, log_every_steps=log_every_steps)

    esn.reset_state()
    for row in x_train:
        esn._step(row)

    test_len = len(x_all) - cut
    forecast = esn.forecast(last_input=x_train[-1], steps=test_len, dt=dt)
    return forecast[:, :3]


def plot_predictions(x_all: np.ndarray, cut: int, lstm_pred: np.ndarray, esn_pred: np.ndarray):
    t = x_all[:, 3]
    t_test = t[cut:]
    actual = x_all[cut:, :3]
    names = ["angular_velocity", "sin", "cos"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t_test, actual[:, i], label="Actual", color="black", linewidth=1.5)
        ax.plot(t_test, lstm_pred[:, i], label="LSTM", color="tab:blue", alpha=0.9)
        ax.plot(t_test, esn_pred[:, i], label="ESN", color="tab:orange", alpha=0.9)
        ax.set_ylabel(names[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("epoch")
    fig.suptitle("Lorenz Waterwheel: Actual vs LSTM vs ESN")
    plt.tight_layout()
    plt.show()


def main():
    p = argparse.ArgumentParser(
        description="Train LSTM + ESN on CSV and show predictions vs actual data."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Input CSV with [angular_velocity, sin, cos, epoch] (no header)",
    )
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--units", type=int, default=64)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-3)

    p.add_argument("--reservoir", type=int, default=400)
    p.add_argument("--spectral-radius", type=float, default=0.9)
    p.add_argument("--leak-rate", type=float, default=0.3)
    p.add_argument("--sparsity", type=float, default=0.95)
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--washout", type=int, default=100)
    p.add_argument(
        "--log-every-epochs",
        type=int,
        default=5,
        help="Log LSTM progress every N epochs",
    )
    p.add_argument(
        "--log-every-steps",
        type=int,
        default=1000,
        help="Log ESN progress every N training steps",
    )
    args = p.parse_args()

    x_all = load_lorenz_waterwheel_csv(args.input)
    n = len(x_all)
    cut = int(n * args.train_ratio)
    cut = max(args.seq_len + 5, min(cut, n - 5))

    if cut <= args.washout + 5:
        raise ValueError("Training split too small for ESN washout. Lower washout or raise train-ratio.")

    lstm_pred = train_and_predict_lstm(
        x_all=x_all,
        cut=cut,
        seq_len=args.seq_len,
        units=args.units,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        log_every_epochs=args.log_every_epochs,
    )
    esn_pred = train_and_predict_esn(
        x_all=x_all,
        cut=cut,
        reservoir=args.reservoir,
        spectral_radius=args.spectral_radius,
        leak_rate=args.leak_rate,
        sparsity=args.sparsity,
        ridge=args.ridge,
        washout=args.washout,
        seed=args.seed,
        log_every_steps=args.log_every_steps,
    )

    plot_predictions(x_all=x_all, cut=cut, lstm_pred=lstm_pred, esn_pred=esn_pred)


if __name__ == "__main__":
    main()
