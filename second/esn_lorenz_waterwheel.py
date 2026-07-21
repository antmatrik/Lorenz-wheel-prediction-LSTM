#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
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

    if len(t) < 10:
        raise ValueError("Need at least 10 rows for ESN training.")

    x = np.column_stack([w, s, c, t])
    return x


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

    def fit(self, x: np.ndarray, y: np.ndarray, washout: int = 100):
        self.reset_state()
        n = x.shape[0]
        if n <= washout + 1:
            raise ValueError("Not enough data after washout for training.")

        states = []
        targets = []
        for i in range(n):
            self._step(x[i])
            if i >= washout:
                states.append(np.concatenate(([1.0], x[i], self.state)))
                targets.append(y[i])

        s = np.asarray(states)
        t = np.asarray(targets)

        reg = self.cfg.ridge * np.eye(s.shape[1])
        self.w_out = np.linalg.solve(s.T @ s + reg, s.T @ t)

    def predict_one(self, u: np.ndarray) -> np.ndarray:
        if self.w_out is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        self._step(u)
        ext = np.concatenate(([1.0], u, self.state))
        return ext @ self.w_out

    def forecast(
        self, last_input: np.ndarray, steps: int, dt: float
    ) -> np.ndarray:
        pred = np.zeros((steps, 4), dtype=float)
        cur = last_input.copy()
        for i in range(steps):
            y = self.predict_one(cur)
            next_t = cur[3] + dt
            next_w, next_s, next_c = y[0], y[1], y[2]
            pred[i] = np.array([next_w, next_s, next_c, next_t], dtype=float)
            cur = pred[i]
        return pred


def split_train(x: np.ndarray, ratio: float):
    n = len(x)
    cut = int(n * ratio)
    cut = max(20, min(cut, n - 5))
    return x[:cut], x[cut:]


def main():
    p = argparse.ArgumentParser(
        description="Echo State Network forecast for Lorenz water wheel CSV."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Input CSV with [angular_velocity, sin, cos, epoch] (no header)",
    )
    p.add_argument("--output", default="forecast.csv", help="Output CSV for predicted next points")
    p.add_argument("--steps", type=int, default=200, help="Number of future points to forecast")
    p.add_argument("--train-ratio", type=float, default=0.8, help="Fraction of series used for training")
    p.add_argument("--reservoir", type=int, default=400, help="Reservoir size")
    p.add_argument("--spectral-radius", type=float, default=0.9)
    p.add_argument("--leak-rate", type=float, default=0.3)
    p.add_argument("--sparsity", type=float, default=0.95)
    p.add_argument("--ridge", type=float, default=1e-6)
    p.add_argument("--washout", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    x = load_lorenz_waterwheel_csv(args.input)
    x_train, _ = split_train(x, args.train_ratio)

    dt = np.median(np.diff(x[:, 3]))
    if not np.isfinite(dt) or dt == 0:
        raise ValueError("Could not infer a valid time step from epoch column.")

    x_in = x_train[:-1]
    y_out = x_train[1:, :3]  # predict next [angular_velocity, sin, cos]

    cfg = ESNConfig(
        input_dim=4,
        output_dim=3,
        reservoir_dim=args.reservoir,
        spectral_radius=args.spectral_radius,
        sparsity=args.sparsity,
        leak_rate=args.leak_rate,
        ridge=args.ridge,
        random_seed=args.seed,
    )
    esn = EchoStateNetwork(cfg)
    esn.fit(x_in, y_out, washout=args.washout)

    # Warm reservoir on full observed sequence before free-run forecasting.
    esn.reset_state()
    for row in x:
        esn._step(row)

    forecast = esn.forecast(last_input=x[-1], steps=args.steps, dt=dt)

    header = "angular_velocity,sin,cos,epoch"
    np.savetxt(args.output, forecast, delimiter=",", header=header, comments="")
    print(f"Wrote {args.steps} forecast points to {args.output}")


if __name__ == "__main__":
    main()
