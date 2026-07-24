#!/usr/bin/env python3
"""Evaluate a trained LSTM on the test-dataset and report angular-velocity error.

For each ``<NN>_in.csv`` / ``<NN>_out.csv`` pair the model is seeded with the last
``--input-rows`` rows of the input file, forecasts as many steps as the output
file has (capped by ``--max-steps``), and the predicted angular velocity is
compared against the actual angular velocity (column 0) of the output file.

The forecast is **batched across files**: all selected files are advanced
together, so the whole run costs one model call per step (``horizon`` calls),
not one call per file-step (``files x horizon``).

Two families of metrics are reported, because the wheel can flip spin direction
at bifurcation points that no model can reliably predict:

* signed   -- ordinary error on omega (direction matters).
* |omega|  -- error on the magnitude of omega (sign-invariant); a forecast that
              is "mirrored" (right magnitude, wrong direction) scores well here.
* mirror   -- per file, the better of the signed error against +actual or
              -actual, i.e. it forgives a single global direction flip.

The aggregate row (mean over files) gives a small set of numbers you can track
across training attempts. The primary single number is the mean |omega| RMSE.

Run from the project root, e.g.:

    python src/common/evaluate.py                          # all pairs, full horizon
    python src/common/evaluate.py --limit 5 --max-steps 300   # quick smoke eval
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Make the model packages importable however this script is launched
# (`python src/common/evaluate.py` puts src/common on sys.path, not src/).
_SRC_ROOT = Path(__file__).resolve().parents[1]  # .../src
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from lstm.lstm_train_predict import (
    LEARNING_RATE,
    LSTM_UNITS,
    PROJECT_ROOT,
    SEQUENCE_LENGTH,
    STATS_INPUT_PATH,
    WEIGHTS_INPUT_PATH,
    build_lstm_model,
    load_lorenz_waterwheel_csv,
    standardize_apply,
)


def compute_metrics(pred_w: np.ndarray, actual_w: np.ndarray) -> dict:
    """Angular-velocity error metrics for one forecast vs its ground truth."""
    signed = pred_w - actual_w
    rmse = float(np.sqrt(np.mean(signed ** 2)))
    mae = float(np.mean(np.abs(signed)))

    abs_diff = np.abs(pred_w) - np.abs(actual_w)
    rmse_abs = float(np.sqrt(np.mean(abs_diff ** 2)))
    mae_abs = float(np.mean(np.abs(abs_diff)))

    # Forgive a single global direction flip: best of +actual / -actual.
    rmse_mirror = float(min(rmse, np.sqrt(np.mean((pred_w + actual_w) ** 2))))

    if np.std(pred_w) > 1e-12 and np.std(actual_w) > 1e-12:
        corr = float(np.corrcoef(pred_w, actual_w)[0, 1])
    else:
        corr = float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "rmse_abs": rmse_abs,
        "mae_abs": mae_abs,
        "rmse_mirror": rmse_mirror,
        "corr": corr,
    }


def _load_pair(in_path: Path, out_path: Path, input_rows: int, max_steps: int, seq_len: int) -> dict:
    """Load one input/output pair into the pieces the batched rollout needs."""
    x_in = load_lorenz_waterwheel_csv(str(in_path))
    x_out = load_lorenz_waterwheel_csv(str(out_path))

    history = x_in[-input_rows:] if input_rows and input_rows < len(x_in) else x_in
    if len(history) < seq_len:
        raise ValueError(
            f"history has {len(history)} rows but SEQUENCE_LENGTH={seq_len}; increase --input-rows."
        )

    horizon = len(x_out)
    if max_steps:
        horizon = min(horizon, max_steps)

    epochs = history[:, 3]
    dt = float(np.median(np.diff(epochs))) if len(epochs) >= 2 else 1.0
    if not np.isfinite(dt) or dt == 0:
        dt = 1.0

    return {
        "name": in_path.name.replace("_in.csv", ""),
        "seed": history[-seq_len:, :3].astype(float),  # real-units [w, sin, cos]
        "theta0": float(np.arctan2(history[-1, 1], history[-1, 2])),
        "w_prev": float(history[-1, 0]),
        "dt": dt,
        "actual_w": x_out[:horizon, 0].astype(float),
        "horizon": horizon,
    }


def run_batched_rollout(model, seeds_scaled, theta0, w_prev, dt, horizon, mean, std) -> np.ndarray:
    """Advance N files together: one batched model call per step (N x seq_len x 3).

    Same physics-informed rollout as the single-file path -- predict omega,
    integrate it into the wheel angle, reconstruct sin/cos -- vectorised over the
    file (batch) dimension. Returns predicted angular velocity, shape (N, horizon).
    """
    roll = np.array(seeds_scaled, dtype=np.float32, copy=True)
    theta = np.asarray(theta0, dtype=float).copy()
    w_prev = np.asarray(w_prev, dtype=float).copy()
    dt = np.asarray(dt, dtype=float)
    n = roll.shape[0]
    preds_w = np.zeros((n, horizon))
    m0, s0 = float(mean[0]), float(std[0])

    for t in range(horizon):
        out = model(roll, training=False)
        out = out.numpy() if hasattr(out, "numpy") else np.asarray(out)
        w_next = out.reshape(n, -1)[:, 0] * s0 + m0  # inverse z-score of channel 0

        theta = theta + 0.5 * (w_prev + w_next) * dt
        sin_next = np.sin(theta)
        cos_next = np.cos(theta)

        preds_w[:, t] = w_next

        nxt = standardize_apply(
            np.stack([w_next, sin_next, cos_next], axis=1), mean, std
        ).astype(np.float32)
        roll[:, :-1] = roll[:, 1:]
        roll[:, -1] = nxt

        w_prev = w_next

    return preds_w


# ---------------------------------------------------------------------------
# Model backends
#
# Each backend turns a batch of file seeds into predicted angular velocity of
# shape (N, horizon). Everything below the backend -- metrics, aggregation, the
# horizon sweep, the naive baselines, the CSV writer -- is model-agnostic, so the
# numbers are computed identically for every model and stay directly comparable.
# ---------------------------------------------------------------------------


class LSTMBackend:
    """TensorFlow LSTM scored with the physics rollout (predict omega, rebuild sin/cos)."""

    def __init__(self, model, mean: np.ndarray, std: np.ndarray, seq_len: int):
        self.model = model
        self.mean = mean
        self.std = std
        self.seq_len = seq_len
        self.name = "lstm"

    def prepare_seeds(self, loaded: list[dict]) -> np.ndarray:
        return np.stack(
            [standardize_apply(d["seed"], self.mean, self.std) for d in loaded]
        ).astype(np.float32)

    def rollout(self, seeds, theta0, w_prev, dt, horizon) -> np.ndarray:
        return run_batched_rollout(
            self.model, seeds, theta0, w_prev, dt, horizon, self.mean, self.std
        )


class TCNBackend:
    """PyTorch TCN. Native 3-channel rollout by default; physics rollout with --physics.

    ``normalize_state`` z-scores all three channels with the checkpoint's scaler (the
    TCN's own convention), unlike the LSTM which standardizes only channel 0.
    """

    def __init__(self, model, scaler: dict, window: int, physics: bool = False):
        self.model = model
        self.scaler = scaler
        self.seq_len = window
        self.physics = physics
        self.name = "tcn(physics)" if physics else "tcn"

    def prepare_seeds(self, loaded: list[dict]) -> np.ndarray:
        from tcn import tcn_forecast
        return np.stack(
            [tcn_forecast.normalize_state(d["seed"], self.scaler) for d in loaded]
        ).astype(np.float32)

    def rollout(self, seeds, theta0, w_prev, dt, horizon) -> np.ndarray:
        from tcn import tcn_forecast
        if self.physics:
            return tcn_forecast.rollout_physics_batched(
                self.model, seeds, theta0, w_prev, dt, horizon, self.scaler
            )
        preds = tcn_forecast.rollout_native_batched(self.model, seeds, horizon, self.scaler)
        return preds[:, :, 0]  # angular-velocity channel


def evaluate_dataset(
    backend,
    test_dir: Path,
    input_rows: int,
    max_steps: int,
    limit: int,
    plot_horizons: list[int] | None = None,
    plot_dir: Path | None = None,
    plot_max_rows: int = 10,
) -> tuple[list[dict], dict]:
    """Batched evaluation over every <NN>_in/_out pair; returns rows + aggregate.

    When ``plot_horizons`` is given, also render one tall actual-vs-predicted PNG
    per horizon (up to ``plot_max_rows`` files) from the rollout already computed
    here (no extra model calls).
    """
    in_files = sorted(test_dir.glob("*_in.csv"))
    if limit:
        in_files = in_files[:limit]
    if not in_files:
        raise ValueError(f"No *_in.csv files found in {test_dir}")

    loaded: list[dict] = []
    for in_path in in_files:
        out_path = in_path.with_name(in_path.name.replace("_in.csv", "_out.csv"))
        if not out_path.exists():
            print(f"[EVAL] skipped {in_path.name}: no matching _out file", flush=True)
            continue
        try:
            loaded.append(_load_pair(in_path, out_path, input_rows, max_steps, backend.seq_len))
        except Exception as exc:  # keep going; report which file failed
            print(f"[EVAL] skipped {in_path.name}: {exc}", flush=True)

    if not loaded:
        raise ValueError("No test pairs were evaluated.")

    seeds = backend.prepare_seeds(loaded)
    theta0 = np.array([d["theta0"] for d in loaded])
    w_prev = np.array([d["w_prev"] for d in loaded])
    dt = np.array([d["dt"] for d in loaded])
    horizon = max(d["horizon"] for d in loaded)

    print(
        f"[EVAL] batched rollout ({backend.name}): {len(loaded)} files x {horizon} steps "
        f"-> {horizon} model calls (batch of {len(loaded)})",
        flush=True,
    )
    preds_w = backend.rollout(seeds, theta0, w_prev, dt, horizon)

    rows: list[dict] = []
    for i, d in enumerate(loaded):
        h = d["horizon"]
        row = compute_metrics(preds_w[i, :h], d["actual_w"][:h])
        row["file"] = d["name"]
        row["steps"] = h
        rows.append(row)
        print(
            f"[EVAL] {row['file']}: rmse={row['rmse']:.4f} "
            f"|w|rmse={row['rmse_abs']:.4f} mirror={row['rmse_mirror']:.4f} "
            f"corr={row['corr']:.3f}",
            flush=True,
        )

    keys = ["rmse", "mae", "rmse_abs", "mae_abs", "rmse_mirror", "corr"]
    aggregate = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
    aggregate["file"] = "AGGREGATE(mean)"
    aggregate["steps"] = int(np.round(np.mean([r["steps"] for r in rows])))

    if plot_horizons:
        render_horizon_plots(
            preds_w, loaded, backend.name,
            plot_dir if plot_dir is not None else (PROJECT_ROOT / "outputs"),
            plot_horizons, max_rows=plot_max_rows,
        )
    return rows, aggregate


def _aggregate_at(preds_w: np.ndarray, loaded: list[dict], h: int) -> dict:
    """Mean metrics over files when each forecast is truncated to ``h`` steps."""
    keys = ["rmse", "mae", "rmse_abs", "mae_abs", "rmse_mirror", "corr"]
    rows = []
    for i, d in enumerate(loaded):
        hh = min(h, d["horizon"])
        rows.append(compute_metrics(preds_w[i, :hh], d["actual_w"][:hh]))
    out = {}
    for k in keys:
        # Drop NaNs manually (a constant baseline has undefined correlation) so
        # np.nanmean does not warn on an all-NaN column.
        vals = [r[k] for r in rows if r[k] == r[k]]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def evaluate_horizons(
    backend,
    test_dir: Path,
    input_rows: int,
    limit: int,
    horizons: list[int],
) -> list[dict]:
    """Skill-vs-horizon sweep with naive baselines.

    Runs ONE batched rollout to the largest horizon (the rollout is purely
    autoregressive, so a step's prediction is the same regardless of the total
    horizon -- we just slice it at each cutoff). At each horizon it also scores
    two references so the model's numbers have a "beat this" anchor:

    * base0    -- predict omega = 0 everywhere;
    * persist  -- hold the last observed omega constant.

    Chaos guarantees skill decays with horizon; this shows *where*.
    """
    horizons = sorted({int(h) for h in horizons if int(h) > 0})
    if not horizons:
        raise ValueError("--horizons must contain at least one positive integer.")

    in_files = sorted(Path(test_dir).glob("*_in.csv"))
    if limit:
        in_files = in_files[:limit]

    loaded: list[dict] = []
    for in_path in in_files:
        out_path = in_path.with_name(in_path.name.replace("_in.csv", "_out.csv"))
        if not out_path.exists():
            print(f"[EVAL] skipped {in_path.name}: no matching _out file", flush=True)
            continue
        try:
            loaded.append(_load_pair(in_path, out_path, input_rows, max(horizons), backend.seq_len))
        except Exception as exc:
            print(f"[EVAL] skipped {in_path.name}: {exc}", flush=True)

    if not loaded:
        raise ValueError("No test pairs were evaluated.")

    max_h = max(d["horizon"] for d in loaded)
    horizons = [h for h in horizons if h <= max_h] or [max_h]
    dt_med = float(np.median([d["dt"] for d in loaded]))

    seeds = backend.prepare_seeds(loaded)
    theta0 = np.array([d["theta0"] for d in loaded])
    w_prev = np.array([d["w_prev"] for d in loaded])
    dt = np.array([d["dt"] for d in loaded])

    print(f"[EVAL] horizon sweep ({backend.name}): {len(loaded)} files, one rollout of "
          f"{max_h} steps -> {max_h} model calls", flush=True)
    preds_w = backend.rollout(seeds, theta0, w_prev, dt, max_h)

    # Naive baselines over the same horizon (shape (n_files, max_h)).
    zero = np.zeros((len(loaded), max_h))
    persist = np.array([[d["w_prev"]] * max_h for d in loaded], dtype=float)

    print("\n=== Skill vs horizon (mean over files) ===")
    print(f"{'steps':>6} {'~sec':>6} | {'corr':>7} {'signed':>8} {'|w|RMSE':>8} "
          f"{'mirror':>8} | {'base0|w|':>9} {'persist|w|':>10}")
    print("-" * 78)
    summary: list[dict] = []
    for h in horizons:
        m = _aggregate_at(preds_w, loaded, h)
        z = _aggregate_at(zero, loaded, h)
        p = _aggregate_at(persist, loaded, h)
        print(f"{h:>6} {h * dt_med:>6.1f} | {m['corr']:>7.3f} {m['rmse']:>8.3f} "
              f"{m['rmse_abs']:>8.3f} {m['rmse_mirror']:>8.3f} | "
              f"{z['rmse_abs']:>9.3f} {p['rmse_abs']:>10.3f}", flush=True)
        summary.append({"steps": h, "sec": h * dt_med, "model": m, "base0": z, "persist": p})

    print("\n  Read it as: the model is useful while |w|RMSE stays well below the")
    print("  base0/persist columns and corr stays high; chaos closes that gap as")
    print("  the horizon grows. Signed error is inflated by direction flips that")
    print("  no model can predict -- compare |w|RMSE (magnitude) across runs.")
    return summary


def render_horizon_plots(
    preds_w: np.ndarray,
    loaded: list[dict],
    model_name: str,
    out_dir: Path,
    horizons: list[int],
    max_rows: int = 10,
) -> list[Path]:
    """Two tall PNGs per horizon -- signed omega and |omega| (magnitude) -- each
    stacking up to ``max_rows`` test files as actual-vs-predicted panels truncated to
    that horizon. The |omega| view is the fair magnitude comparison: a mirrored
    forecast (right speed, wrong direction at a bifurcation) looks good there even
    though it scores poorly on the signed plot.

    Every panel is sliced from ``preds_w`` (the rollout already computed by the
    caller), so this adds no model calls -- only matplotlib rendering. Files are named
    ``eval_plots_<model>_h<steps>.png`` (signed) and ``..._h<steps>_abs.png`` (|omega|).
    Returns the written PNG paths (empty if matplotlib is unavailable).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: write files, never open a window
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[EVAL] --plots skipped: matplotlib unavailable ({exc}); "
              f"pip install matplotlib", flush=True)
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(len(loaded), max_rows)
    if len(loaded) > max_rows:
        print(f"[EVAL] plots: {len(loaded)} files > max_rows={max_rows}; "
              f"plotting the first {n}.", flush=True)

    avail = preds_w.shape[1]
    # Clamp to what was actually rolled out and de-duplicate (e.g. with --max-steps).
    horizons = sorted({min(int(h), avail) for h in horizons if int(h) > 0})
    dt_med = float(np.median([d["dt"] for d in loaded]))
    slug = "".join(c if c.isalnum() else "_" for c in model_name).strip("_") or "model"
    actual_c, pred_c = "#1f77b4", "#d62728"
    paths: list[Path] = []

    def _figure(H: int, absolute: bool) -> Path:
        """Render one tall PNG for horizon H; signed omega, or |omega| if ``absolute``."""
        fig, axes = plt.subplots(n, 1, figsize=(12, max(2.2, 1.35 * n)), squeeze=False)
        axes = axes[:, 0]
        for i in range(n):
            d = loaded[i]
            h = min(H, d["horizon"], avail)
            t = np.arange(h) * d["dt"]
            pred, actual = preds_w[i, :h], d["actual_w"][:h]
            m = compute_metrics(pred, actual)

            if absolute:
                y_pred, y_actual = np.abs(pred), np.abs(actual)
                # Magnitude correlation, the |ω| analogue of the signed plot's corr.
                if np.std(y_pred) > 1e-12 and np.std(y_actual) > 1e-12:
                    mag_corr = float(np.corrcoef(y_pred, y_actual)[0, 1])
                else:
                    mag_corr = float("nan")
                label = f"|ω|RMSE={m['rmse_abs']:.2f}   corr|ω|={mag_corr:.2f}"
                a_lbl, p_lbl = "actual |ω|", "predicted |ω|"
            else:
                y_pred, y_actual = pred, actual
                label = f"|ω|RMSE={m['rmse_abs']:.2f}   corr={m['corr']:.2f}"
                a_lbl, p_lbl = "actual ω", "predicted ω"

            ax = axes[i]
            ax.plot(t, y_actual, color=actual_c, lw=0.9, label=a_lbl)
            ax.plot(t, y_pred, color=pred_c, lw=0.9, alpha=0.8, label=p_lbl)
            ax.axhline(0, color="0.7", lw=0.5)
            ax.margins(x=0)
            ax.tick_params(labelsize=7)
            ax.text(0.006, 0.90, d["name"], transform=ax.transAxes,
                    ha="left", va="top", fontsize=8, fontweight="bold")
            ax.text(0.994, 0.90, label, transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, color="0.35")
            if i == 0:
                ax.legend(loc="upper center", fontsize=7, ncol=2, framealpha=0.6)
            if i < n - 1:
                ax.tick_params(labelbottom=False)  # only the bottom row shows the time axis
        axes[-1].set_xlabel("time ahead (s)", fontsize=9)
        kind = "|ω| (magnitude)" if absolute else "angular velocity (ω)"
        fig.suptitle(
            f"Model: {model_name}  —  actual vs predicted {kind}  —  "
            f"horizon {H} steps ≈ {H * dt_med:.0f} s  —  {n} test files",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.995])
        path = out_dir / f"eval_plots_{slug}_h{H}{'_abs' if absolute else ''}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return path

    # Each horizon gets both the signed view and the |ω| (magnitude) view.
    for H in horizons:
        for absolute in (False, True):
            path = _figure(H, absolute)
            paths.append(path)
            print(f"[EVAL] saved plot: {path}  ({n} files, horizon {H} ≈ "
                  f"{H * dt_med:.0f}s, {'|ω|' if absolute else 'signed'})", flush=True)

    return paths


def _write_csv(output_path: Path, rows: list[dict], aggregate: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["file", "steps", "rmse", "mae", "rmse_abs", "mae_abs", "rmse_mirror", "corr"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})
        writer.writerow({k: aggregate[k] for k in fields})


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        choices=["lstm", "tcn"],
        default="lstm",
        help="Which trained model to score (default: lstm). The metrics are identical "
        "for both, so runs are directly comparable.",
    )
    p.add_argument(
        "--physics",
        action="store_true",
        help="TCN only: use the LSTM-style physics rollout (predict omega, rebuild "
        "sin/cos) instead of the TCN's native 3-channel rollout. Isolates the "
        "architecture from the rollout strategy.",
    )
    p.add_argument(
        "--test-dir",
        default=str(PROJECT_ROOT / "data" / "test-dataset"),
        help="Directory of <NN>_in.csv / <NN>_out.csv pairs.",
    )
    p.add_argument(
        "--input-rows",
        type=int,
        default=0,
        help="Use only the last N rows of each input file as history "
        "(0 = all rows). Must be >= SEQUENCE_LENGTH.",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Cap the forecast horizon per file (0 = full output-file length). "
        "Fewer steps = fewer model calls = faster.",
    )
    p.add_argument(
        "--horizons",
        default="",
        help="Comma-separated horizons for a skill-vs-horizon sweep with naive "
        "baselines, e.g. '25,50,100,200,400,800,1800'. Runs ONE rollout to the "
        "largest and reports metrics at each. Overrides --max-steps when set.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N pairs (0 = all).",
    )
    p.add_argument(
        "--plots",
        action="store_true",
        help="Render one tall PNG per --plot-horizons value: up to --plot-max-rows "
        "test files stacked, each an actual-vs-predicted omega panel. Sliced from the "
        "rollout already computed (no extra model calls). Requires matplotlib. Ignored "
        "with --horizons.",
    )
    p.add_argument(
        "--plot-horizons",
        default="100,200,500,1800",
        help="Comma-separated step horizons to render when --plots is set "
        "(default: %(default)s; 1800 steps is the full ~60 s output).",
    )
    p.add_argument(
        "--plot-max-rows",
        type=int,
        default=10,
        help="Max test files (rows) per --plots PNG (default: %(default)s). "
        "Extra files beyond this are skipped, plotting the first N.",
    )
    p.add_argument(
        "--plot-dir",
        default=str(PROJECT_ROOT / "outputs"),
        help="Directory for the --plots PNGs (default: outputs/).",
    )
    p.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH,
                   help="LSTM only: must match the trained model (default: %(default)s). "
                        "The TCN's window comes from its checkpoint.")
    p.add_argument("--units", type=int, default=LSTM_UNITS,
                   help="LSTM only: must match the trained model (default: %(default)s).")
    p.add_argument("--weights", default=WEIGHTS_INPUT_PATH, help="LSTM weights path.")
    p.add_argument("--stats", default=STATS_INPUT_PATH,
                   help="LSTM normalization-stats .npz path.")
    p.add_argument("--checkpoint", default=str(PROJECT_ROOT / "outputs" / "tcn_checkpoint.pt"),
                   help="TCN checkpoint .pt path (bundles weights + scaler + window + arch).")
    p.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "eval_results.csv"),
        help="Per-file metrics CSV output path.",
    )
    return p


def _build_lstm_backend(args) -> "LSTMBackend":
    stats_path = Path(args.stats)
    weights_path = Path(args.weights)
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    stats = np.load(stats_path)
    mean = np.asarray(stats["mean"], dtype=float)
    std = np.asarray(stats["std"], dtype=float)
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError("Invalid stats file: expected mean/std with shape (3,).")

    _, model = build_lstm_model(args.sequence_length, args.units, LEARNING_RATE)
    model.load_weights(str(weights_path))
    print(f"[EVAL] loaded LSTM weights: {weights_path}", flush=True)
    return LSTMBackend(model, mean, std, args.sequence_length)


def _build_tcn_backend(args) -> "TCNBackend":
    from tcn import tcn_forecast  # lazy: only the tcn path needs torch

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"TCN checkpoint not found: {ckpt_path}")
    model, scaler, window = tcn_forecast.load_checkpoint(str(ckpt_path))
    rollout = "physics" if args.physics else "native"
    print(f"[EVAL] loaded TCN checkpoint: {ckpt_path} (window={window}, rollout={rollout})",
          flush=True)
    return TCNBackend(model, scaler, window, physics=args.physics)


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.physics and args.model != "tcn":
        raise SystemExit("--physics only applies to --model tcn.")

    backend = _build_tcn_backend(args) if args.model == "tcn" else _build_lstm_backend(args)

    if args.horizons:
        if args.plots:
            print("[EVAL] note: --plots is ignored with --horizons; run without "
                  "--horizons to render per-file plots.", flush=True)
        horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
        evaluate_horizons(
            backend=backend,
            test_dir=Path(args.test_dir),
            input_rows=args.input_rows,
            limit=args.limit,
            horizons=horizons,
        )
        return

    plot_horizons = None
    if args.plots:
        plot_horizons = [int(x) for x in args.plot_horizons.split(",") if x.strip()]

    rows, aggregate = evaluate_dataset(
        backend=backend,
        test_dir=Path(args.test_dir),
        input_rows=args.input_rows,
        max_steps=args.max_steps,
        limit=args.limit,
        plot_horizons=plot_horizons,
        plot_dir=Path(args.plot_dir),
        plot_max_rows=args.plot_max_rows,
    )

    _write_csv(Path(args.output), rows, aggregate)

    print("\n=== Evaluation summary "
          f"({len(rows)} files, ~{aggregate['steps']} steps each) ===")
    print(f"  signed   RMSE={aggregate['rmse']:.4f}   MAE={aggregate['mae']:.4f}")
    print(f"  |omega|  RMSE={aggregate['rmse_abs']:.4f}   MAE={aggregate['mae_abs']:.4f}   <- magnitude (sign-invariant)")
    print(f"  mirror   RMSE={aggregate['rmse_mirror']:.4f}   <- forgives a global direction flip")
    print(f"  corr (signed) = {aggregate['corr']:.3f}")
    print(f"\n  PRIMARY comparison metric -> |omega| RMSE = {aggregate['rmse_abs']:.4f}")
    print(f"  per-file metrics written to: {args.output}")


if __name__ == "__main__":
    main()
