#!/usr/bin/env python3
"""LSTM training/prediction workflow for Lorenz water-wheel time series.

Two modes (set via the MODE constant below):
1) train   -> train on many CSV files in data/train/, save weights + normalization stats
2) predict -> load weights, take the first PREDICT_INPUT_POINTS rows of data/test.csv
              as history, and export PREDICT_OUTPUT_POINTS future points

Run from the project root, e.g.:  python src/lstm_train_predict.py
"""

import os
import signal
from glob import glob
from pathlib import Path

import numpy as np

# Project root = the parent of this file's src/ directory. Data and output
# paths are resolved against it so the script works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =========================
# Editable configuration
# =========================
# Use "train" to train on many files, or "predict" to evaluate one file with saved weights.
MODE = "train"  # "train" | "predict"

# CSV format for all files (no header):
# angular_velocity,sin,cos,epoch

# -------- train mode settings --------
# Pattern that matches many training files (example: around 25 files).
TRAIN_FILES_GLOB = str(PROJECT_ROOT / "data" / "train" / "*.csv")

# How much of each file is used for training windows.
TRAIN_RATIO = 1

# Where to save learned model parameters and normalization stats.
WEIGHTS_OUTPUT_PATH = str(PROJECT_ROOT / "outputs" / "lstm_weights.weights.h5")
STATS_OUTPUT_PATH = str(PROJECT_ROOT / "outputs" / "lstm_stats.npz")

# -------- predict mode settings --------
# File to evaluate.
PREDICT_FILE_PATH = str(PROJECT_ROOT / "data" / "test.csv")

# Use the first N points from input file as prediction history.
PREDICT_INPUT_POINTS = 9000

# Number of future points to generate from that history.
PREDICT_OUTPUT_POINTS = 1800

# Print rollout progress every N predicted points (0 disables progress logs).
PREDICT_LOG_EVERY_STEPS = 100
# Where to save prediction results from predict mode.
PREDICT_OUTPUT_PATH = str(PROJECT_ROOT / "outputs" / "lstm_predictions.csv")

# Previously saved weights/stats paths to load.
WEIGHTS_INPUT_PATH = str(PROJECT_ROOT / "outputs" / "lstm_weights.weights.h5")
STATS_INPUT_PATH = str(PROJECT_ROOT / "outputs" / "lstm_stats.npz")

# -------- model/training settings --------
RANDOM_SEED = 67
SEQUENCE_LENGTH = 256
LSTM_UNITS = 256
# Used when TRAIN_INDEFINITELY=False.
EPOCHS = 200
# When True, keep training until Ctrl+C/SIGTERM is received.
TRAIN_INDEFINITELY = False
BATCH_SIZE = 256
LEARNING_RATE = 4e-4
LOG_EVERY_EPOCHS = 1
TRAIN_VALIDATION_RATIO = 0.08
EARLY_STOPPING_PATIENCE = 25
LR_REDUCE_PATIENCE = 8
LR_REDUCE_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-7
ADAM_CLIPNORM = 1.0
TRAIN_SHUFFLE_BUFFER = 30000
STANDARDIZE_CHANNELS = (0,)

# Minimum rows required for a file to participate in training/evaluation.
MIN_ROWS_PER_FILE = 40
WINDOW_STRIDE = 4   # subsample training windows


_TF_RUNTIME_CONFIGURED = False


def _parse_sm_capability(capability) -> tuple[int, int] | None:
    """Parse values like 'sm_89', 'compute_90', or (8, 9) into (major, minor)."""
    if isinstance(capability, (tuple, list)) and len(capability) >= 2:
        try:
            return int(capability[0]), int(capability[1])
        except (TypeError, ValueError):
            return None

    if isinstance(capability, str):
        value = capability.strip().lower()
        for prefix in ("sm_", "compute_"):
            if value.startswith(prefix):
                value = value[len(prefix) :]
                break
        if "." in value:
            major_str, minor_str = value.split(".", 1)
        elif len(value) >= 2 and value.isdigit():
            major_str, minor_str = value[:-1], value[-1]
        else:
            return None
        try:
            return int(major_str), int(minor_str)
        except ValueError:
            return None
    return None


def configure_tensorflow_runtime(tf):
    """Configure TensorFlow device runtime for broad NVIDIA compatibility."""
    global _TF_RUNTIME_CONFIGURED
    if _TF_RUNTIME_CONFIGURED:
        return

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        _TF_RUNTIME_CONFIGURED = True
        return

    # Always enable memory growth for desktop GPUs.
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    try:
        build_info = tf.sysconfig.get_build_info()
    except Exception:
        build_info = {}

    build_caps_raw = build_info.get("cuda_compute_capabilities", [])
    if isinstance(build_caps_raw, str):
        build_caps_raw = [x.strip() for x in build_caps_raw.split(",") if x.strip()]
    build_caps = [
        cap
        for cap in (_parse_sm_capability(x) for x in build_caps_raw)
        if cap is not None
    ]
    max_supported = max(build_caps) if build_caps else None

    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    tf.config.optimizer.set_jit(True)

    if max_supported is not None:
        unsupported = []
        for gpu in gpus:
            details = tf.config.experimental.get_device_details(gpu)
            cc = _parse_sm_capability(details.get("compute_capability"))
            if cc is not None and cc > max_supported:
                unsupported.append((gpu.name, cc))

        if unsupported:
            print(
                "[TF] Detected GPU architecture newer than this TensorFlow build supports; "
                "falling back to CPU runtime for stability.",
                flush=True,
            )
            for name, cc in unsupported:
                print(f"[TF] GPU {name} compute capability={cc[0]}.{cc[1]}", flush=True)
            tf.config.set_visible_devices([], "GPU")

    _TF_RUNTIME_CONFIGURED = True


def load_lorenz_waterwheel_csv(path: str) -> np.ndarray:
    """Load one CSV and validate shape."""
    data = np.genfromtxt(path, delimiter=",", dtype=float)
    if data.size == 0:
        raise ValueError(f"CSV is empty or unreadable: {path}")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 4:
        raise ValueError(
            f"Expected 4 columns [angular_velocity, sin, cos, epoch], got {data.shape[1]} in {path}."
        )
    if len(data) < MIN_ROWS_PER_FILE:
        raise ValueError(f"Need at least {MIN_ROWS_PER_FILE} rows in {path}.")
    return np.asarray(data, dtype=float)


def standardize_fit(train_2d: np.ndarray):
    """Compute mean/std only for selected channels."""
    mean = np.zeros(3, dtype=float)
    std = np.ones(3, dtype=float)

    channels = np.array(STANDARDIZE_CHANNELS)

    mean[channels] = train_2d[:, channels].mean(axis=0)
    std[channels] = train_2d[:, channels].std(axis=0)

    std[std == 0] = 1.0

    return mean, std


def standardize_apply(values: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Apply z-score only to selected channels."""
    out = values.copy()

    channels = np.array(STANDARDIZE_CHANNELS)

    out[:, channels] = (out[:, channels] - mean[channels]) / std[channels]

    return out


def renormalize_sincos(state: np.ndarray):
    """
    Force sin/cos back onto unit circle.
    state shape: (...,3)
    """

    sin_val = state[..., 1]
    cos_val = state[..., 2]

    radius = np.sqrt(sin_val * sin_val + cos_val * cos_val)

    mask = radius > 1e-8

    sin_val[mask] /= radius[mask]
    cos_val[mask] /= radius[mask]

    # fallback if network outputs exactly [0,0]
    sin_val[~mask] = 0.0
    cos_val[~mask] = 1.0

    state[..., 1] = sin_val
    state[..., 2] = cos_val

    return state


def standardize_inverse(values: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Inverse z-score only for standardized channels."""
    out = values.copy()

    channels = np.array(STANDARDIZE_CHANNELS)

    out[:, channels] = out[:, channels] * std[channels] + mean[channels]

    return out


def build_windows(series_wsc: np.ndarray, seq_len: int):
    """Build [seq_len,3] -> [3] windows for next-step prediction."""
    x, y = [], []
    for i in range(len(series_wsc) - seq_len):
        x.append(series_wsc[i : i + seq_len])
        y.append(series_wsc[i + seq_len])
    if not x:
        raise ValueError("Not enough points for chosen sequence length.")
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def build_lstm_model(seq_len: int, units: int, learning_rate: float):
    """Create and compile LSTM model architecture."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is required. Install with: pip install tensorflow"
        ) from exc

    configure_tensorflow_runtime(tf)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(seq_len, 3)),
            tf.keras.layers.LSTM(
                units,
                return_sequences=True,
                dropout=0.10,
            ),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.LSTM(
                units,
                dropout=0.10,
            ),
            tf.keras.layers.Dense(
                units // 2,
                activation="swish",
            ),
            tf.keras.layers.Dense(
                units // 4,
                activation="swish",
            ),
            tf.keras.layers.Dense(3, dtype="float32"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=ADAM_CLIPNORM,
        ),
        loss="mse",
    )
    return tf, model


def run_autoregressive_prediction(
    model,
    input_wsc,
    forecast_points,
    seq_len,
    mean,
    std,
    log_every_steps=0,
):

    wsc_scaled = standardize_apply(input_wsc, mean, std).astype(np.float32)

    rollout_seq = wsc_scaled[-seq_len:].copy().reshape(1, seq_len, 3)

    preds = np.zeros((forecast_points, 3))

    for i in range(forecast_points):
        y_next_scaled = model(rollout_seq, training=False).numpy()[0]

        y_real = standardize_inverse(y_next_scaled.reshape(1, 3), mean, std)[0]

        y_real = renormalize_sincos(y_real.reshape(1, 3))[0]

        preds[i] = y_real

        y_scaled_corrected = standardize_apply(y_real.reshape(1, 3), mean, std)[0]

        rollout_seq[0, :-1] = rollout_seq[0, 1:]
        rollout_seq[0, -1] = y_scaled_corrected

        if log_every_steps:
            step = i + 1
            if step % log_every_steps == 0:
                print(f"[PREDICT] {step}/{forecast_points}", flush=True)

    return preds


def save_prediction_file(path: str, pred_epochs: np.ndarray, preds: np.ndarray):
    """Save predict-mode outputs as CSV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out = np.column_stack([pred_epochs, preds])
    np.savetxt(
        out_path,
        out,
        delimiter=",",
        fmt="%.10f",
        header="epoch,pred_angular_velocity,pred_sin,pred_cos",
        comments="",
    )
    print(f"[PREDICT] predictions saved: {out_path}", flush=True)


def _compute_train_cut(length: int) -> int:
    cut = int(length * TRAIN_RATIO)
    return max(SEQUENCE_LENGTH + 5, min(cut, length - 5))


def build_future_epochs(input_epochs: np.ndarray, output_points: int) -> np.ndarray:
    """Build epochs for future forecast points using the input epoch step."""
    if len(input_epochs) < 2:
        step = 1.0
    else:
        diffs = np.diff(input_epochs)
        step = float(np.median(diffs))
        if step == 0:
            step = 1.0
    start = float(input_epochs[-1]) + step
    return start + np.arange(output_points, dtype=float) * step


def build_window_datasets(tf, file_splits, mean, std, seq_len: int):
    """Build streamed tf.data datasets for train and validation windows."""
    if not (0.0 < TRAIN_VALIDATION_RATIO < 0.5):
        raise ValueError("TRAIN_VALIDATION_RATIO must be in (0, 0.5).")

    def make_generator(subset):

        def gen():

            for scaled, cut, train_windows, val_windows in file_splits:
                total_windows = cut - seq_len

                if subset == "train":
                    start = 0
                    end = train_windows
                else:
                    start = total_windows - val_windows
                    end = total_windows

                step = WINDOW_STRIDE if subset == "train" else 1
                for i in range(start, end, step):
                    yield (scaled[i : i + seq_len], scaled[i + seq_len])

        return gen

    signature = (
        tf.TensorSpec(shape=(seq_len, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(3,), dtype=tf.float32),
    )

    train_ds = tf.data.Dataset.from_generator(
        make_generator("train"),
        output_signature=signature,
    )
    train_ds = train_ds.shuffle(
        buffer_size=max(BATCH_SIZE * 4, TRAIN_SHUFFLE_BUFFER),
        seed=RANDOM_SEED,
        reshuffle_each_iteration=True,
    ).batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        make_generator("val"),
        output_signature=signature,
    ).batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def train_mode():
    """Train on many files and persist weights + normalization stats."""
    # Ensure the output directory exists before any checkpoint/weights are saved.
    Path(WEIGHTS_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    train_files = sorted(glob(TRAIN_FILES_GLOB))
    if not train_files:
        raise ValueError(f"No files matched TRAIN_FILES_GLOB: {TRAIN_FILES_GLOB}")

    print(f"[TRAIN] matched files: {len(train_files)}")

    series = []
    for path in train_files:
        try:
            x_all = load_lorenz_waterwheel_csv(path)
            cut = _compute_train_cut(len(x_all))
            if cut <= SEQUENCE_LENGTH:
                raise ValueError("training split is too short")
            series.append((path, x_all, cut))
        except Exception as exc:
            print(f"[TRAIN] skipped {path}: {exc}", flush=True)

    if not series:
        raise ValueError("No valid files remained after validation.")

    # Fit one global scaler from all training segments.
    train_chunks = [x_all[:cut, :3] for _, x_all, cut in series]
    train_all_wsc = np.vstack(train_chunks)
    mean, std = standardize_fit(train_all_wsc)
    file_splits = []
    total_train_windows = 0
    total_val_windows = 0

    for path, x_all, cut in series:
        total_windows = cut - SEQUENCE_LENGTH

        if total_windows < 2:
            print(f"[TRAIN] skipped {path}: not enough windows after split", flush=True)
            continue

        val_windows = max(1, int(total_windows * TRAIN_VALIDATION_RATIO))
        train_windows = total_windows - val_windows

        if train_windows < 1:
            print(f"[TRAIN] skipped {path}: train windows became empty", flush=True)
            continue

        scaled = standardize_apply(
            x_all[:, :3],
            mean,
            std,
        ).astype(np.float32)

        file_splits.append((scaled, cut, train_windows, val_windows))

        total_train_windows += train_windows
        total_val_windows += val_windows

    if not file_splits:
        raise ValueError("No valid files produced train/validation windows.")

    print(
        "[TRAIN] usable files: "
        f"{len(file_splits)}, train windows: {total_train_windows}, "
        f"val windows: {total_val_windows}"
    )

    tf, model = build_lstm_model(SEQUENCE_LENGTH, LSTM_UNITS, LEARNING_RATE)
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    train_ds, val_ds = build_window_datasets(
        tf=tf,
        file_splits=file_splits,
        mean=mean,
        std=std,
        seq_len=SEQUENCE_LENGTH,
    )

    class LSTMProgress(tf.keras.callbacks.Callback):
        """Print percentage + loss during training."""

        def __init__(self, total_epochs: int | None, every_epochs: int):
            super().__init__()
            self.total_epochs = total_epochs
            self.every_epochs = max(1, every_epochs)

        def on_epoch_end(self, epoch, logs=None):
            current = epoch + 1
            should_log = current % self.every_epochs == 0
            if self.total_epochs is not None:
                should_log = should_log or current == self.total_epochs
            if should_log:
                loss = None if logs is None else logs.get("loss")
                if self.total_epochs is None:
                    prefix = f"[LSTM] epoch {current}"
                else:
                    pct = int(round((current / self.total_epochs) * 100))
                    prefix = f"[LSTM] epoch {current}/{self.total_epochs} ({pct}%)"
                if loss is None:
                    print(prefix, flush=True)
                else:
                    print(f"{prefix} loss={loss:.6f}", flush=True)

    class GracefulStopper(tf.keras.callbacks.Callback):
        """Handle SIGINT/SIGTERM and stop training so weights can be saved."""

        def __init__(self):
            super().__init__()
            self.stop_requested = False
            self._signal_count = 0
            self._old_handlers = {}

        def install(self):
            for sig in (signal.SIGINT, signal.SIGTERM):
                self._old_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle_signal)

        def restore(self):
            for sig, handler in self._old_handlers.items():
                signal.signal(sig, handler)
            self._old_handlers = {}

        def _handle_signal(self, signum, frame):
            del frame  # unused
            self._signal_count += 1
            if self._signal_count == 1:
                self.stop_requested = True
                signame = signal.Signals(signum).name
                print(
                    f"\n[LSTM] {signame} received. Stopping training and saving weights...",
                    flush=True,
                )
                return
            raise KeyboardInterrupt

    effective_epochs = 2_000_000_000 if TRAIN_INDEFINITELY else EPOCHS
    progress_total = None if TRAIN_INDEFINITELY else EPOCHS
    stop_cb = GracefulStopper()
    callbacks = [
        LSTMProgress(progress_total, LOG_EVERY_EPOCHS),
        stop_cb,
        tf.keras.callbacks.ModelCheckpoint(
            filepath=WEIGHTS_OUTPUT_PATH,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    if not TRAIN_INDEFINITELY:
        callbacks.insert(
            2,
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                min_delta=1e-6,
            ),
        )

    mode = "indefinite (Ctrl+C to stop)" if TRAIN_INDEFINITELY else f"{EPOCHS} epochs"
    print(
        f"[LSTM] training started: mode={mode}, train_windows={total_train_windows}",
        flush=True,
    )
    stop_cb.install()
    interrupted = False
    try:
        model.fit(
            train_ds,
            epochs=effective_epochs,
            validation_data=val_ds,
            verbose=0,
            callbacks=callbacks,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("[LSTM] forced stop requested; saving current weights.", flush=True)
    finally:
        stop_cb.restore()

    if stop_cb.stop_requested:
        print("[LSTM] graceful stop completed.", flush=True)
    elif interrupted:
        print("[LSTM] training interrupted.", flush=True)
    else:
        print("[LSTM] training completed", flush=True)

    model.save_weights(WEIGHTS_OUTPUT_PATH)
    np.savez(STATS_OUTPUT_PATH, mean=mean, std=std)
    print(f"[TRAIN] weights saved: {WEIGHTS_OUTPUT_PATH}")
    print(f"[TRAIN] stats saved:   {STATS_OUTPUT_PATH}")


def predict_mode():
    """Load model/stats and export future predictions from a fixed input window."""
    weights_path = Path(WEIGHTS_INPUT_PATH)
    stats_path = Path(STATS_INPUT_PATH)
    print(
        f"[PREDICT] file={PREDICT_FILE_PATH}, "
        f"input_points={PREDICT_INPUT_POINTS}, output_points={PREDICT_OUTPUT_POINTS}",
        flush=True,
    )
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    x_all = load_lorenz_waterwheel_csv(PREDICT_FILE_PATH)
    input_points = int(PREDICT_INPUT_POINTS)
    output_points = int(PREDICT_OUTPUT_POINTS)
    if input_points <= 0:
        raise ValueError("PREDICT_INPUT_POINTS must be > 0.")
    if output_points <= 0:
        raise ValueError("PREDICT_OUTPUT_POINTS must be > 0.")
    if len(x_all) < input_points:
        raise ValueError(
            f"Input file has {len(x_all)} rows, but PREDICT_INPUT_POINTS={input_points}."
        )
    if input_points < SEQUENCE_LENGTH:
        raise ValueError(
            "PREDICT_INPUT_POINTS must be at least SEQUENCE_LENGTH for prediction."
        )
    input_block = x_all[:input_points]

    stats = np.load(stats_path)
    mean = np.asarray(stats["mean"], dtype=float)
    std = np.asarray(stats["std"], dtype=float)
    if mean.shape != (3,) or std.shape != (3,):
        raise ValueError("Invalid stats file: expected mean/std with shape (3,).")
    print(f"[PREDICT] loaded stats: {stats_path}", flush=True)

    _, model = build_lstm_model(SEQUENCE_LENGTH, LSTM_UNITS, LEARNING_RATE)
    model.load_weights(str(weights_path))
    print(f"[PREDICT] loaded weights: {weights_path}")
    print(
        f"[PREDICT] using first {input_points} rows as input history; "
        f"forecasting next {output_points} rows",
        flush=True,
    )

    preds = run_autoregressive_prediction(
        model=model,
        input_wsc=input_block[:, :3],
        forecast_points=output_points,
        seq_len=SEQUENCE_LENGTH,
        mean=mean,
        std=std,
        log_every_steps=PREDICT_LOG_EVERY_STEPS,
    )

    pred_epochs = build_future_epochs(input_block[:, 3], output_points)

    save_prediction_file(
        path=PREDICT_OUTPUT_PATH,
        pred_epochs=pred_epochs,
        preds=preds,
    )


def main():
    if MODE == "train":
        train_mode()
    elif MODE == "predict":
        predict_mode()
    else:
        raise ValueError(f"Unsupported MODE='{MODE}'. Use 'train' or 'predict'.")


if __name__ == "__main__":
    main()
