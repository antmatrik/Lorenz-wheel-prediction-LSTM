#!/usr/bin/env python3
"""Temporal Convolutional Network (TCN) forecaster for Lorenz water-wheel data.

A PyTorch alternative to the TensorFlow LSTM in ``lstm_train_predict.py``, adapted
from https://github.com/Tartas37/ABFS-2026-Lorenzs-wheel-ai-model. The network is a
stack of dilated causal-convolution residual blocks with a linear head. Unlike the
LSTM -- which predicts only the next angular velocity and reconstructs sin/cos by
integrating the wheel angle -- the TCN predicts the full next state
``[angular_velocity, sin, cos]`` directly and feeds it back autoregressively.

Two modes, selected with --mode (default from the MODE constant below):
1) train   -> train on many CSV files in data/train/, save a .pt checkpoint to outputs/
2) predict -> load the checkpoint, take the first PREDICT_INPUT_POINTS rows of
              data/test.csv as history, and export PREDICT_OUTPUT_POINTS future points

The checkpoint bundles the model weights, the (global) normalization scaler, the input
window length and the architecture, so ``evaluate.py --model tcn`` can rebuild it without
extra flags.

Adaptations vs upstream (kept faithful otherwise):
* Reads the repo's headerless 4-column CSV directly and derives ``dt`` from the raw epoch
  column -- upstream's CSV->npy step overwrites epoch with diffs and then diffs again,
  producing an unreliable ``dt_median``; here that never happens.
* Fits ONE global scaler across all training files (like the LSTM) instead of averaging
  per-file scalers, so training and inference share the same coordinate system.

Run from the project root, e.g.:

    python src/tcn/tcn_forecast.py --mode train            # full training
    python src/tcn/tcn_forecast.py --mode train --sanity   # quick check: 1 epoch, 1 file
    python src/tcn/tcn_forecast.py --mode predict
    python src/tcn/tcn_forecast.py --help
"""

import argparse
import time
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# Project root = two levels above this file's src/<group>/ directory. Data and output
# paths are resolved against it so the script works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =========================
# Editable configuration
# =========================
MODE = "train"  # "train" | "predict"

# CSV format for all files (no header): angular_velocity,sin,cos,epoch

# -------- train mode settings --------
TRAIN_FILES_GLOB = str(PROJECT_ROOT / "data" / "train" / "*.csv")
# Limit training to the first N matched files (None or 0 = use all).
MAX_TRAIN_FILES = None

CHECKPOINT_OUTPUT_PATH = str(PROJECT_ROOT / "outputs" / "tcn_checkpoint.pt")

# -------- predict mode settings --------
PREDICT_FILE_PATH = str(PROJECT_ROOT / "data" / "test.csv")
PREDICT_INPUT_POINTS = 9000
PREDICT_OUTPUT_POINTS = 1800
PREDICT_LOG_EVERY_STEPS = 100
PREDICT_OUTPUT_PATH = str(PROJECT_ROOT / "outputs" / "tcn_predictions.csv")

CHECKPOINT_INPUT_PATH = str(PROJECT_ROOT / "outputs" / "tcn_checkpoint.pt")

# -------- model/training settings --------
RANDOM_SEED = 0
WINDOW = 64
NUM_CHANNELS = (64, 64, 64, 64)
KERNEL_SIZE = 3
DROPOUT = 0.15
# Autoregressive rollout horizon used as the training target, with a per-step discount.
HORIZON = 5
HORIZON_DISCOUNT = 0.9
EPOCHS = 60
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
GRAD_CLIP = 1.0
VAL_FRACTION = 0.15
# Learning-rate schedule (ReduceLROnPlateau) + early stopping, both keyed off val loss.
LR_PATIENCE = 5           # halve the LR after this many epochs with no val improvement
LR_FACTOR = 0.5           # LR multiplier applied on plateau
MIN_LR = 1e-7             # floor for the LR schedule
EARLY_STOP_PATIENCE = 10  # stop after this many epochs with no val improvement (0 = never)
MIN_DELTA = 1e-6          # smallest val-loss drop that counts as an improvement
# num_workers for the DataLoader; 0 keeps things simple/portable (safe on macOS/CI).
NUM_WORKERS = 0

# Minimum rows required for a file to participate in training/evaluation.
MIN_ROWS_PER_FILE = 40


# ---------------------------------------------------------------------------
# 1. TCN model
# ---------------------------------------------------------------------------

class Chomp1d(nn.Module):
    """Trim the right-side padding so convolutions stay causal."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Two dilated causal convs + residual connection (a TCN layer)."""

    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.drop1,
            self.conv2, self.chomp2, self.relu2, self.drop2,
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Stack of TemporalBlocks with exponentially growing dilation."""

    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNPredictor(nn.Module):
    """TCN body + linear head: (B, window, n_features) -> (B, n_features)."""

    def __init__(self, n_features, num_channels=(64, 64, 64, 64), kernel_size=3, dropout=0.15):
        super().__init__()
        self.tcn = TemporalConvNet(n_features, list(num_channels), kernel_size, dropout)
        self.head = nn.Linear(num_channels[-1], n_features)

    def forward(self, x):
        # x: (B, window, C) -> conv wants (B, C, window)
        x = x.transpose(1, 2)
        y = self.tcn(x)
        last_step = y[:, :, -1]
        return self.head(last_step)


def _train_rollout(model, x, horizon):
    """Autoregressive rollout used to compute the multi-step training target."""
    preds = []
    buf = x
    for _ in range(horizon):
        pred = model(buf)
        preds.append(pred)
        buf = torch.cat([buf[:, 1:, :], pred.unsqueeze(1)], dim=1)
    return torch.stack(preds, dim=1)


# ---------------------------------------------------------------------------
# 2. Data
# ---------------------------------------------------------------------------

def load_lorenz_waterwheel_csv(path: str) -> np.ndarray:
    """Load one headerless 4-column CSV [angular_velocity, sin, cos, epoch]."""
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


def infer_dt(epochs: np.ndarray) -> float:
    """Median time step (seconds) from the epoch column; robust to gaps."""
    dt = float(np.median(np.diff(epochs))) if len(epochs) >= 2 else 1.0
    if not np.isfinite(dt) or dt == 0:
        dt = 1.0
    return dt


class SequenceDataset(Dataset):
    """Sliding windows over a normalized (N, 3) array -> (window, 3) / (horizon, 3)."""

    def __init__(self, data, window, horizon=1):
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.window = window
        self.horizon = horizon

    def __len__(self):
        return max(0, len(self.data) - self.window - self.horizon + 1)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.window]
        y = self.data[idx + self.window: idx + self.window + self.horizon]
        return x, y


def fit_global_scaler(series):
    """Fit one mean/std over the 3 state channels across all training files."""
    all_states = np.vstack([s[:, :3] for s in series])
    mean = all_states.mean(axis=0)
    std = all_states.std(axis=0)
    std[std == 0] = 1.0
    dt_median = float(np.median([infer_dt(s[:, 3]) for s in series]))
    return {"mean": mean, "std": std, "dt_median": dt_median}


def normalize_state(state: np.ndarray, scaler: dict) -> np.ndarray:
    """(rows, 3) real-unit state -> z-scored with the (3,) scaler."""
    return (np.asarray(state, dtype=float) - scaler["mean"]) / scaler["std"]


def denormalize_state(state_norm: np.ndarray, scaler: dict) -> np.ndarray:
    return np.asarray(state_norm, dtype=float) * scaler["std"] + scaler["mean"]


def build_file_datasets(series, scaler, window, horizon, val_fraction):
    """Per-file chronological train/val split, normalized with the global scaler."""
    train_dss, val_dss = [], []
    for s in series:
        state = normalize_state(s[:, :3], scaler)
        n_val = max(1, int(len(state) * val_fraction))
        train_norm = state[:-n_val]
        # Keep `window` rows of context before the val region so val windows have history.
        val_norm = state[-(n_val + window):]
        tr = SequenceDataset(train_norm, window, horizon)
        va = SequenceDataset(val_norm, window, horizon)
        if len(tr) > 0:
            train_dss.append(tr)
        if len(va) > 0:
            val_dss.append(va)
    return train_dss, val_dss


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------

def train(
    model,
    train_ds,
    val_ds,
    scaler,
    window,
    num_channels,
    kernel_size,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    horizon_discount=HORIZON_DISCOUNT,
    grad_clip=GRAD_CLIP,
    lr_patience=LR_PATIENCE,
    lr_factor=LR_FACTOR,
    min_lr=MIN_LR,
    patience=EARLY_STOP_PATIENCE,
    min_delta=MIN_DELTA,
    num_workers=NUM_WORKERS,
    device=None,
    out_path=CHECKPOINT_OUTPUT_PATH,
):
    """Train with a horizon-discounted autoregressive MSE loss.

    The best-so-far model (lowest val loss) is written to ``out_path`` every time val
    improves, so an interrupted run still leaves the best checkpoint on disk. Stops early
    after ``patience`` epochs with no val improvement (0 disables); ``ReduceLROnPlateau``
    (``lr_patience``/``lr_factor``) shrinks the LR on plateaus, and the current LR is logged.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if len(train_ds) == 0:
        raise ValueError("train_ds has 0 windows.")

    pin_memory = device == "cuda"
    use_amp = device == "cuda"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    have_val = len(val_ds) > 0
    val_loader = (
        DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin_memory)
        if have_val else None
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=lr_factor, patience=lr_patience, min_lr=min_lr
    )
    loss_fn = nn.MSELoss(reduction="none")
    amp_scaler = torch.amp.GradScaler(device, enabled=use_amp)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    epochs_since_improve = 0

    print(
        f"[TCN] device={device} amp={use_amp} train_windows={len(train_ds)} "
        f"val_windows={len(val_ds)} params={sum(p.numel() for p in model.parameters()):,}",
        flush=True,
    )
    if not have_val:
        print("[TCN] no validation windows -> LR schedule + early stopping disabled; "
              "saving the final epoch.", flush=True)

    def _rollout_loss(xb, yb):
        horizon = yb.shape[1]
        weights = horizon_discount ** torch.arange(horizon, device=device)
        pred = _train_rollout(model, xb, horizon)
        per_step = loss_fn(pred, yb).mean(dim=(0, 2))
        return (per_step * weights).sum() / weights.sum()

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t_epoch = time.time()
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=pin_memory)
            yb = yb.to(device, non_blocking=pin_memory)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=use_amp):
                loss = _rollout_loss(xb, yb)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            amp_scaler.step(opt)
            amp_scaler.update()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        lr_now = opt.param_groups[0]["lr"]
        val_str = "val_loss (n/a)"
        improved = False
        if have_val:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device, non_blocking=pin_memory)
                    yb = yb.to(device, non_blocking=pin_memory)
                    val_loss += _rollout_loss(xb, yb).item() * xb.size(0)
            val_loss /= max(1, len(val_ds))
            sched.step(val_loss)
            val_str = f"val_loss {val_loss:.6f}"
            if val_loss < best_val - min_delta:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_since_improve = 0
                improved = True
                # Persist the best model on every improvement, so an interrupted run
                # still leaves the best-so-far checkpoint on disk.
                save_checkpoint(model, scaler, window, num_channels, kernel_size,
                                path=out_path, verbose=False)
            else:
                epochs_since_improve += 1

        print(
            f"[TCN] epoch {epoch:3d}/{epochs} | train_loss {train_loss:.6f} | "
            f"{val_str} | lr {lr_now:.2e} | {time.time() - t_epoch:.2f}s"
            + ("  <- new best (saved)" if improved else ""),
            flush=True,
        )

        lr_after = opt.param_groups[0]["lr"]
        if lr_after < lr_now:
            print(f"[TCN] ReduceLROnPlateau: lr {lr_now:.2e} -> {lr_after:.2e} "
                  f"(no val improvement for {lr_patience} epochs)", flush=True)

        if patience and have_val and epochs_since_improve >= patience:
            print(f"[TCN] early stop at epoch {epoch}: no val improvement for {patience} "
                  f"epochs (best val_loss {best_val:.6f} at epoch {best_epoch}).", flush=True)
            break

    print(f"[TCN] training finished in {time.time() - t_start:.1f}s", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    save_checkpoint(model, scaler, window, num_channels, kernel_size, path=out_path)
    return model, best_val


# ---------------------------------------------------------------------------
# 4. Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(model, scaler, window, num_channels, kernel_size,
                    path=CHECKPOINT_OUTPUT_PATH, verbose=True):
    """Persist weights + scaler + input window + architecture in one .pt file.

    ``verbose=False`` suppresses the log line (used for the frequent per-epoch best saves).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "scaler": {
                "mean": np.asarray(scaler["mean"], dtype=float),
                "std": np.asarray(scaler["std"], dtype=float),
                "dt_median": float(scaler["dt_median"]),
            },
            "window": int(window),
            "n_features": 3,
            "num_channels": list(num_channels),
            "kernel_size": int(kernel_size),
        },
        path,
    )
    if verbose:
        print(f"[TCN] checkpoint saved: {path}", flush=True)


def load_checkpoint(path=CHECKPOINT_INPUT_PATH, device=None):
    """Rebuild the model + scaler + window from a .pt checkpoint."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TCNPredictor(
        n_features=ckpt["n_features"],
        num_channels=tuple(ckpt["num_channels"]),
        kernel_size=ckpt["kernel_size"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt["scaler"], int(ckpt["window"])


# ---------------------------------------------------------------------------
# 5. Rollout adapters (shared by predict mode and evaluate.py)
# ---------------------------------------------------------------------------

def rollout_native_batched(model, seeds_norm, horizon, scaler, device=None):
    """Advance N series together with the native 3-channel rollout.

    seeds_norm: (N, window, 3) already normalized with ``scaler``.
    Returns real-unit predictions of shape (N, horizon, 3).
    """
    device = device or next(model.parameters()).device
    mean = np.asarray(scaler["mean"], dtype=float)
    std = np.asarray(scaler["std"], dtype=float)

    roll = torch.as_tensor(np.asarray(seeds_norm, dtype=np.float32), device=device).clone()
    n = roll.shape[0]
    preds = np.zeros((n, horizon, 3))

    model.eval()
    with torch.no_grad():
        for t in range(horizon):
            out = model(roll)  # (N, 3) normalized
            out_np = out.detach().cpu().numpy()
            preds[:, t, :] = out_np * std + mean
            roll = torch.cat([roll[:, 1:, :], out.unsqueeze(1)], dim=1)
    return preds


def rollout_physics_batched(model, seeds_norm, theta0, w_prev, dt, horizon, scaler, device=None):
    """Physics rollout: use only the model's omega output, reconstruct sin/cos.

    Mirrors ``evaluate.run_batched_rollout`` so the TCN can be scored on exactly the
    same footing as the LSTM (architecture isolated from rollout strategy). Returns
    predicted angular velocity, shape (N, horizon).
    """
    device = device or next(model.parameters()).device
    mean = np.asarray(scaler["mean"], dtype=float)
    std = np.asarray(scaler["std"], dtype=float)

    roll = torch.as_tensor(np.asarray(seeds_norm, dtype=np.float32), device=device).clone()
    theta = np.asarray(theta0, dtype=float).copy()
    w_prev = np.asarray(w_prev, dtype=float).copy()
    dt = np.asarray(dt, dtype=float)
    n = roll.shape[0]
    preds_w = np.zeros((n, horizon))

    model.eval()
    with torch.no_grad():
        for t in range(horizon):
            out = model(roll).detach().cpu().numpy()  # (N, 3) normalized
            w_next = out[:, 0] * std[0] + mean[0]

            theta = theta + 0.5 * (w_prev + w_next) * dt
            sin_next = np.sin(theta)
            cos_next = np.cos(theta)
            preds_w[:, t] = w_next

            nxt = normalize_state(
                np.stack([w_next, sin_next, cos_next], axis=1), scaler
            ).astype(np.float32)
            roll = torch.cat(
                [roll[:, 1:, :], torch.as_tensor(nxt, device=device).unsqueeze(1)], dim=1
            )
            w_prev = w_next
    return preds_w


# ---------------------------------------------------------------------------
# 6. Modes
# ---------------------------------------------------------------------------

def build_future_epochs(input_epochs: np.ndarray, output_points: int) -> np.ndarray:
    """Future epochs for forecast points, using the median input epoch step."""
    if len(input_epochs) < 2:
        step = 1.0
    else:
        step = float(np.median(np.diff(input_epochs)))
        if step == 0:
            step = 1.0
    start = float(input_epochs[-1]) + step
    return start + np.arange(output_points, dtype=float) * step


def train_mode():
    """Train on many files and persist weights + normalization stats."""
    train_files = sorted(glob(TRAIN_FILES_GLOB))
    if not train_files:
        raise ValueError(f"No files matched TRAIN_FILES_GLOB: {TRAIN_FILES_GLOB}")
    if MAX_TRAIN_FILES:
        train_files = train_files[:MAX_TRAIN_FILES]

    print(f"[TCN] matched files: {len(train_files)}", flush=True)

    series = []
    for path in train_files:
        try:
            series.append(load_lorenz_waterwheel_csv(path))
        except Exception as exc:
            print(f"[TCN] skipped {path}: {exc}", flush=True)
    if not series:
        raise ValueError("No valid training files remained.")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    scaler = fit_global_scaler(series)
    train_dss, val_dss = build_file_datasets(
        series, scaler, WINDOW, HORIZON, VAL_FRACTION
    )
    if not train_dss:
        raise ValueError("No training windows produced; check WINDOW/HORIZON vs file lengths.")

    combined_train = ConcatDataset(train_dss)
    combined_val = ConcatDataset(val_dss) if val_dss else ConcatDataset([])
    print(
        f"[TCN] usable files: {len(train_dss)} | train windows: {len(combined_train)} | "
        f"val windows: {len(combined_val)}",
        flush=True,
    )

    model = TCNPredictor(
        n_features=3, num_channels=NUM_CHANNELS, kernel_size=KERNEL_SIZE, dropout=DROPOUT
    )
    train(
        model=model,
        train_ds=combined_train,
        val_ds=combined_val,
        scaler=scaler,
        window=WINDOW,
        num_channels=NUM_CHANNELS,
        kernel_size=KERNEL_SIZE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        lr_patience=LR_PATIENCE,
        lr_factor=LR_FACTOR,
        min_lr=MIN_LR,
        patience=EARLY_STOP_PATIENCE,
        min_delta=MIN_DELTA,
        out_path=CHECKPOINT_OUTPUT_PATH,
    )


def predict_mode():
    """Load the checkpoint and export a native autoregressive forecast."""
    ckpt_path = Path(CHECKPOINT_INPUT_PATH)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    x_all = load_lorenz_waterwheel_csv(PREDICT_FILE_PATH)
    input_points = int(PREDICT_INPUT_POINTS)
    output_points = int(PREDICT_OUTPUT_POINTS)
    if input_points <= 0 or output_points <= 0:
        raise ValueError("PREDICT_INPUT_POINTS and PREDICT_OUTPUT_POINTS must be > 0.")
    if len(x_all) < input_points:
        raise ValueError(f"Input file has {len(x_all)} rows, need {input_points}.")

    model, scaler, window = load_checkpoint(CHECKPOINT_INPUT_PATH)
    if input_points < window:
        raise ValueError(f"PREDICT_INPUT_POINTS must be >= window ({window}).")
    print(f"[TCN] loaded checkpoint: {ckpt_path} (window={window})", flush=True)

    input_block = x_all[:input_points]
    seed_norm = normalize_state(input_block[-window:, :3], scaler).astype(np.float32)

    print(
        f"[TCN] using first {input_points} rows as history; forecasting {output_points} rows",
        flush=True,
    )
    preds = rollout_native_batched(
        model, seed_norm[None, ...], output_points, scaler
    )[0]  # (output_points, 3)

    pred_epochs = build_future_epochs(input_block[:, 3], output_points)
    out = np.column_stack([pred_epochs, preds])
    Path(PREDICT_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        PREDICT_OUTPUT_PATH, out, delimiter=",", fmt="%.10f",
        header="epoch,pred_angular_velocity,pred_sin,pred_cos", comments="",
    )
    print(f"[TCN] predictions saved: {PREDICT_OUTPUT_PATH}", flush=True)


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=["train", "predict"], default=MODE,
                   help=f"Run mode (default: {MODE}).")
    p.add_argument("--sanity", action="store_true",
                   help="Quick end-to-end check: 1 epoch on 1 file.")

    g = p.add_argument_group("training")
    g.add_argument("--epochs", type=int, help=f"Training epochs (default: {EPOCHS}).")
    g.add_argument("--batch-size", type=int, help=f"Batch size (default: {BATCH_SIZE}).")
    g.add_argument("--window", type=int, help=f"Input window length (default: {WINDOW}).")
    g.add_argument("--channels", help=f"TCN channels per layer, comma-separated "
                                       f"(default: {','.join(str(c) for c in NUM_CHANNELS)}).")
    g.add_argument("--kernel-size", type=int, help=f"Conv kernel size (default: {KERNEL_SIZE}).")
    g.add_argument("--horizon", type=int, help=f"Train rollout horizon (default: {HORIZON}).")
    g.add_argument("--learning-rate", type=float, help=f"Adam learning rate (default: {LEARNING_RATE}).")
    g.add_argument("--patience", type=int,
                   help=f"Early stop after N epochs with no val improvement; 0 disables "
                        f"(default: {EARLY_STOP_PATIENCE}).")
    g.add_argument("--lr-patience", type=int,
                   help=f"ReduceLROnPlateau: shrink LR after N stagnant epochs (default: {LR_PATIENCE}).")
    g.add_argument("--lr-factor", type=float,
                   help=f"ReduceLROnPlateau: LR multiplier on plateau (default: {LR_FACTOR}).")
    g.add_argument("--max-train-files", type=int, help="Use only the first N training files (0 = all).")
    g.add_argument("--train-glob", help="Override the training-file glob pattern.")

    g = p.add_argument_group("predict")
    g.add_argument("--predict-file", help=f"Input CSV for predict mode (default: {PREDICT_FILE_PATH}).")
    g.add_argument("--predict-input-points", type=int, help=f"History points (default: {PREDICT_INPUT_POINTS}).")
    g.add_argument("--predict-output-points", type=int, help=f"Future points (default: {PREDICT_OUTPUT_POINTS}).")
    g.add_argument("--predict-output", help="Output CSV path for predictions.")

    g = p.add_argument_group("artifacts")
    g.add_argument("--checkpoint", help="Checkpoint path (train output / predict input).")
    return p


def _apply_overrides(args) -> None:
    global MODE, EPOCHS, BATCH_SIZE, WINDOW, NUM_CHANNELS, KERNEL_SIZE, HORIZON, LEARNING_RATE
    global MAX_TRAIN_FILES, TRAIN_FILES_GLOB, EARLY_STOP_PATIENCE, LR_PATIENCE, LR_FACTOR
    global PREDICT_FILE_PATH, PREDICT_INPUT_POINTS, PREDICT_OUTPUT_POINTS, PREDICT_OUTPUT_PATH
    global CHECKPOINT_OUTPUT_PATH, CHECKPOINT_INPUT_PATH

    MODE = args.mode
    if args.sanity:
        EPOCHS = 1
        MAX_TRAIN_FILES = 1

    if args.epochs is not None:
        EPOCHS = args.epochs
    if args.patience is not None:
        EARLY_STOP_PATIENCE = args.patience
    if args.lr_patience is not None:
        LR_PATIENCE = args.lr_patience
    if args.lr_factor is not None:
        LR_FACTOR = args.lr_factor
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    if args.window is not None:
        WINDOW = args.window
    if args.channels is not None:
        NUM_CHANNELS = tuple(int(c) for c in args.channels.split(","))
    if args.kernel_size is not None:
        KERNEL_SIZE = args.kernel_size
    if args.horizon is not None:
        HORIZON = args.horizon
    if args.learning_rate is not None:
        LEARNING_RATE = args.learning_rate
    if args.max_train_files is not None:
        MAX_TRAIN_FILES = args.max_train_files
    if args.train_glob is not None:
        TRAIN_FILES_GLOB = args.train_glob

    if args.predict_file is not None:
        PREDICT_FILE_PATH = args.predict_file
    if args.predict_input_points is not None:
        PREDICT_INPUT_POINTS = args.predict_input_points
    if args.predict_output_points is not None:
        PREDICT_OUTPUT_POINTS = args.predict_output_points
    if args.predict_output is not None:
        PREDICT_OUTPUT_PATH = args.predict_output

    if args.checkpoint is not None:
        CHECKPOINT_OUTPUT_PATH = CHECKPOINT_INPUT_PATH = args.checkpoint


def main():
    args = _build_arg_parser().parse_args()
    _apply_overrides(args)
    if MODE == "train":
        train_mode()
    elif MODE == "predict":
        predict_mode()
    else:
        raise ValueError(f"Unsupported MODE='{MODE}'. Use 'train' or 'predict'.")


if __name__ == "__main__":
    main()
