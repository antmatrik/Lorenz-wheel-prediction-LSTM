# Lorenz Water Wheel Forecasting

Time-series forecasting for **Lorenz water-wheel** data using an **LSTM** and an
**Echo State Network (ESN)**. The repository contains a multi-file LSTM training
pipeline, standalone single-file forecasters for both model families, and a few
utilities for comparing and visualizing results.

## Project layout

```
.
├── src/                       # all Python scripts (each runnable standalone)
│   ├── lstm_train_predict.py  # main pipeline: train on many files, then forecast
│   ├── lstm_forecast.py       # single-file LSTM forecaster (CLI)
│   ├── esn_forecast.py        # single-file ESN forecaster (CLI)
│   ├── compare_lstm_esn.py    # train LSTM + ESN on one file and plot vs actual
│   ├── compare_files.py       # squared-error metrics: predictions vs actuals
│   └── visualize.py           # plot one input + two output CSVs
├── data/
│   ├── train/                 # training CSVs for lstm_train_predict.py (25 files)
│   ├── test.csv               # default input for predict mode
│   └── samples/               # assorted sample CSVs
├── notebooks/
│   └── train_colab.ipynb      # Colab GPU training + weight transfer
├── outputs/                   # generated weights/stats/predictions (git-ignored)
├── requirements.txt
└── README.md
```

Run all commands from the **repository root**.

## Input CSV format

Every CSV is **headerless** and uses this exact column order:

- `angular_velocity`
- `sin`
- `cos`
- `epoch`

Example:

```csv
1.000030,0.001000,0.999999,1784447801.932
1.133510,0.035586,0.999367,1784447801.965
...
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Main pipeline — `lstm_train_predict.py`

Trains one LSTM across all files in `data/train/`, saves weights and
normalization stats to `outputs/`, then (in predict mode) rolls the model
forward to generate future points. The network takes all three channels
(`angular_velocity, sin, cos`) as input but **predicts only the next angular
velocity**; during the forecast, `sin`/`cos` are reconstructed by integrating
that velocity into the wheel angle. The constants at the top of the file are
defaults; most can be overridden on the command line (`--help` lists them all).

```bash
# Full training run (writes outputs/lstm_weights.weights.h5 + outputs/lstm_stats.npz)
python src/lstm_train_predict.py --mode train

# Quick sanity check: 1 epoch on a single file
python src/lstm_train_predict.py --mode train --sanity

# Forecast from data/test.csv using the saved weights (writes outputs/lstm_predictions.csv)
python src/lstm_train_predict.py --mode predict
```

Common overrides: `--epochs`, `--batch-size`, `--sequence-length`, `--units`,
`--learning-rate`, `--max-train-files N` (train on the first N files only),
`--weights` / `--stats` (artifact paths). Other settings — `WINDOW_STRIDE`
(training-window subsampling), `PREDICT_INPUT_POINTS`, `PREDICT_OUTPUT_POINTS` —
remain constants at the top of the file.

### Training on Colab

`notebooks/train_colab.ipynb` clones the repo, installs dependencies, runs a
sanity or production training on Colab's GPU, and copies the resulting weights to
Google Drive (or downloads them). Set the runtime to GPU first.

## Standalone forecasters

Both read a single CSV and forecast future points via autoregressive rollout.

Echo State Network — random reservoir + ridge-regression readout:

```bash
python src/esn_forecast.py --input data/samples/water_wheel.csv --steps 300 --output outputs/forecast_esn.csv
```

Useful options: `--reservoir`, `--spectral-radius`, `--leak-rate`, `--sparsity`, `--ridge`, `--washout`.

Single-file LSTM:

```bash
python src/lstm_forecast.py --input data/samples/water_wheel.csv --steps 300 --output outputs/forecast_lstm.csv
```

Useful options: `--seq-len`, `--units`, `--epochs`, `--batch-size`, `--learning-rate`.

Each writes a CSV of predicted future points with four columns: `angular_velocity,sin,cos,epoch`.

## Comparing models on one file — `compare_lstm_esn.py`

Trains both an LSTM and an ESN on the same series and overlays their forecasts
against the actual held-out tail:

```bash
python src/compare_lstm_esn.py --input data/samples/water_wheel.csv
```

## File comparator (actual vs predictions) — `compare_files.py`

Compares many prediction files against matching actual files and computes
squared-error metrics. Built for cases like N actual files and one prediction
folder per model.

Each output row contains: `model`, `file`, `rows`, `cols`, `mse_overall`,
`sse_overall`, and `mse_col_1 ... mse_col_N` (per-column MSE).

```bash
python src/compare_files.py \
  --actual-dir /path/to/actual_csvs \
  --pred-dir model_a=/path/to/model_a_predictions \
  --pred-dir model_b=/path/to/model_b_predictions \
  --output outputs/results.csv
```

Notes:
- Files are matched by filename.
- By default it expects exactly 50 files in `--actual-dir`; pass `--expected-files 0` to disable that check.

## Visualizer — `visualize.py`

Plots one input file and two output files together:

```bash
python src/visualize.py /path/to/input.csv /path/to/output_1.csv /path/to/output_2.csv
```

Expected CSV columns (with or without header): `velocity,sin,cos,epoch`.
