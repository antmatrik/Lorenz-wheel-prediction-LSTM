# Lorenz Water Wheel Forecasting

This folder contains two forecasting scripts for Lorenz water wheel time-series data:

1. `esn_lorenz_waterwheel.py` (Echo State Network)
2. `lstm_lorenz_waterwheel.py` (LSTM)

Both scripts read a CSV with angular velocity, `sin(theta)`, `cos(theta)`, and epoch time, then predict future points.

## Input CSV format

Your CSV must be **headerless** and use this exact column order:

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

## Echo State Network (ESN)

Run:

```bash
python3 esn_lorenz_waterwheel.py --input your_data.csv --steps 300 --output forecast_esn.csv
```

How it works:

- Builds a random reservoir (fixed recurrent core).
- Trains only the output weights (ridge regression).
- Rolls forward autoregressively to generate next points.

Useful options:

- `--reservoir`, `--spectral-radius`, `--leak-rate`, `--sparsity`, `--ridge`, `--washout`

## LSTM

Run:

```bash
python3 lstm_lorenz_waterwheel.py --input your_data.csv --steps 300 --output forecast_lstm.csv
```

How it works:

- Creates sliding windows of `[angular_velocity, sin, cos]`.
- Trains an LSTM to predict the next `[angular_velocity, sin, cos]`.
- Uses autoregressive rollout for future steps.

Useful options:

- `--seq-len`, `--units`, `--epochs`, `--batch-size`, `--learning-rate`

## Output

Both scripts write a CSV of predicted future points with four columns:

`angular_velocity,sin,cos,epoch`

Each row is one predicted next point in time.

## File comparator (actual vs predictions)

Use `file_comporator.py` to compare many prediction files against matching actual files and compute squared-error metrics.

It is built for cases like:
- 50 actual CSV files
- 50 predicted CSV files per model
- multiple models (for example 3 model folders)

Each row in the output CSV contains:
- `model`, `file`, `rows`, `cols`
- `mse_overall`, `sse_overall`
- `mse_col_1 ... mse_col_N` (per-column mean squared error)

Run:

```bash
python3 file_comporator.py \
  --actual-dir /path/to/actual_csvs \
  --pred-dir model_a=/path/to/model_a_predictions \
  --pred-dir model_b=/path/to/model_b_predictions \
  --pred-dir model_c=/path/to/model_c_predictions \
  --output /path/to/results.csv
```

Notes:
- Files are matched by filename.
- By default it expects exactly 50 files in `--actual-dir`.
- To disable that check, pass `--expected-files 0`.

## Visualizer (input + 2 outputs)

Use `visualizer.py` to plot one input file and two output files together.

Run:

```bash
python3 visualizer.py /path/to/input.csv /path/to/output_1.csv /path/to/output_2.csv
```

Expected CSV columns (with or without header): `velocity,sin,cos,epoch`
