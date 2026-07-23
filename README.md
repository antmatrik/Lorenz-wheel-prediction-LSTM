# Lorenz Water Wheel Forecasting

Time-series forecasting for **Lorenz water-wheel** data using an **LSTM**, an
**Echo State Network (ESN)**, and a **Temporal Convolutional Network (TCN)**. The
repository contains a multi-file LSTM training pipeline, a PyTorch TCN training pipeline,
standalone single-file forecasters, and a few utilities for comparing and visualizing
results.

## Change log

| Version | Date | Changed by | Type | Summary | Risk |
|---|---|---|---|---|---|
| 1.1 | 2026-07-23 22:39 | "Antons Matrosovs" - Claude | Content | Add PyTorch TCN forecaster (`src/tcn/tcn_forecast.py`), wire it into `src/common/evaluate.py` (`--model tcn`, `--physics`), add a pytest suite, and add Colab train/compare cells | Medium |
| 1.2 | 2026-07-24 01:07 | "Antons Matrosovs" - Claude | Structure | Reorganize `src/` into `lstm/ tcn/ esn/ common/` packages; split `requirements/` per model; commit canonical weights under `weights/`; split the Colab notebook into `train_lstm`, `train_tcn`, and `evaluate` | Medium |
| 1.3 | 2026-07-24 01:58 | "Antons Matrosovs" - Claude | Content | Add per-file actual-vs-predicted plots to `src/common/evaluate.py` (`--plots`: one stacked PNG per horizon 100/300/600/1800, up to 50 files, model name in title); add matplotlib to `requirements/eval.txt`; render + download cells in the evaluate Colab notebook; add render tests | Medium |

## Project layout

```
.
├── src/                            # Python scripts, grouped by model (each runnable standalone)
│   ├── lstm/
│   │   ├── lstm_train_predict.py   # main LSTM pipeline: train on many files, then forecast
│   │   └── lstm_forecast.py        # single-file LSTM forecaster (CLI)
│   ├── tcn/
│   │   └── tcn_forecast.py         # PyTorch TCN pipeline: train on many files, then forecast
│   ├── esn/
│   │   └── esn_forecast.py         # single-file ESN forecaster (CLI)
│   └── common/
│       ├── evaluate.py             # score a trained LSTM/TCN on data/test-dataset (compare runs)
│       ├── compare_lstm_esn.py     # train LSTM + ESN on one file and plot vs actual
│       ├── compare_files.py        # squared-error metrics: predictions vs actuals
│       └── visualize.py            # plot one input + two output CSVs
├── tests/                          # pytest suite (TCN units + evaluator integration)
├── data/
│   ├── train/                      # training CSVs for the training pipelines (25 files)
│   ├── test-dataset/               # 50 <NN>_in / <NN>_out pairs used by evaluate.py
│   ├── test.csv                    # default input for predict mode
│   └── samples/                    # assorted sample CSVs
├── notebooks/
│   ├── train_lstm_colab.ipynb      # Colab GPU: train the LSTM, save weights into weights/
│   ├── train_tcn_colab.ipynb       # Colab GPU: train the TCN, save checkpoint into weights/
│   └── evaluate_colab.ipynb        # Colab: score committed weights/ and compare LSTM vs TCN
├── weights/                        # committed model artifacts the eval notebook scores
├── requirements/                   # per-model dependency subsets (lstm / tcn / eval)
├── outputs/                        # generated weights/stats/predictions (git-ignored)
├── requirements.txt                # union of requirements/ (local dev + tests)
└── README.md
```

Run all commands from the **repository root**. Scripts live in per-model packages under
`src/` but each is still launched directly, e.g. `python src/lstm/lstm_train_predict.py`.

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
pip install -r requirements.txt          # everything (both frameworks) — for local dev + tests
```

`requirements.txt` is the union of the per-model subsets under `requirements/`; install just
one when you only need a single model:

```bash
pip install -r requirements/lstm.txt     # LSTM training  (numpy, tensorflow, matplotlib)
pip install -r requirements/tcn.txt      # TCN training   (numpy, torch, matplotlib)
pip install -r requirements/eval.txt     # evaluation     (numpy, tensorflow, torch)
```

## Main pipeline — `src/lstm/lstm_train_predict.py`

Trains one LSTM across all files in `data/train/`, saves weights and
normalization stats to `outputs/`, then (in predict mode) rolls the model
forward to generate future points. The network takes all three channels
(`angular_velocity, sin, cos`) as input but **predicts only the next angular
velocity**; during the forecast, `sin`/`cos` are reconstructed by integrating
that velocity into the wheel angle. The constants at the top of the file are
defaults; most can be overridden on the command line (`--help` lists them all).

```bash
# Full training run (writes outputs/lstm_weights.weights.h5 + outputs/lstm_stats.npz)
python src/lstm/lstm_train_predict.py --mode train

# Quick sanity check: 1 epoch on a single file
python src/lstm/lstm_train_predict.py --mode train --sanity

# Forecast from data/test.csv using the saved weights (writes outputs/lstm_predictions.csv)
python src/lstm/lstm_train_predict.py --mode predict
```

Common overrides: `--epochs`, `--batch-size`, `--sequence-length`, `--units`,
`--learning-rate`, `--max-train-files N` (train on the first N files only),
`--weights` / `--stats` (artifact paths). Other settings — `WINDOW_STRIDE`
(training-window subsampling), `PREDICT_INPUT_POINTS`, `PREDICT_OUTPUT_POINTS` —
remain constants at the top of the file.

### Colab notebooks

Three notebooks under `notebooks/`, each cloning the repo and setting up only what it needs
(set the runtime to GPU first for the training ones):

- **`train_lstm_colab.ipynb`** — installs `requirements/lstm.txt`, trains the LSTM, and saves
  the weights into `weights/` for you to download and commit.
- **`train_tcn_colab.ipynb`** — installs `requirements/tcn.txt`, trains the TCN, and saves the
  checkpoint into `weights/` for you to download and commit.
- **`evaluate_colab.ipynb`** — a **test run** with no training: installs `requirements/eval.txt`
  and scores the committed `weights/` (LSTM and TCN) on `data/test-dataset`, printing the
  directly-comparable metrics. See [committed weights](#committed-weights) below.

## TCN forecaster — `src/tcn/tcn_forecast.py`

A PyTorch **Temporal Convolutional Network** (dilated causal-convolution residual blocks
+ a linear head) — a second full training pipeline alongside the LSTM. Unlike the LSTM,
which predicts only the next angular velocity and reconstructs `sin`/`cos` by integrating
the wheel angle, the TCN predicts the full next state `[angular_velocity, sin, cos]`
directly and feeds it back autoregressively. It reads the same headerless 4-column CSVs
and writes a single `outputs/tcn_checkpoint.pt` bundling the weights, the normalization
scaler, the input window, and the architecture (so `evaluate.py --model tcn` needs no
extra flags).

```bash
# Full training run on data/train/ (writes outputs/tcn_checkpoint.pt)
python src/tcn/tcn_forecast.py --mode train

# Quick sanity check: 1 epoch on a single file
python src/tcn/tcn_forecast.py --mode train --sanity

# Forecast from data/test.csv using the checkpoint (writes outputs/tcn_predictions.csv)
python src/tcn/tcn_forecast.py --mode predict
```

Common overrides: `--window` (input length), `--channels` (e.g. `128,128,128,128`),
`--kernel-size`, `--horizon` (autoregressive training horizon), `--epochs`,
`--batch-size`, `--learning-rate`, `--max-train-files N`, `--checkpoint` (artifact path).
Requires PyTorch (`torch`, in `requirements.txt`); it uses a GPU automatically when one is
available and falls back to CPU otherwise. Adapted from
[Tartas37/ABFS-2026-Lorenzs-wheel-ai-model](https://github.com/Tartas37/ABFS-2026-Lorenzs-wheel-ai-model).

## Evaluating / comparing training runs — `src/common/evaluate.py`

After training, score a trained model against `data/test-dataset/` (50 `<NN>_in.csv` /
`<NN>_out.csv` pairs) to get comparable numbers across training attempts and across
models. Pick the model with `--model` (default `lstm`); every model flows through the
**same** metric, aggregation, and horizon-sweep code, so the numbers are directly
comparable:

```bash
python src/common/evaluate.py                             # LSTM: full run, all 50 pairs (batched)
python src/common/evaluate.py --model tcn                 # TCN instead (uses outputs/tcn_checkpoint.pt)
python src/common/evaluate.py --model tcn --physics       # TCN via the LSTM-style physics rollout
python src/common/evaluate.py --limit 5 --max-steps 300   # quick smoke eval
python src/common/evaluate.py --input-rows 1800           # seed from only the last minute of input
python src/common/evaluate.py --horizons 25,50,100,200,400,800,1800   # skill-vs-horizon sweep + baselines
python src/common/evaluate.py --plots                     # + per-file actual-vs-predicted PNGs (see below)
```

Point `--weights`/`--stats` (LSTM) or `--checkpoint` (TCN) at the committed artifacts to
score them straight from a clone, e.g.
`python src/common/evaluate.py --model lstm --weights weights/lstm.weights.h5 --stats weights/lstm_stats.npz`.

The LSTM and the TCN differ in more than architecture: the LSTM predicts only `ω` and
rebuilds `sin`/`cos` by integrating the wheel angle, while the TCN predicts all three
channels directly. By default each model is scored **as designed** (TCN = native
3-channel rollout). Add `--physics` to run the TCN through the LSTM's physics rollout
instead — same `ω`-only rollout for both — to isolate the architecture from the rollout
strategy. Either way the reported metrics are computed identically.

All files are advanced **in one batch**, so a full run costs about one model call
per forecast step (~1800 total), not one per file-step (50×1800); `--max-steps`
reduces it further.

### Skill vs horizon (`--horizons`)

Because the wheel is chaotic, pointwise skill decays as the forecast reaches
further ahead — a single full-horizon number hides this. `--horizons` runs **one**
rollout to the largest horizon and reports metrics at each cutoff, alongside two
naive baselines (`base0` = predict ω=0; `persist` = hold the last observed ω) so
each number has a "beat this" anchor:

```
 steps   ~sec |    corr   signed  |w|RMSE   mirror |  base0|w| persist|w|
    25    0.8 |   0.90     ...      1.4      ...    |    ~7.8     ...
   100    3.3 |   0.47     7.16     3.93     6.70   |    ...      ...
   300    9.9 |   0.17    10.04     5.58     9.68   |    ...      ...
  1800   60.1 |   0.03    10.99     6.39    10.79   |    ...      ...
```

Read it as: the model is trustworthy while `|w|RMSE` stays well below the baseline
columns and `corr` stays high. Signed error is inflated by chaotic direction flips
(bifurcations) that no model can predict, so compare **`|w|RMSE` (magnitude)**
across training runs.

For each pair it seeds the model with the last `--input-rows` rows of the input,
forecasts as many steps as the output file has, and compares predicted vs actual
**angular velocity**. It reports:

- **signed** RMSE/MAE — direction matters;
- **`|ω|`** RMSE/MAE — magnitude only, sign-invariant;
- **mirror** RMSE — the better of the error against `+actual` or `−actual`, i.e.
  it forgives a single global direction flip (the wheel can reverse at
  bifurcation points that no model can reliably predict);
- signed correlation.

The **primary comparison number is the mean `|ω|` RMSE**; per-file metrics are
written to `outputs/eval_results.csv`.

### Per-file plots (`--plots`)

`--plots` additionally renders **one tall PNG per horizon** — up to 50 test files
stacked as rows, each a full-width panel of **actual ω (blue) vs predicted ω (red)**,
with the file name and its `|ω|`RMSE / `corr` labelled. The model name is in the
figure title, so LSTM and TCN plots are easy to tell apart.

```bash
python src/common/evaluate.py --plots                      # horizons 100,300,600,1800 (default)
python src/common/evaluate.py --model tcn --plots --plot-horizons 100,300
```

The panels are **sliced from the rollout that already runs** (no extra model calls);
`--plot-horizons` picks the step cutoffs (`1800` ≈ the full ~60 s), `--plot-dir` the
output folder (default `outputs/`). Files are named
`eval_plots_<model>_h<steps>.png`. Requires `matplotlib` (in `requirements/eval.txt`).
The Colab `evaluate` notebook runs this and downloads the PNGs for you.

Note: the rollout conditions only on the last `SEQUENCE_LENGTH` rows, so
`--input-rows` does not change the forecast as long as it stays ≥ `SEQUENCE_LENGTH`
(≈8.5 s at this data's step) — it mainly controls the `dt` estimate.

## Committed weights

`weights/` holds the canonical, version-controlled model artifacts so a fresh clone can be
scored without retraining — this is what `notebooks/evaluate_colab.ipynb` uses. It is the
only place git tracks weight files (a `.gitignore` exception; stray `*.h5`/`*.npz`/`*.pt`
elsewhere stay ignored).

| File | Model |
|---|---|
| `weights/lstm.weights.h5` | LSTM weights (Keras) |
| `weights/lstm_stats.npz` | LSTM normalization stats |
| `weights/tcn_checkpoint.pt` | TCN checkpoint (added once you train + commit a TCN) |

To refresh: train on Colab, download the produced artifacts into `weights/` under these
names, then commit and push. The LSTM file must keep the `.weights.h5` suffix — Keras
selects its loader by that suffix. See [weights/README.md](weights/README.md).

## Standalone forecasters

Both read a single CSV and forecast future points via autoregressive rollout.

Echo State Network — random reservoir + ridge-regression readout:

```bash
python src/esn/esn_forecast.py --input data/samples/water_wheel.csv --steps 300 --output outputs/forecast_esn.csv
```

Useful options: `--reservoir`, `--spectral-radius`, `--leak-rate`, `--sparsity`, `--ridge`, `--washout`.

Single-file LSTM:

```bash
python src/lstm/lstm_forecast.py --input data/samples/water_wheel.csv --steps 300 --output outputs/forecast_lstm.csv
```

Useful options: `--seq-len`, `--units`, `--epochs`, `--batch-size`, `--learning-rate`.

Each writes a CSV of predicted future points with four columns: `angular_velocity,sin,cos,epoch`.

## Comparing models on one file — `src/common/compare_lstm_esn.py`

Trains both an LSTM and an ESN on the same series and overlays their forecasts
against the actual held-out tail:

```bash
python src/common/compare_lstm_esn.py --input data/samples/water_wheel.csv
```

## File comparator (actual vs predictions) — `src/common/compare_files.py`

Compares many prediction files against matching actual files and computes
squared-error metrics. Built for cases like N actual files and one prediction
folder per model.

Each output row contains: `model`, `file`, `rows`, `cols`, `mse_overall`,
`sse_overall`, and `mse_col_1 ... mse_col_N` (per-column MSE).

```bash
python src/common/compare_files.py \
  --actual-dir /path/to/actual_csvs \
  --pred-dir model_a=/path/to/model_a_predictions \
  --pred-dir model_b=/path/to/model_b_predictions \
  --output outputs/results.csv
```

Notes:
- Files are matched by filename.
- By default it expects exactly 50 files in `--actual-dir`; pass `--expected-files 0` to disable that check.

## Visualizer — `src/common/visualize.py`

Plots one input file and two output files together:

```bash
python src/common/visualize.py /path/to/input.csv /path/to/output_1.csv /path/to/output_2.csv
```

Expected CSV columns (with or without header): `velocity,sin,cos,epoch`.
