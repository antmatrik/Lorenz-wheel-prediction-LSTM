# Committed model weights

These are the canonical, version-controlled model artifacts. `notebooks/evaluate_colab.ipynb`
clones the repo and scores whatever is present here — no training required. They are the
**only** weight files git tracks (a `.gitignore` exception; stray `*.h5`/`*.npz`/`*.pt`
elsewhere stay ignored).

| File | Model | Produced by |
|---|---|---|
| `lstm.weights.h5` | LSTM weights (Keras) | `src/lstm/lstm_train_predict.py --mode train` → `outputs/lstm_weights.weights.h5` |
| `lstm_stats.npz` | LSTM normalization stats (`mean`/`std`, shape `(3,)`) | same run → `outputs/lstm_stats.npz` |
| `tcn_checkpoint.pt` | TCN weights + scaler + window + arch (PyTorch) | `src/tcn/tcn_forecast.py --mode train` → `outputs/tcn_checkpoint.pt` |

> The LSTM file must keep the `.weights.h5` suffix — Keras selects its modern loader by
> that suffix, and any other name (e.g. `*-2.h5`) falls back to the legacy loader and
> fails with "0 saved layers".

## Refreshing the weights

After training on Colab (see `notebooks/train_lstm_colab.ipynb` /
`notebooks/train_tcn_colab.ipynb`), download the produced artifacts, drop them here under
the canonical names above, then commit and push. Score them with:

```bash
python src/common/evaluate.py --model lstm --weights weights/lstm.weights.h5 --stats weights/lstm_stats.npz
python src/common/evaluate.py --model tcn  --checkpoint weights/tcn_checkpoint.pt
```

`tcn_checkpoint.pt` is absent until you train and commit a TCN model; the evaluation
notebook skips whichever model has no weights here.
