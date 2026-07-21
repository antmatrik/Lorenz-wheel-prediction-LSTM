#!/usr/bin/env python3
"""Plot one input CSV and two output CSVs together (velocity, sin, cos vs epoch).

Example:
    python src/visualize.py data/test.csv outputs/pred_a.csv outputs/pred_b.csv
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def has_header(csv_path: Path) -> bool:
    with csv_path.open("r", newline="") as f:
        first_row = next(csv.reader(f), None)
    if not first_row:
        raise ValueError(f"{csv_path} is empty.")
    for cell in first_row:
        try:
            float(cell.strip())
        except ValueError:
            return True
    return False


def load_series(csv_path: Path) -> np.ndarray:
    header = has_header(csv_path)
    data = np.genfromtxt(
        csv_path,
        delimiter=",",
        dtype=float,
        skip_header=1 if header else 0,
    )
    if data.size == 0:
        raise ValueError(f"{csv_path} has no numeric rows.")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != 4:
        raise ValueError(
            f"{csv_path} must have 4 columns [velocity, sin, cos, epoch], got {data.shape[1]}."
        )
    if not np.isfinite(data).all():
        raise ValueError(f"{csv_path} contains NaN or Inf values.")
    return data


def plot_all(input_data: np.ndarray, out1_data: np.ndarray, out2_data: np.ndarray) -> None:
    names = ["velocity", "sin", "cos"]
    datasets = [
        ("input", input_data, "black"),
        ("output_1", out1_data, "tab:blue"),
        ("output_2", out2_data, "tab:orange"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for col_idx, ax in enumerate(axes):
        for label, data, color in datasets:
            ax.plot(data[:, 3], data[:, col_idx], label=label, color=color, linewidth=1.2)
        ax.set_ylabel(names[col_idx])
        ax.grid(True, alpha=0.3)
        if col_idx == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("epoch")
    fig.suptitle("Input and Outputs")
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot one input CSV and two output CSVs together.",
    )
    parser.add_argument("input_file", help="Input CSV path.")
    parser.add_argument("output_file_1", help="First output CSV path.")
    parser.add_argument("output_file_2", help="Second output CSV path.")
    args = parser.parse_args()

    input_data = load_series(Path(args.input_file).expanduser().resolve())
    out1_data = load_series(Path(args.output_file_1).expanduser().resolve())
    out2_data = load_series(Path(args.output_file_2).expanduser().resolve())
    plot_all(input_data=input_data, out1_data=out1_data, out2_data=out2_data)


if __name__ == "__main__":
    main()
