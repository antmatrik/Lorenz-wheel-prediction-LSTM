#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

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


def load_numeric_csv(csv_path: Path) -> np.ndarray:
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
    if not np.isfinite(data).all():
        raise ValueError(f"{csv_path} contains NaN/Inf values.")
    return data


def list_csv_files(directory: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for p in directory.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() != ".csv":
            continue
        if p.name.startswith(".") or p.name.startswith("._"):
            continue
        files[p.name] = p
    return files


def parse_model_dirs(values: list[str]) -> dict[str, Path]:
    model_dirs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --pred-dir value '{value}'. Use format MODEL_NAME=/path/to/predictions."
            )
        model, path_str = value.split("=", 1)
        model = model.strip()
        path = Path(path_str).expanduser().resolve()
        if not model:
            raise ValueError("Model name in --pred-dir cannot be empty.")
        if model in model_dirs:
            raise ValueError(f"Duplicate model name '{model}' in --pred-dir.")
        if not path.is_dir():
            raise ValueError(f"Prediction directory for model '{model}' does not exist: {path}")
        model_dirs[model] = path
    return model_dirs


def build_result_rows(
    actual_files: dict[str, Path],
    model_dirs: dict[str, Path],
) -> tuple[list[str], list[dict[str, float | int | str]]]:
    rows: list[dict[str, float | int | str]] = []
    max_cols = 0

    for model_name, model_dir in model_dirs.items():
        pred_files = list_csv_files(model_dir)
        missing = sorted(set(actual_files) - set(pred_files))
        if missing:
            preview = ", ".join(missing[:5])
            extra = "..." if len(missing) > 5 else ""
            raise ValueError(
                f"Model '{model_name}' is missing {len(missing)} files that exist in actual dir: {preview}{extra}"
            )

        for file_name, actual_path in sorted(actual_files.items()):
            pred_path = pred_files[file_name]
            actual = load_numeric_csv(actual_path)
            pred = load_numeric_csv(pred_path)

            if actual.shape != pred.shape:
                raise ValueError(
                    f"Shape mismatch in file '{file_name}' for model '{model_name}': "
                    f"actual {actual.shape} vs predicted {pred.shape}"
                )

            sq_err = np.square(pred - actual)
            col_mse = sq_err.mean(axis=0)
            max_cols = max(max_cols, actual.shape[1])

            row: dict[str, float | int | str] = {
                "model": model_name,
                "file": file_name,
                "rows": int(actual.shape[0]),
                "cols": int(actual.shape[1]),
                "mse_overall": float(sq_err.mean()),
                "sse_overall": float(sq_err.sum()),
            }
            for idx, value in enumerate(col_mse, start=1):
                row[f"mse_col_{idx}"] = float(value)
            rows.append(row)

    headers = ["model", "file", "rows", "cols", "mse_overall", "sse_overall"]
    headers.extend([f"mse_col_{i}" for i in range(1, max_cols + 1)])
    return headers, rows


def write_results_csv(
    output_path: Path,
    headers: list[str],
    rows: list[dict[str, float | int | str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare prediction CSV files against actual CSV files and write squared-error metrics."
    )
    parser.add_argument(
        "--actual-dir",
        required=True,
        help="Directory with actual CSV files (example: 50 files).",
    )
    parser.add_argument(
        "--pred-dir",
        required=True,
        action="append",
        help=(
            "Prediction model directory in format MODEL_NAME=/path/to/csvs. "
            "Pass this flag once per model (for example 3 times for 3 models)."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path for squared-error results.",
    )
    parser.add_argument(
        "--expected-files",
        type=int,
        default=50,
        help="Expected number of CSV files in actual-dir (set to 0 to disable check).",
    )
    args = parser.parse_args()

    actual_dir = Path(args.actual_dir).expanduser().resolve()
    if not actual_dir.is_dir():
        raise ValueError(f"Actual directory does not exist: {actual_dir}")

    actual_files = list_csv_files(actual_dir)
    if not actual_files:
        raise ValueError(f"No CSV files found in actual directory: {actual_dir}")

    if args.expected_files > 0 and len(actual_files) != args.expected_files:
        raise ValueError(
            f"Expected {args.expected_files} actual files in {actual_dir}, found {len(actual_files)}."
        )

    model_dirs = parse_model_dirs(args.pred_dir)
    headers, rows = build_result_rows(actual_files=actual_files, model_dirs=model_dirs)
    write_results_csv(Path(args.output).expanduser().resolve(), headers, rows)
    print(f"Wrote {len(rows)} result rows to {args.output}")


if __name__ == "__main__":
    main()
