#!/usr/bin/env python3
"""Process raw MPC error grid data and generate heatmaps."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_grid(values_radius, values_weight, rows, value_key):
    r_to_i = {r: i for i, r in enumerate(values_radius)}
    w_to_j = {w: j for j, w in enumerate(values_weight)}
    grid = np.full((len(values_radius), len(values_weight)), np.nan, dtype=float)
    for _, row in rows.iterrows():
        r = float(row["radius"])
        w = float(row["weight"])
        if r not in r_to_i or w not in w_to_j:
            continue
        grid[r_to_i[r], w_to_j[w]] = float(row[value_key])
    return grid


def save_heatmap(data: np.ndarray, title: str, fname: Path, weight_vals, radius_vals, cmap: str = "viridis") -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=[weight_vals[0], weight_vals[-1], radius_vals[0], radius_vals[-1]],
        cmap=cmap,
    )
    plt.colorbar(label=title)
    plt.xlabel("vector_glide_weight")
    plt.ylabel("ICR radius (m)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


def _resolve_fill_value(series: pd.Series, fill_missing: str) -> float:
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return 0.0
    if fill_missing == "min":
        return float(valid.min())
    if fill_missing == "median":
        return float(valid.median())
    if fill_missing == "zero":
        return 0.0
    if fill_missing == "max":
        return float(valid.max())
    try:
        return float(fill_missing)
    except ValueError:
        return float(valid.max())


def _neighbor_stats(grid: np.ndarray, i: int, j: int) -> tuple:
    i0 = max(i - 1, 0)
    i1 = min(i + 2, grid.shape[0])
    j0 = max(j - 1, 0)
    j1 = min(j + 2, grid.shape[1])
    window = grid[i0:i1, j0:j1]
    mask = np.ones_like(window, dtype=bool)
    mask[i - i0, j - j0] = False
    vals = window[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, 0
    return float(np.mean(vals)), float(np.std(vals)), int(vals.size)


def _neighbor_mad(grid: np.ndarray, i: int, j: int) -> tuple:
    i0 = max(i - 1, 0)
    i1 = min(i + 2, grid.shape[0])
    j0 = max(j - 1, 0)
    j1 = min(j + 2, grid.shape[1])
    window = grid[i0:i1, j0:j1]
    mask = np.ones_like(window, dtype=bool)
    mask[i - i0, j - j0] = False
    vals = window[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, 0
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return med, mad, int(vals.size)


def _filter_outliers(
    grid: np.ndarray,
    k: float,
    abs_thresh: float | None,
    min_neighbors: int,
    method: str,
) -> tuple:
    outlier_mask = np.zeros_like(grid, dtype=bool)
    filled = grid.copy()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid[i, j]
            if not np.isfinite(val):
                continue
            if method == "mad":
                center, spread, count = _neighbor_mad(grid, i, j)
                scale = 1.4826 * spread
            else:
                center, spread, count = _neighbor_stats(grid, i, j)
                scale = spread
            if count < min_neighbors or not np.isfinite(center):
                continue
            diff = abs(val - center)
            is_outlier = diff > (k * scale) if scale > 0 else False
            if abs_thresh is not None:
                is_outlier = is_outlier or (diff > abs_thresh)
            if is_outlier:
                outlier_mask[i, j] = True
                filled[i, j] = center
    return filled, outlier_mask


def process_group(
    df: pd.DataFrame,
    out_dir: Path,
    suffix: str,
    fill_missing: str,
    outlier_k: float,
    outlier_abs: float | None,
    outlier_min_neighbors: int,
    outlier_method: str,
    write_filled_raw: str | None,
) -> None:
    radius_vals = sorted(df["radius"].unique().tolist())
    weight_vals = sorted(df["weight"].unique().tolist())
    if not radius_vals or not weight_vals:
        return

    df = df.copy()
    df["pos_err_norm"] = np.sqrt(df["pos_err_x"] ** 2 + df["pos_err_y"] ** 2)
    df["yaw_err_abs"] = df["yaw_err"].abs()
    if "joint_dev_norm" not in df.columns:
        raise SystemExit("joint_dev_norm missing in raw data")

    df["joint_dev_norm"] = df["joint_dev_norm"].astype(float)

    combined = np.sqrt(
        df["pos_err_norm"] ** 2 + df["yaw_err_abs"] ** 2 + df["joint_dev_norm"] ** 2
    )
    df["combined_dev"] = combined

    valid = df["combined_dev"].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        raise SystemExit("No valid rows to compute baseline")

    idx_best = valid.idxmin()
    best_row = df.loc[idx_best]
    base_pos = float(best_row["pos_err_norm"])
    base_yaw = float(best_row["yaw_err_abs"])
    base_joint = float(best_row["joint_dev_norm"])

    df["pos_dev_from_best"] = df["pos_err_norm"] - base_pos
    df["yaw_dev_from_best"] = df["yaw_err_abs"] - base_yaw
    df["joint_dev_from_best"] = df["joint_dev_norm"] - base_joint
    df["combined_dev_from_best"] = np.sqrt(
        df["pos_dev_from_best"] ** 2 + df["yaw_dev_from_best"] ** 2 + df["joint_dev_from_best"] ** 2
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"processed_metrics{suffix}.csv"
    df.to_csv(out_csv, index=False)

    missing_mask = df[["pos_err_norm", "yaw_err_abs", "joint_dev_norm", "combined_dev_from_best"]].isna().any(axis=1)
    if missing_mask.any():
        miss = df.loc[missing_mask, ["radius", "weight"]].copy()
        miss.to_csv(out_dir / f"missing_points{suffix}.csv", index=False)

    pos_grid = make_grid(radius_vals, weight_vals, df, "pos_err_norm")
    yaw_grid = make_grid(radius_vals, weight_vals, df, "yaw_err_abs")
    joint_grid = make_grid(radius_vals, weight_vals, df, "joint_dev_norm")
    dev_grid = make_grid(radius_vals, weight_vals, df, "combined_dev_from_best")

    # Outlier filtering based on yaw grid, then apply to all grids.
    if outlier_k is not None and outlier_k > 0:
        yaw_filled, out_mask = _filter_outliers(
            yaw_grid, outlier_k, outlier_abs, outlier_min_neighbors, outlier_method
        )
        if out_mask.any():
            outliers = []
            for i, r in enumerate(radius_vals):
                for j, w in enumerate(weight_vals):
                    if out_mask[i, j]:
                        outliers.append({"radius": r, "weight": w, "metric": "yaw_err_abs"})
            pd.DataFrame(outliers).to_csv(out_dir / f"outliers{suffix}.csv", index=False)
            yaw_grid = yaw_filled
            # apply same mask to other grids
            for grid_name, grid in (
                ("pos", pos_grid),
                ("yaw", yaw_grid),
                ("joint", joint_grid),
                ("dev", dev_grid),
            ):
                repl = grid.copy()
                for i in range(grid.shape[0]):
                    for j in range(grid.shape[1]):
                        if out_mask[i, j]:
                            mean, _, _ = _neighbor_stats(grid, i, j)
                            if np.isfinite(mean):
                                repl[i, j] = mean
                if grid_name == "pos":
                    pos_grid = repl
                elif grid_name == "yaw":
                    yaw_grid = repl
                elif grid_name == "dev":
                    dev_grid = repl
                else:
                    joint_grid = repl

    fill_pos = _resolve_fill_value(df["pos_err_norm"], fill_missing)
    fill_yaw = _resolve_fill_value(df["yaw_err_abs"], fill_missing)
    fill_joint = _resolve_fill_value(df["joint_dev_norm"], fill_missing)
    fill_dev = _resolve_fill_value(df["combined_dev_from_best"], fill_missing)

    pos_grid = np.nan_to_num(pos_grid, nan=fill_pos, posinf=fill_pos, neginf=fill_pos)
    yaw_grid = np.nan_to_num(yaw_grid, nan=fill_yaw, posinf=fill_yaw, neginf=fill_yaw)
    joint_grid = np.nan_to_num(joint_grid, nan=fill_joint, posinf=fill_joint, neginf=fill_joint)
    dev_grid = np.nan_to_num(dev_grid, nan=fill_dev, posinf=fill_dev, neginf=fill_dev)

    save_heatmap(pos_grid, "Position error norm", out_dir / f"pos_err_norm{suffix}.png", weight_vals, radius_vals)
    save_heatmap(yaw_grid, "Yaw error (abs)", out_dir / f"yaw_err_abs{suffix}.png", weight_vals, radius_vals)
    save_heatmap(joint_grid, "Joint deviation norm", out_dir / f"joint_dev_norm{suffix}.png", weight_vals, radius_vals)
    save_heatmap(dev_grid, "Deviation from best (combined)", out_dir / f"dev_from_best{suffix}.png", weight_vals, radius_vals)

    np.save(out_dir / f"pos_err_norm{suffix}.npy", pos_grid)
    np.save(out_dir / f"yaw_err_abs{suffix}.npy", yaw_grid)
    np.save(out_dir / f"joint_dev_norm{suffix}.npy", joint_grid)
    np.save(out_dir / f"dev_from_best{suffix}.npy", dev_grid)

    if write_filled_raw:
        raw_cols = [
            "pos_err_x",
            "pos_err_y",
            "yaw_err",
            "vel_err_x",
            "vel_err_y",
            "yaw_rate_err",
            "joint_dev_norm",
        ]
        filled = {"radius": [], "weight": []}
        for col in raw_cols:
            if col in df.columns:
                grid = make_grid(radius_vals, weight_vals, df, col)
                fill_val = _resolve_fill_value(df[col], "median")
                grid = np.nan_to_num(grid, nan=fill_val, posinf=fill_val, neginf=fill_val)
                # neighbor fill for isolated NaNs
                grid, _ = _filter_outliers(grid, outlier_k, outlier_abs, outlier_min_neighbors, outlier_method)
                filled[col] = grid
        for i, r in enumerate(radius_vals):
            for j, w in enumerate(weight_vals):
                filled["radius"].append(r)
                filled["weight"].append(w)
        out_rows = []
        for idx in range(len(filled["radius"])):
            row = {"radius": filled["radius"][idx], "weight": filled["weight"][idx]}
            for col in raw_cols:
                if col in filled:
                    r = filled["radius"][idx]
                    w = filled["weight"][idx]
                    i = radius_vals.index(r)
                    j = weight_vals.index(w)
                    row[col] = float(filled[col][i, j])
            out_rows.append(row)
        out_filled = Path(write_filled_raw)
        out_filled.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(out_rows).to_csv(out_filled, index=False)

    print(f"Saved processed outputs to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, default="examples/analysis_outputs/raw_error_grid.csv")
    parser.add_argument("--output-dir", type=str, default="examples/analysis_outputs")
    parser.add_argument(
        "--fill-missing",
        type=str,
        default="max",
        help="Fill value for NaN cells in heatmaps: max|min|median|zero|<number> (default: max)",
    )
    parser.add_argument("--outlier-k", type=float, default=3.0, help="Outlier threshold in stds (default: 3)")
    parser.add_argument("--outlier-abs", type=float, default=None, help="Absolute outlier threshold override")
    parser.add_argument("--outlier-min-neighbors", type=int, default=3, help="Min neighbors to judge outlier")
    parser.add_argument(
        "--outlier-method",
        type=str,
        default="mad",
        choices=["mad", "std"],
        help="Outlier method: mad (robust) or std",
    )
    parser.add_argument(
        "--write-filled-raw",
        type=str,
        default="",
        help="Write a filled raw CSV (for NaN cells) to this path",
    )
    args = parser.parse_args()

    in_path = Path(args.input_csv)
    if not in_path.exists():
        raise SystemExit(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path)
    if df.empty:
        raise SystemExit("Input CSV is empty")

    out_dir = Path(args.output_dir)

    # Merge use_vector_glide_cost=False rows into weight==0 column.
    if "use_vector_glide_cost" in df.columns:
        false_mask = df["use_vector_glide_cost"] == False
        if false_mask.any():
            df.loc[false_mask, "weight"] = 0.0
        df = df.drop(columns=["use_vector_glide_cost"])

    # If duplicates exist after merge, average numeric columns.
    if df.duplicated(subset=["radius", "weight"]).any():
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        df = df.groupby(["radius", "weight"], as_index=False)[numeric_cols].mean()

    process_group(
        df,
        out_dir,
        "",
        args.fill_missing,
        args.outlier_k,
        args.outlier_abs,
        args.outlier_min_neighbors,
        args.outlier_method,
        args.write_filled_raw or None,
    )


if __name__ == "__main__":
    main()
