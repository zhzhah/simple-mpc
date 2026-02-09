#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot vector-glide ablation heatmaps.")
    parser.add_argument("--summary", type=str, default="out/vector_glide_ablation/summary.csv")
    parser.add_argument("--out-dir", type=str, default="out/vector_glide_ablation")
    return parser.parse_args()


def heatmap(ax, data, x_labels, y_labels, title, vmin=None, vmax=None):
    im = ax.imshow(data, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels([f"{x:.2f}" for x in x_labels], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels([f"{y:.1f}" for y in y_labels])
    ax.set_xlabel("radius (m)")
    ax.set_ylabel("vector_glide_weight")
    return im


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)
    # aggregate over yaw0
    g = df.groupby(["radius", "weight"], as_index=False).agg(
        lin_err_mean=("lin_err_mean", "mean"),
        yaw_err_mean=("yaw_err_mean", "mean"),
        completed=("completed", "mean"),
    )

    radii = np.sort(g["radius"].unique())
    weights = np.sort(g["weight"].unique())
    lin_grid = np.full((len(weights), len(radii)), np.nan, dtype=float)
    yaw_grid = np.full((len(weights), len(radii)), np.nan, dtype=float)
    comp_grid = np.full((len(weights), len(radii)), np.nan, dtype=float)

    idx_r = {r: i for i, r in enumerate(radii)}
    idx_w = {w: i for i, w in enumerate(weights)}
    for _, row in g.iterrows():
        wi = idx_w[row["weight"]]
        ri = idx_r[row["radius"]]
        lin_grid[wi, ri] = row["lin_err_mean"]
        yaw_grid[wi, ri] = row["yaw_err_mean"]
        comp_grid[wi, ri] = row["completed"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    im0 = heatmap(axes[0], lin_grid, radii, weights, "Linear Speed Error Mean")
    im1 = heatmap(axes[1], yaw_grid, radii, weights, "Yaw Rate Error Mean")
    im2 = heatmap(axes[2], comp_grid, radii, weights, "Completion Ratio")
    fig.colorbar(im0, ax=axes[0])
    fig.colorbar(im1, ax=axes[1])
    fig.colorbar(im2, ax=axes[2])

    out_path = out_dir / "vector_glide_ablation_heatmaps.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
