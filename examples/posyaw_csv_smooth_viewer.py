#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive smoothing for a CSV heatmap.")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--smooth-k", type=int, default=3)
    parser.add_argument("--smooth-alpha", type=float, default=0.692)
    parser.add_argument("--export", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def box_smooth(grid: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return grid.copy()
    if k % 2 == 0:
        k += 1
    pad = k // 2
    valid = np.isfinite(grid)
    data = np.where(valid, grid, 0.0)
    data_p = np.pad(data, pad, mode="edge")
    mask_p = np.pad(valid.astype(float), pad, mode="edge")

    # Integral images with a 1-cell zero border so window sums keep original size.
    s = np.pad(data_p, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    m = np.pad(mask_p, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)

    def window_sum(S: np.ndarray) -> np.ndarray:
        return S[k:, k:] - S[:-k, k:] - S[k:, :-k] + S[:-k, :-k]

    sum_w = window_sum(s)
    cnt_w = window_sum(m)
    out = sum_w / np.maximum(cnt_w, 1.0)
    out[cnt_w == 0] = np.nan
    return out


def apply_smoothing(grid: np.ndarray, k: int, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if k <= 1 or alpha <= 0.0:
        return grid.copy()
    sm = box_smooth(grid, k)
    out = grid.copy()
    base_ok = np.isfinite(grid)
    sm_ok = np.isfinite(sm)
    both = base_ok & sm_ok
    out[both] = (1.0 - alpha) * grid[both] + alpha * sm[both]
    only_sm = (~base_ok) & sm_ok
    out[only_sm] = sm[only_sm]
    return out


def read_grid_csv(path: Path) -> tuple[np.ndarray, list[str], list[float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    data_lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not data_lines:
        raise ValueError("CSV has no data rows.")
    header = data_lines[0].split(",")
    radii = [float(x) for x in header[1:]]
    y_labels: list[str] = []
    rows: list[list[float]] = []
    for ln in data_lines[1:]:
        parts = ln.split(",")
        if not parts:
            continue
        y_labels.append(parts[0])
        row_vals: list[float] = []
        for cell in parts[1:]:
            if cell.strip() == "":
                row_vals.append(float("nan"))
            else:
                row_vals.append(float(cell))
        rows.append(row_vals)
    if not rows:
        raise ValueError("CSV has no grid values.")
    grid = np.asarray(rows, dtype=float)
    return grid, y_labels, radii


def heatmap(ax, data, x_labels, y_labels, title):
    im = ax.imshow(data, origin="lower", aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels([f"{x:.2f}" for x in x_labels], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("radius (m)")
    ax.set_ylabel("vector_glide_weight")
    return im


def save_grid_csv(path: Path, radii: list[float], y_labels: list[str], grid: np.ndarray, params: dict[str, float]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for k, v in params.items():
            f.write(f"# {k}: {v}\n")
        header = ["weight/radius"] + [f"{r:.3f}" for r in radii]
        f.write(",".join(header) + "\n")
        for wi, label in enumerate(y_labels):
            row = [str(label)] + [f"{grid[wi, ri]:.6f}" if np.isfinite(grid[wi, ri]) else "" for ri in range(len(radii))]
            f.write(",".join(row) + "\n")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    grid, y_labels, radii = read_grid_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    smooth = apply_smoothing(grid, args.smooth_k, args.smooth_alpha)
    im = heatmap(ax, smooth, radii, y_labels, "Smoothed Heatmap")
    fig.colorbar(im, ax=ax)

    ax_sm = fig.add_axes([0.2, 0.05, 0.6, 0.04])
    ax_al = fig.add_axes([0.2, 0.01, 0.6, 0.04])
    s_sm = Slider(ax_sm, "smooth_k", 1, 9, valinit=args.smooth_k, valstep=1)
    s_al = Slider(ax_al, "alpha", 0.0, 1.0, valinit=args.smooth_alpha)

    def update(_):
        sm_k = int(s_sm.val)
        alpha = float(s_al.val)
        data = apply_smoothing(grid, sm_k, alpha)
        im.set_data(data)
        print(f"[smooth] smooth_k={sm_k} alpha={alpha:.3f}", flush=True)
        fig.canvas.draw_idle()

    s_sm.on_changed(update)
    s_al.on_changed(update)
    update(None)

    if args.export:
        params = {"smooth_k": float(args.smooth_k), "smooth_alpha": float(args.smooth_alpha)}
        save_grid_csv(Path(args.export), radii, y_labels, smooth, params)
        print(f"saved: {args.export}")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
