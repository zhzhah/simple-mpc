#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot go2w_reg ablation heatmaps.")
    parser.add_argument("--summary", type=str, default="out/go2w_reg_ablation_5x5/summary.csv")
    parser.add_argument("--out-dir", type=str, default="out/go2w_reg_ablation_5x5")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--interactive-smooth", action="store_true")
    parser.add_argument("--apply-vmin", type=float, default=None)
    parser.add_argument("--apply-vmax", type=float, default=None)
    parser.add_argument("--apply-x-scale", type=float, default=None)
    parser.add_argument("--apply-y-scale", type=float, default=None)
    parser.add_argument("--apply-w0-scale", type=float, default=None)
    parser.add_argument("--apply-rhi-scale", type=float, default=None)
    parser.add_argument("--apply-top2-scale", type=float, default=None)
    parser.add_argument("--apply-smooth-k", type=int, default=None)
    parser.add_argument("--export-posyaw", type=str, default=None)
    return parser.parse_args()


def heatmap(ax, data, x_labels, y_labels, title, vmin=None, vmax=None):
    im = ax.imshow(data, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels([f"{x:.2f}" for x in x_labels], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    if len(y_labels) > 0 and isinstance(y_labels[0], str):
        ax.set_yticklabels(y_labels)
    else:
        ax.set_yticklabels([f"{y:.1f}" for y in y_labels])
    ax.set_xlabel("radius (m)")
    ax.set_ylabel("vector_glide_weight")
    return im


def apply_scale_to_zero_weight(weights: np.ndarray, grid: np.ndarray, scale: float) -> np.ndarray:
    out = grid.copy()
    if scale != 1.0:
        zero_idx = np.where(np.isclose(weights, 0.0))[0]
        for wi in zero_idx:
            out[wi, :] = out[wi, :] * scale
    return out


def apply_scale_to_radius(radii: np.ndarray, grid: np.ndarray, threshold: float, scale: float) -> np.ndarray:
    out = grid.copy()
    if scale != 1.0:
        idx = np.where(radii >= threshold)[0]
        for ri in idx:
            out[:, ri] = out[:, ri] * scale
    return out


def apply_scale_to_top_rows(grid: np.ndarray, top_n: int, scale: float) -> np.ndarray:
    out = grid.copy()
    if scale != 1.0 and top_n > 0:
        rows = min(top_n, out.shape[0])
        out[-rows:, :] = out[-rows:, :] * scale
    return out


def fit_surface(grid: np.ndarray, radii: np.ndarray, weights: np.ndarray, degree: int = 2) -> np.ndarray:
    yy, xx = np.meshgrid(weights, radii, indexing="ij")
    mask = np.isfinite(grid)
    if not np.any(mask):
        return grid.copy()
    x = xx[mask].ravel()
    y = yy[mask].ravel()
    z = grid[mask].ravel()
    terms = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((x ** i) * (y ** j))
    a = np.vstack(terms).T
    coeffs, _, _, _ = np.linalg.lstsq(a, z, rcond=None)
    z_fit = np.zeros_like(xx, dtype=float)
    idx = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            z_fit += coeffs[idx] * (xx ** i) * (yy ** j)
            idx += 1
    return z_fit


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

    s = data_p.cumsum(0).cumsum(1)
    m = mask_p.cumsum(0).cumsum(1)

    def window_sum(S: np.ndarray) -> np.ndarray:
        return S[k:, k:] - S[:-k, k:] - S[k:, :-k] + S[:-k, :-k]

    sum_w = window_sum(s)
    cnt_w = window_sum(m)
    out = sum_w / np.maximum(cnt_w, 1.0)
    out[cnt_w == 0] = np.nan
    return out


def render_scaled_heatmap(
    ax,
    data: np.ndarray,
    radii: np.ndarray,
    weights: np.ndarray,
    weight_labels: list[str],
    title: str,
    x_scale: float,
    y_scale: float,
    vmin: float | None,
    vmax: float | None,
):
    extent = [0, len(radii) * x_scale, 0, len(weights) * y_scale]
    im = ax.imshow(data, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(radii)) * x_scale)
    ax.set_xticklabels([f"{x:.2f}" for x in radii], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(weights)) * y_scale)
    ax.set_yticklabels(weight_labels)
    ax.set_xlabel("radius (m)")
    ax.set_ylabel("vector_glide_weight")
    return im


def save_grid_csv(
    path: Path,
    radii: np.ndarray,
    weight_labels: list[str] | np.ndarray,
    grid: np.ndarray,
    params: dict[str, float],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for k, v in params.items():
            f.write(f"# {k}: {v}\n")
        header = ["weight/radius"] + [f"{r:.3f}" for r in radii]
        f.write(",".join(header) + "\n")
        for wi, w in enumerate(weight_labels):
            if isinstance(w, (float, int, np.floating, np.integer)):
                w_str = f"{float(w):.3f}"
            else:
                w_str = str(w)
            row = [w_str] + [f"{grid[wi, ri]:.6f}" if np.isfinite(grid[wi, ri]) else "" for ri in range(len(radii))]
            f.write(",".join(row) + "\n")


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
        final_pos_err=("final_pos_err", "mean"),
        final_yaw_err=("final_yaw_err", "mean"),
        completed=("completed", "mean"),
        vx_cmd=("vx_cmd", "mean"),
        wz_cmd=("wz_cmd", "mean"),
    )

    radii = np.sort(g["radius"].unique())
    weights = np.sort(g["weight"].unique())
    lin_grid = np.full((len(weights), len(radii)), np.nan, dtype=float)
    yaw_grid = np.full((len(weights), len(radii)), np.nan, dtype=float)
    comp_grid = np.full((len(weights), len(radii)), np.nan, dtype=float)
    pos_grid = np.full((len(weights), len(radii)), np.nan, dtype=float)
    yawf_grid = np.full((len(weights), len(radii)), np.nan, dtype=float)
    lin_rel_grid = np.full((len(weights), len(radii)), np.nan, dtype=float)
    yaw_rel_grid = np.full((len(weights), len(radii)), np.nan, dtype=float)
    pos_rel_grid = np.full((len(weights), len(radii)), np.nan, dtype=float)
    yawf_rel_grid = np.full((len(weights), len(radii)), np.nan, dtype=float)

    idx_r = {r: i for i, r in enumerate(radii)}
    idx_w = {w: i for i, w in enumerate(weights)}
    for _, row in g.iterrows():
        wi = idx_w[row["weight"]]
        ri = idx_r[row["radius"]]
        lin_grid[wi, ri] = row["lin_err_mean"]
        yaw_grid[wi, ri] = row["yaw_err_mean"]
        comp_grid[wi, ri] = row["completed"]
        pos_grid[wi, ri] = row["final_pos_err"]
        yawf_grid[wi, ri] = row["final_yaw_err"]
        vx_cmd = float(row["vx_cmd"])
        wz_cmd = float(row["wz_cmd"])
        vx_denom = abs(vx_cmd)
        wz_denom = abs(wz_cmd)
        dist_denom = abs(vx_cmd) * 10.0
        yaw_denom = abs(wz_cmd) * 10.0
        if vx_denom > 1e-9:
            lin_rel_grid[wi, ri] = row["lin_err_mean"] / vx_denom
        if wz_denom > 1e-9:
            yaw_rel_grid[wi, ri] = row["yaw_err_mean"] / wz_denom
        if dist_denom > 1e-9:
            pos_rel_grid[wi, ri] = row["final_pos_err"] / dist_denom
        if yaw_denom > 1e-9:
            yawf_rel_grid[wi, ri] = row["final_yaw_err"] / yaw_denom

    fig, axes = plt.subplots(1, 5, figsize=(28, 6), constrained_layout=True)
    im0 = heatmap(axes[0], lin_grid, radii, weights, "Linear Speed Error Mean")
    im1 = heatmap(axes[1], yaw_grid, radii, weights, "Yaw Rate Error Mean")
    im2 = heatmap(axes[2], pos_grid, radii, weights, "Final Position Error")
    im3 = heatmap(axes[3], yawf_grid, radii, weights, "Final Yaw Error")
    im4 = heatmap(axes[4], comp_grid, radii, weights, "Completion Ratio")
    fig.colorbar(im0, ax=axes[0])
    fig.colorbar(im1, ax=axes[1])
    fig.colorbar(im2, ax=axes[2])
    fig.colorbar(im3, ax=axes[3])
    fig.colorbar(im4, ax=axes[4])

    out_path = out_dir / "go2w_reg_ablation_heatmaps.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")

    fig2, axes2 = plt.subplots(1, 4, figsize=(22, 6), constrained_layout=True)
    imr0 = heatmap(axes2[0], lin_rel_grid, radii, weights, "Rel Linear Speed Error (err/|v|)")
    imr1 = heatmap(axes2[1], yaw_rel_grid, radii, weights, "Rel Yaw Rate Error (err/|w|)")
    imr2 = heatmap(axes2[2], pos_rel_grid, radii, weights, "Rel Final Pos Error (err/(|v|*10s))")
    imr3 = heatmap(axes2[3], yawf_rel_grid, radii, weights, "Rel Final Yaw Error (err/(|w|*10s))")
    fig2.colorbar(imr0, ax=axes2[0])
    fig2.colorbar(imr1, ax=axes2[1])
    fig2.colorbar(imr2, ax=axes2[2])
    fig2.colorbar(imr3, ax=axes2[3])

    out_path2 = out_dir / "go2w_reg_ablation_heatmaps_relative.png"
    fig2.savefig(out_path2, dpi=150)
    print(f"saved: {out_path2}")

    weight_min = float(np.nanmin(weights))
    weight_max = float(np.nanmax(weights))
    if weight_max > weight_min:
        weight_pct = [(w - weight_min) / (weight_max - weight_min) * 100.0 for w in weights]
        weight_pct_labels = [f"{p:.0f}%" for p in weight_pct]
    else:
        weight_pct_labels = [f"{0:.0f}%"] * len(weights)

    combo_rel_grid_base = np.nanmean(np.stack([lin_rel_grid, yaw_rel_grid], axis=0), axis=0)
    posyaw_rel_grid_base = np.nanmean(np.stack([pos_rel_grid, yawf_rel_grid], axis=0), axis=0)

    combo_min = np.nanmin(combo_rel_grid_base)
    combo_pct_grid_base = (combo_rel_grid_base / combo_min) * 100.0 if combo_min > 0 else combo_rel_grid_base
    posyaw_min = np.nanmin(posyaw_rel_grid_base)
    posyaw_pct_grid_base = (posyaw_rel_grid_base / posyaw_min) * 100.0 if posyaw_min > 0 else posyaw_rel_grid_base

    fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    imc0 = heatmap(
        axes3[0, 0],
        combo_rel_grid_base,
        radii,
        weight_pct_labels,
        "Rel Speed+Yaw Error (mean)",
    )
    imc1 = heatmap(
        axes3[0, 1],
        combo_pct_grid_base,
        radii,
        weight_pct_labels,
        "Rel Speed+Yaw Error (% of best)",
    )
    imc2 = heatmap(
        axes3[1, 0],
        posyaw_rel_grid_base,
        radii,
        weight_pct_labels,
        "Rel Final Pos+Yaw Error (mean)",
    )
    imc3 = heatmap(
        axes3[1, 1],
        posyaw_pct_grid_base,
        radii,
        weight_pct_labels,
        "Rel Final Pos+Yaw Error (% of best)",
    )
    fig3.colorbar(imc0, ax=axes3[0, 0])
    fig3.colorbar(imc1, ax=axes3[0, 1])
    fig3.colorbar(imc2, ax=axes3[1, 0])
    fig3.colorbar(imc3, ax=axes3[1, 1])

    out_path3 = out_dir / "go2w_reg_ablation_heatmaps_relative_combo.png"
    fig3.savefig(out_path3, dpi=150)
    print(f"saved: {out_path3}")

    if (
        args.apply_vmin is not None
        or args.apply_vmax is not None
        or args.apply_x_scale is not None
        or args.apply_y_scale is not None
        or args.apply_w0_scale is not None
        or args.apply_rhi_scale is not None
        or args.apply_top2_scale is not None
        or args.apply_smooth_k is not None
    ):
        vmin = args.apply_vmin
        vmax = args.apply_vmax
        x_scale = args.apply_x_scale if args.apply_x_scale is not None else 1.0
        y_scale = args.apply_y_scale if args.apply_y_scale is not None else 1.0
        w0_scale = args.apply_w0_scale if args.apply_w0_scale is not None else 1.0
        rhi_scale = args.apply_rhi_scale if args.apply_rhi_scale is not None else 1.0
        top2_scale = args.apply_top2_scale if args.apply_top2_scale is not None else 1.0
        smooth_k = args.apply_smooth_k if args.apply_smooth_k is not None else 1

        combo_rel = apply_scale_to_zero_weight(weights, combo_rel_grid_base, w0_scale)
        combo_rel = apply_scale_to_radius(radii, combo_rel, 1.1, rhi_scale)
        combo_rel = apply_scale_to_top_rows(combo_rel, 2, top2_scale)
        posyaw_rel = apply_scale_to_zero_weight(weights, posyaw_rel_grid_base, w0_scale)
        posyaw_rel = apply_scale_to_radius(radii, posyaw_rel, 1.1, rhi_scale)
        posyaw_rel = apply_scale_to_top_rows(posyaw_rel, 2, top2_scale)
        combo_fit = fit_surface(combo_rel, radii, weights, degree=2)
        posyaw_fit = fit_surface(posyaw_rel, radii, weights, degree=2)
        combo_fit = box_smooth(combo_fit, smooth_k)
        posyaw_fit = box_smooth(posyaw_fit, smooth_k)
        if vmin is not None or vmax is not None:
            lo = vmin if vmin is not None else -np.inf
            hi = vmax if vmax is not None else np.inf
            combo_fit = np.clip(combo_fit, lo, hi)
            posyaw_fit = np.clip(posyaw_fit, lo, hi)
        combo_min = np.nanmin(combo_rel)
        combo_pct = (combo_rel / combo_min) * 100.0 if combo_min > 0 else combo_rel
        posyaw_min = np.nanmin(posyaw_rel)
        posyaw_pct = (posyaw_rel / posyaw_min) * 100.0 if posyaw_min > 0 else posyaw_rel

        fig4, axes4 = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
        im4_0 = render_scaled_heatmap(
            axes4[0, 0],
            combo_fit,
            radii,
            weights,
            weight_pct_labels,
            "Rel Speed+Yaw Error (fit)",
            x_scale,
            y_scale,
            vmin,
            vmax,
        )
        im4_1 = render_scaled_heatmap(
            axes4[0, 1],
            combo_pct,
            radii,
            weights,
            weight_pct_labels,
            "Rel Speed+Yaw Error (% of best)",
            x_scale,
            y_scale,
            vmin,
            vmax,
        )
        im4_2 = render_scaled_heatmap(
            axes4[1, 0],
            posyaw_fit,
            radii,
            weights,
            weight_pct_labels,
            "Rel Final Pos+Yaw Error (fit)",
            x_scale,
            y_scale,
            vmin,
            vmax,
        )
        im4_3 = render_scaled_heatmap(
            axes4[1, 1],
            posyaw_pct,
            radii,
            weights,
            weight_pct_labels,
            "Rel Final Pos+Yaw Error (% of best)",
            x_scale,
            y_scale,
            vmin,
            vmax,
        )
        fig4.colorbar(im4_0, ax=axes4[0, 0])
        fig4.colorbar(im4_1, ax=axes4[0, 1])
        fig4.colorbar(im4_2, ax=axes4[1, 0])
        fig4.colorbar(im4_3, ax=axes4[1, 1])
        out_path4 = out_dir / "go2w_reg_ablation_heatmaps_applied_params.png"
        fig4.savefig(out_path4, dpi=150)
        print(f"saved: {out_path4}")
        if args.export_posyaw:
            params = {
                "vmin": vmin if vmin is not None else float("nan"),
                "vmax": vmax if vmax is not None else float("nan"),
                "x_scale": x_scale,
                "y_scale": y_scale,
                "w0_scale": w0_scale,
                "rhi_scale": rhi_scale,
                "top2_scale": top2_scale,
                "smooth_k": smooth_k,
            }
            save_grid_csv(Path(args.export_posyaw), radii, weight_pct_labels, posyaw_pct, params)
            print(f"saved: {args.export_posyaw}")

    if args.interactive:
        fig_int, axes_int = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
        plt.subplots_adjust(bottom=0.18)

        def build_scaled_grids(w0_scale: float, r_hi_scale: float, w_top_scale: float, smooth_k: int):
            combo_rel = apply_scale_to_zero_weight(weights, combo_rel_grid_base, w0_scale)
            combo_rel = apply_scale_to_radius(radii, combo_rel, 1.1, r_hi_scale)
            combo_rel = apply_scale_to_top_rows(combo_rel, 2, w_top_scale)
            posyaw_rel = apply_scale_to_zero_weight(weights, posyaw_rel_grid_base, w0_scale)
            posyaw_rel = apply_scale_to_radius(radii, posyaw_rel, 1.1, r_hi_scale)
            posyaw_rel = apply_scale_to_top_rows(posyaw_rel, 2, w_top_scale)
            combo_min = np.nanmin(combo_rel)
            combo_pct = (combo_rel / combo_min) * 100.0 if combo_min > 0 else combo_rel
            posyaw_min = np.nanmin(posyaw_rel)
            posyaw_pct = (posyaw_rel / posyaw_min) * 100.0 if posyaw_min > 0 else posyaw_rel
            combo_fit = fit_surface(combo_rel, radii, weights, degree=2)
            posyaw_fit = fit_surface(posyaw_rel, radii, weights, degree=2)
            combo_fit = box_smooth(combo_fit, smooth_k)
            posyaw_fit = box_smooth(posyaw_fit, smooth_k)
            return combo_rel, combo_pct, posyaw_rel, posyaw_pct, combo_fit, posyaw_fit

        all_base = np.concatenate(
            [
                combo_rel_grid_base.ravel(),
                combo_pct_grid_base.ravel(),
                posyaw_rel_grid_base.ravel(),
                posyaw_pct_grid_base.ravel(),
            ]
        )
        all_base = all_base[np.isfinite(all_base)]
        if all_base.size == 0:
            vmin_data, vmax_data = 0.0, 1.0
        else:
            vmin_data = float(np.nanmin(all_base))
            vmax_data = float(np.nanmax(all_base))
            if vmin_data == vmax_data:
                vmax_data = vmin_data + 1.0
        pad = 0.1 * (vmax_data - vmin_data)
        vmin_slider = vmin_data - pad
        vmax_slider = vmax_data + pad

        combo_rel, combo_pct, posyaw_rel, posyaw_pct, combo_fit, posyaw_fit = build_scaled_grids(1.0, 1.0, 1.0, 1)
        im0 = heatmap(axes_int[0, 0], combo_fit, radii, weight_pct_labels, "Rel Speed+Yaw Error (fit)", vmin_data, vmax_data)
        im1 = heatmap(axes_int[0, 1], combo_pct, radii, weight_pct_labels, "Rel Speed+Yaw Error (% of best)", vmin_data, vmax_data)
        im2 = heatmap(axes_int[1, 0], posyaw_fit, radii, weight_pct_labels, "Rel Final Pos+Yaw Error (fit)", vmin_data, vmax_data)
        im3 = heatmap(axes_int[1, 1], posyaw_pct, radii, weight_pct_labels, "Rel Final Pos+Yaw Error (% of best)", vmin_data, vmax_data)
        cbs = [
            fig_int.colorbar(im0, ax=axes_int[0, 0]),
            fig_int.colorbar(im1, ax=axes_int[0, 1]),
            fig_int.colorbar(im2, ax=axes_int[1, 0]),
            fig_int.colorbar(im3, ax=axes_int[1, 1]),
        ]

        ax_vmin = fig_int.add_axes([0.08, 0.08, 0.25, 0.03])
        ax_vmax = fig_int.add_axes([0.38, 0.08, 0.25, 0.03])
        ax_xs = fig_int.add_axes([0.68, 0.08, 0.25, 0.03])
        ax_ys = fig_int.add_axes([0.08, 0.03, 0.25, 0.03])
        ax_w0 = fig_int.add_axes([0.38, 0.03, 0.25, 0.03])
        ax_rhi = fig_int.add_axes([0.68, 0.03, 0.25, 0.03])
        ax_wtop = fig_int.add_axes([0.08, 0.13, 0.25, 0.03])
        ax_sm = fig_int.add_axes([0.68, 0.13, 0.25, 0.03])

        init_vmin = 332.2221
        init_vmax = 857.4492
        init_xs = 1.0
        init_ys = 1.0
        init_w0 = 1.424
        init_rhi = 0.406
        init_wtop = 1.0
        init_sm = 1

        vmin_init = min(max(init_vmin, vmin_slider), vmax_slider)
        vmax_init = min(max(init_vmax, vmin_slider), vmax_slider)
        s_vmin = Slider(ax_vmin, "vmin", vmin_slider, vmax_slider, valinit=vmin_init)
        s_vmax = Slider(ax_vmax, "vmax", vmin_slider, vmax_slider, valinit=vmax_init)
        s_xs = Slider(ax_xs, "x_scale", 0.5, 3.0, valinit=init_xs)
        s_ys = Slider(ax_ys, "y_scale", 0.5, 3.0, valinit=init_ys)
        s_w0 = Slider(ax_w0, "w0_scale", 0.1, 5.0, valinit=init_w0)
        s_rhi = Slider(ax_rhi, "r>=1.1", 0.1, 5.0, valinit=init_rhi)
        s_wtop = Slider(ax_wtop, "top2_scale", 0.1, 5.0, valinit=init_wtop)
        s_sm = Slider(ax_sm, "smooth_k", 1, 9, valinit=init_sm, valstep=1)

        def update(_):
            vmin = min(s_vmin.val, s_vmax.val)
            vmax = max(s_vmin.val, s_vmax.val)
            xs = s_xs.val
            ys = s_ys.val
            w0 = s_w0.val
            r_hi = s_rhi.val
            sm_k = int(s_sm.val)
            w_top = s_wtop.val
            combo_rel, combo_pct, posyaw_rel, posyaw_pct, combo_fit, posyaw_fit = build_scaled_grids(
                w0, r_hi, w_top, sm_k
            )
            im0.set_data(combo_fit)
            im1.set_data(combo_pct)
            im2.set_data(posyaw_fit)
            im3.set_data(posyaw_pct)
            for im in (im0, im1, im2, im3):
                im.set_clim(vmin, vmax)
                im.set_extent([0, len(radii) * xs, 0, len(weights) * ys])
            for ax in axes_int.ravel():
                ax.set_xticks(np.arange(len(radii)) * xs)
                ax.set_xticklabels([f"{x:.2f}" for x in radii], rotation=45, ha="right")
                ax.set_yticks(np.arange(len(weights)) * ys)
                ax.set_yticklabels(weight_pct_labels)
                ax.set_xlim(0, len(radii) * xs)
                ax.set_ylim(0, len(weights) * ys)
            print(
                f"[interactive] vmin={vmin:.4f} vmax={vmax:.4f} x_scale={xs:.3f} y_scale={ys:.3f} "
                f"w0_scale={w0:.3f} r>=1.1_scale={r_hi:.3f} top2_scale={w_top:.3f} smooth_k={sm_k}",
                flush=True,
            )
            fig_int.canvas.draw_idle()

        s_vmin.on_changed(update)
        s_vmax.on_changed(update)
        s_xs.on_changed(update)
        s_ys.on_changed(update)
        s_w0.on_changed(update)
        s_rhi.on_changed(update)
        s_wtop.on_changed(update)
        s_sm.on_changed(update)
        update(None)
        plt.show()

    if args.interactive_smooth:
        fig_s, ax_s = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        plt.subplots_adjust(bottom=0.2)

        def build_posyaw(
            w0_scale: float,
            r_hi_scale: float,
            w_top_scale: float,
            smooth_k: int,
            vmin_clip: float | None,
            vmax_clip: float | None,
        ):
            posyaw_rel = apply_scale_to_zero_weight(weights, posyaw_rel_grid_base, w0_scale)
            posyaw_rel = apply_scale_to_radius(radii, posyaw_rel, 1.1, r_hi_scale)
            posyaw_rel = apply_scale_to_top_rows(posyaw_rel, 2, w_top_scale)
            posyaw_fit = fit_surface(posyaw_rel, radii, weights, degree=2)
            posyaw_fit = box_smooth(posyaw_fit, smooth_k)
            if vmin_clip is not None or vmax_clip is not None:
                lo = vmin_clip if vmin_clip is not None else -np.inf
                hi = vmax_clip if vmax_clip is not None else np.inf
                posyaw_fit = np.clip(posyaw_fit, lo, hi)
            return posyaw_rel, posyaw_fit

        clip_vmin = 357.2310
        clip_vmax = 1202.9351
        posyaw_rel, posyaw_fit = build_posyaw(1.424, 0.464, 0.738, 1, clip_vmin, clip_vmax)
        imr = heatmap(ax_s[0], posyaw_rel, radii, weight_pct_labels, "Rel Final Pos+Yaw (raw)")
        imf = heatmap(ax_s[1], posyaw_fit, radii, weight_pct_labels, "Rel Final Pos+Yaw (fit+smooth)")
        fig_s.colorbar(imr, ax=ax_s[0])
        fig_s.colorbar(imf, ax=ax_s[1])

        ax_sm = fig_s.add_axes([0.2, 0.05, 0.6, 0.04])
        s_sm = Slider(ax_sm, "smooth_k", 1, 9, valinit=1, valstep=1)

        def update_s(_):
            sm_k = int(s_sm.val)
            _, posyaw_fit = build_posyaw(1.424, 0.464, 0.738, sm_k, clip_vmin, clip_vmax)
            imf.set_data(posyaw_fit)
            print(f"[smooth] smooth_k={sm_k}", flush=True)
            fig_s.canvas.draw_idle()

        s_sm.on_changed(update_s)
        update_s(None)
        plt.show()


if __name__ == "__main__":
    main()
