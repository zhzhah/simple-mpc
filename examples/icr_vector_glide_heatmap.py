#!/usr/bin/env python3
import argparse
from pathlib import Path
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pinocchio as pin


def rpy_from_quat_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
    quat = pin.Quaternion(float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))
    R = quat.toRotationMatrix()
    return pin.rpy.matrixToRpy(R)


def load_npz_files(traj_dir: Path):
    files = sorted(traj_dir.glob("*.npz"))
    data = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        data.append((f, d))
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", type=str, default="examples/analysis_outputs_wz3/trajectories")
    parser.add_argument("--output-dir", type=str, default="examples/analysis_outputs_wz3")
    args = parser.parse_args()

    traj_dir = Path(args.traj_dir)
    if not traj_dir.exists():
        raise SystemExit(f"Trajectory dir not found: {traj_dir}")

    data = load_npz_files(traj_dir)
    if not data:
        raise SystemExit(f"No .npz files in {traj_dir}")

    radius_vals = sorted({int(np.load(f, allow_pickle=True)["radius"]) for f, _ in data})
    weight_vals = sorted({int(np.load(f, allow_pickle=True)["weight_base"]) for f, _ in data})

    r_to_i = {r: i for i, r in enumerate(radius_vals)}
    w_to_j = {w: j for j, w in enumerate(weight_vals)}

    final_err_map = np.full((len(radius_vals), len(weight_vals)), np.nan, dtype=float)
    final_err_abs_map = np.full((len(radius_vals), len(weight_vals)), np.nan, dtype=float)
    yaw_err_map = np.full((len(radius_vals), len(weight_vals)), np.nan, dtype=float)

    for f, d in data:
        radius = int(d["radius"].item())
        weight = int(d["weight_base"].item())
        if radius not in r_to_i or weight not in w_to_j:
            continue
        xs = np.asarray(d["xs"])
        refs = np.asarray(d["refs"])
        if xs.size == 0 or refs.size == 0:
            continue
        x_final = xs[-1]
        x_ref = refs[-1]
        q_final = x_final[:7]
        q_ref = x_ref[:7]
        v_final = x_final[7:13]
        v_ref = x_ref[7:13]

        pos_err = q_final[:3] - q_ref[:3]
        rpy_final = rpy_from_quat_xyzw(q_final[3:7])
        rpy_ref = rpy_from_quat_xyzw(q_ref[3:7])
        yaw_err = rpy_final[2] - rpy_ref[2]
        yaw_err = (yaw_err + np.pi) % (2.0 * np.pi) - np.pi

        vel_err = v_final[:3] - v_ref[:3]
        yaw_rate_err = v_final[5] - v_ref[5]

        err_vec = np.array([
            pos_err[0], pos_err[1], pos_err[2],
            yaw_err,
            vel_err[0], vel_err[1], vel_err[2],
            yaw_rate_err,
        ], dtype=float)
        # Normalize by target-to-initial deltas to make units comparable
        x_init = np.asarray(d["x_measured"])
        q_init = x_init[:7]
        v_init = x_init[7:13]
        rpy_init = rpy_from_quat_xyzw(q_init[3:7])
        pos_scale = np.abs(q_ref[:3] - q_init[:3])
        yaw_scale = abs(rpy_ref[2] - rpy_init[2])
        vel_scale = np.abs(v_ref[:3] - v_init[:3])
        yaw_rate_scale = abs(v_ref[5] - v_init[5])
        scale_vec = np.array(
            [
                pos_scale[0], pos_scale[1], pos_scale[2],
                yaw_scale,
                vel_scale[0], vel_scale[1], vel_scale[2],
                yaw_rate_scale,
            ],
            dtype=float,
        )
        scale_vec = np.where(scale_vec > 1e-6, scale_vec, 1.0)
        err_vec_scaled = err_vec / scale_vec
        err = float(np.linalg.norm(err_vec_scaled))
        err_norm = err / (1.0 + float(radius))

        final_err_map[r_to_i[radius], w_to_j[weight]] = err_norm
        final_err_abs_map[r_to_i[radius], w_to_j[weight]] = err
        yaw_err_map[r_to_i[radius], w_to_j[weight]] = abs(yaw_err)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(
        final_err_map,
        origin="lower",
        aspect="auto",
        extent=[weight_vals[0], weight_vals[-1], radius_vals[0], radius_vals[-1]],
        cmap="viridis",
    )
    plt.colorbar(label="Final error /(1+radius)")
    plt.xlabel("vector_glide_weight_base")
    plt.ylabel("ICR radius (m, left)")
    plt.title("Final error (pos+ yaw + vel + yawrate) normalized by radius")
    plt.tight_layout()
    plt.savefig(out_dir / "final_error_norm_radius_heatmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.imshow(
        final_err_abs_map,
        origin="lower",
        aspect="auto",
        extent=[weight_vals[0], weight_vals[-1], radius_vals[0], radius_vals[-1]],
        cmap="viridis",
    )
    plt.colorbar(label="Final error (absolute)")
    plt.xlabel("vector_glide_weight_base")
    plt.ylabel("ICR radius (m, left)")
    plt.title("Final error (pos+ yaw + vel + yawrate) absolute")
    plt.tight_layout()
    plt.savefig(out_dir / "final_error_abs_heatmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.imshow(
        yaw_err_map,
        origin="lower",
        aspect="auto",
        extent=[weight_vals[0], weight_vals[-1], radius_vals[0], radius_vals[-1]],
        cmap="viridis",
    )
    plt.colorbar(label="Terminal yaw error (rad)")
    plt.xlabel("vector_glide_weight_base")
    plt.ylabel("ICR radius (m, left)")
    plt.title("Terminal yaw error (absolute)")
    plt.tight_layout()
    plt.savefig(out_dir / "terminal_yaw_error_heatmap.png", dpi=200)
    plt.close()

    np.save(out_dir / "final_error_norm_radius.npy", final_err_map)
    np.save(out_dir / "final_error_abs.npy", final_err_abs_map)
    np.save(out_dir / "terminal_yaw_error.npy", yaw_err_map)
    print(f"Saved heatmap to {out_dir}")


if __name__ == "__main__":
    main()
