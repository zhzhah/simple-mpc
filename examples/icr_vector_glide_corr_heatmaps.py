#!/usr/bin/env python3
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pinocchio as pin
from scipy.stats import spearmanr


def rpy_from_quat_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
    quat = pin.Quaternion(float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))
    return pin.rpy.matrixToRpy(quat.toRotationMatrix())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", type=str, default="examples/analysis_outputs_wz3/trajectories")
    parser.add_argument("--output-dir", type=str, default="examples/analysis_outputs_wz3")
    parser.add_argument("--rho-thresh", type=float, default=0.4)
    parser.add_argument("--p-thresh", type=float, default=0.05)
    args = parser.parse_args()

    traj_dir = Path(args.traj_dir)
    if not traj_dir.exists():
        raise SystemExit(f"Trajectory dir not found: {traj_dir}")

    files = sorted(traj_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"No .npz files in {traj_dir}")

    # Collect metrics per (radius, weight)
    records = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        radius = float(d["radius"])
        weight = float(d["weight_base"])
        xs = np.asarray(d["xs"])
        refs = np.asarray(d["refs"])
        if xs.size == 0 or refs.size == 0:
            continue

        x_final = xs[-1]
        x_ref = refs[-1]
        qf = x_final[:7]
        qr = x_ref[:7]
        vf = x_final[7:13]
        vr = x_ref[7:13]

        rpyf = rpy_from_quat_xyzw(qf[3:7])
        rpyr = rpy_from_quat_xyzw(qr[3:7])
        yaw_err = ((rpyf[2] - rpyr[2] + np.pi) % (2 * np.pi)) - np.pi

        pos_err = float(np.linalg.norm(qf[:3] - qr[:3]))
        vel_err = float(np.linalg.norm(vf[:3] - vr[:3]))
        yawrate_err = float(abs(vf[5] - vr[5]))

        q0 = np.asarray(d["q_init"])
        qmin = np.asarray(d["qmin"])
        qmax = np.asarray(d["qmax"])
        jidx = np.asarray(d["joint_pos_indices"])
        if jidx.size:
            q0j = q0[jidx]
            jrange = np.maximum(qmax[jidx] - qmin[jidx], 1e-6)
            jth = 0.3 * jrange
            dj = np.abs(xs[:, jidx] - q0j)
            joint_violation = (dj > jth).mean(axis=1)
            joint_violation_frac = float(joint_violation.mean())
        else:
            joint_violation_frac = np.nan

        records.append(
            dict(
                radius=radius,
                weight=weight,
                pos_err=pos_err,
                yaw_err=abs(float(yaw_err)),
                vel_err=vel_err,
                yawrate_err=yawrate_err,
                joint_violation_frac=joint_violation_frac,
            )
        )

    if not records:
        raise SystemExit("No valid trajectory records")

    radius_vals = sorted({int(r["radius"]) for r in records})
    weight_vals = sorted({int(r["weight"]) for r in records})
    r_to_i = {r: i for i, r in enumerate(radius_vals)}
    w_to_j = {w: j for j, w in enumerate(weight_vals)}

    metrics = ["pos_err", "yaw_err", "vel_err", "yawrate_err", "joint_violation_frac"]
    maps = {m: np.full((len(radius_vals), len(weight_vals)), np.nan, dtype=float) for m in metrics}

    for rec in records:
        i = r_to_i[int(rec["radius"])]
        j = w_to_j[int(rec["weight"])]
        for m in metrics:
            maps[m][i, j] = rec[m]

    # Compute spearman correlation across weights (mean over radii)
    agg = defaultdict(list)
    for rec in records:
        agg[rec["weight"]].append(rec)

    weights = sorted(agg.keys())
    mean_by_weight = {m: [] for m in metrics}
    for w in weights:
        arr = np.array([[r[m] for m in metrics] for r in agg[w]], float)
        mean = np.nanmean(arr, axis=0)
        for idx, m in enumerate(metrics):
            mean_by_weight[m].append(mean[idx])

    correlated = []
    for m in metrics:
        rho, p = spearmanr(weights, mean_by_weight[m])
        if np.isfinite(rho) and abs(rho) >= args.rho_thresh and p <= args.p_thresh:
            correlated.append((m, float(rho), float(p)))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "vector_glide_correlated_metrics.txt"
    with summary_path.open("w") as f:
        if not correlated:
            f.write("No metrics passed the correlation thresholds.\n")
        else:
            f.write("metric\trho\tp\n")
            for m, rho, p in correlated:
                f.write(f"{m}\t{rho:.6g}\t{p:.6g}\n")

    for m, rho, p in correlated:
        plt.figure(figsize=(8, 6))
        plt.imshow(
            maps[m],
            origin="lower",
            aspect="auto",
            extent=[weight_vals[0], weight_vals[-1], radius_vals[0], radius_vals[-1]],
            cmap="viridis",
        )
        plt.colorbar(label=f"{m}")
        plt.xlabel("vector_glide_weight_base")
        plt.ylabel("ICR radius (m, left)")
        plt.title(f"{m} (rho={rho:.3f}, p={p:.3g})")
        plt.tight_layout()
        plt.savefig(out_dir / f"corr_{m}_heatmap.png", dpi=200)
        plt.close()

    print(f"Saved correlated heatmaps to {out_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
