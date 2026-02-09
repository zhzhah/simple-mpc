#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pinocchio as pin

from simple_mpc import KinodynamicsOCP, MPC
from urdf_joint_viewer_lowfps_mpc_reg_circ import (
    build_mpc,
    quat_from_rpy,
    FIXED_URDF,
    LOW_FPS_CSV,
)


def rpy_from_quat_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
    quat = pin.Quaternion(float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]))
    R = quat.toRotationMatrix()
    return pin.rpy.matrixToRpy(R)


def build_arc_reference_states(
    model_handler,
    horizon_T: int,
    dt: float,
    yaw0: float,
    roll0: float,
    pitch0: float,
    joints0: np.ndarray,
    base_z: float,
    vx_cmd: float,
    omega_cmd: float,
) -> list:
    model = model_handler.getModel()
    nq = model.nq
    refs = []
    cy_local = math.cos(yaw0)
    sy_local = math.sin(yaw0)
    for t_idx in range(horizon_T):
        t = t_idx * dt
        if abs(omega_cmd) < 1e-9:
            dx_local = vx_cmd * t
            dy_local = 0.0
        else:
            R_local = vx_cmd / omega_cmd
            dtheta_local = omega_cmd * t
            dx_local = R_local * math.sin(dtheta_local)
            dy_local = R_local * (1.0 - math.cos(dtheta_local))
        dx_world = cy_local * dx_local - sy_local * dy_local
        dy_world = sy_local * dx_local + cy_local * dy_local
        q_ref = model_handler.getReferenceState()[:nq].copy()
        q_ref[0] = dx_world
        q_ref[1] = dy_world
        q_ref[2] = base_z
        yaw_ref = yaw0 + omega_cmd * t
        q_ref[3:7] = quat_from_rpy(roll0, pitch0, yaw_ref)
        for joint_name, qv in zip(JOINT_ORDER, joints0):
            joint_id = model.getJointId(joint_name)
            idx_q = model.joints[joint_id].idx_q
            q_ref[idx_q : idx_q + model.joints[joint_id].nq] = qv
        x_ref = model_handler.getReferenceState().copy()
        x_ref[:nq] = q_ref
        x_ref[nq:] = 0.0
        refs.append(x_ref)
    return refs


def compute_metrics(
    mpc,
    x_ref_terminal: np.ndarray,
    q0: np.ndarray,
    qmin: np.ndarray,
    qmax: np.ndarray,
    roll0: float,
    pitch0: float,
    dt: float,
    joint_pos_indices: np.ndarray,
) -> tuple:
    xs = mpc.xs
    if not xs:
        return np.nan, np.nan, np.nan
    x_final = np.asarray(xs[-1])
    q_final = x_final[: q0.shape[0]]
    q_ref = x_ref_terminal[: q0.shape[0]]
    pos_err = q_final[:3] - q_ref[:3]
    rpy_final = rpy_from_quat_xyzw(q_final[3:7])
    rpy_ref = rpy_from_quat_xyzw(q_ref[3:7])
    rpy_err = rpy_final - rpy_ref
    rpy_err = (rpy_err + np.pi) % (2.0 * np.pi) - np.pi
    final_err = float(np.linalg.norm(np.concatenate([pos_err, rpy_err])))

    roll_pitch_thresh = np.deg2rad(20.0)
    q0_j = q0[joint_pos_indices]
    qmin_j = qmin[joint_pos_indices]
    qmax_j = qmax[joint_pos_indices]
    joint_range = np.maximum(qmax_j - qmin_j, 1e-6)
    joint_thresh = 0.3 * joint_range

    body_violation_steps = 0
    joint_violation_acc = 0.0
    for xk in xs:
        qk = np.asarray(xk)[: q0.shape[0]]
        rpy = rpy_from_quat_xyzw(qk[3:7])
        if abs(rpy[0] - roll0) > roll_pitch_thresh or abs(rpy[1] - pitch0) > roll_pitch_thresh:
            body_violation_steps += 1
        dj = np.abs(qk[joint_pos_indices] - q0_j)
        joint_violation_acc += float(np.mean(dj > joint_thresh))

    body_time = body_violation_steps * dt
    joint_time = joint_violation_acc * dt
    weighted = 0.5 * body_time + 0.5 * joint_time
    return final_err, body_time, weighted


JOINT_ORDER = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_wheel_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_wheel_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_wheel_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_wheel_joint",
]


def compute_point(
    radius: float,
    weight_base: float,
    model_handler,
    problem_conf,
    gravity,
    force_size,
    horizon_T,
    mpc_conf,
    contact_phase_quadru,
    q_init,
    x_measured,
    qmin,
    qmax,
    roll0,
    pitch0,
    yaw0,
    joints0,
    base_z,
    dt,
    joint_pos_indices,
) -> tuple:
    omega = omega_fixed
    vx_cmd = float(omega * radius)
    problem_conf["icr_arc_cstr"] = True
    problem_conf["icr_arc_min_omega"] = 1e-9
    problem_conf["use_vector_glide_cost"] = True
    problem_conf["vector_glide_weight_base"] = float(weight_base)
    problem_conf["vector_glide_weight"] = float(weight_base)

    dynproblem = KinodynamicsOCP(problem_conf, model_handler)
    dynproblem.createProblem(model_handler.getReferenceState(), horizon_T, force_size, gravity[2], False)
    mpc = MPC(mpc_conf, dynproblem)
    mpc.generateCycleHorizon([contact_phase_quadru])

    v_world_cmd = np.array([vx_cmd, 0.0, 0.0, 0.0, 0.0, omega])
    mpc.velocity_base = v_world_cmd

    if abs(radius) < 1e-9:
        icr_params = np.array([0.0, 0.0, 0.0, 0.0])
    else:
        cx = -math.sin(yaw0) * radius
        cy = math.cos(yaw0) * radius
        icr_params = np.array([cx, cy, radius, 1.0])
    try:
        mpc.ocp_handler.setIcrArcParams(icr_params)
    except Exception:
        pass

    refs = build_arc_reference_states(
        model_handler,
        horizon_T,
        dt,
        yaw0,
        roll0,
        pitch0,
        joints0,
        base_z,
        vx_cmd,
        omega,
    )
    for t_idx, x_ref in enumerate(refs):
        mpc.ocp_handler.setReferenceState(t_idx, x_ref)
    mpc.ocp_handler.setTerminalReferenceState(refs[-1])

    mpc.iterate(x_measured)

    final_err, body_time, weighted_time = compute_metrics(
        mpc,
        refs[-1],
        q_init,
        qmin,
        qmax,
        roll0,
        pitch0,
        dt,
        joint_pos_indices,
    )
    return final_err, body_time, weighted_time, refs, mpc.xs, mpc.us


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius-min", type=float, default=0.0)
    parser.add_argument("--radius-max", type=float, default=20.0)
    parser.add_argument("--radius-steps", type=int, default=21)
    parser.add_argument("--weight-min", type=float, default=0.0)
    parser.add_argument("--weight-max", type=float, default=200000.0)
    parser.add_argument("--weight-steps", type=int, default=21)
    parser.add_argument("--fixed-vx", type=float, default=2.0)
    parser.add_argument("--fixed-wz", type=float, default=3.0)
    parser.add_argument("--fix-icr-center", action="store_true", help="Do not vary ICR center; use fixed vx/wz")
    parser.add_argument("--weight-exp", action="store_true", help="Use exponential spacing for weight values")
    parser.add_argument("--output-dir", type=str, default="examples/analysis_outputs")
    parser.add_argument("--subprocess", action="store_true", help="Run each grid point in a subprocess")
    parser.add_argument("--single", action="store_true", help="Compute a single grid point and write JSON")
    parser.add_argument("--radius", type=float, default=0.0)
    parser.add_argument("--weight", type=float, default=0.0)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--output-npz", type=str, default="")
    parser.add_argument("--save-trajectories", action="store_true", help="Save full trajectories per grid point")
    args = parser.parse_args()

    if not FIXED_URDF.exists():
        raise SystemExit(f"URDF not found: {FIXED_URDF}")
    if not LOW_FPS_CSV.exists():
        raise SystemExit(f"Low-FPS CSV not found: {LOW_FPS_CSV}")

    df = pd.read_csv(LOW_FPS_CSV)
    joint_cols = [f"JointPos_{i}" for i in range(16)]
    for c in joint_cols:
        if c not in df.columns:
            raise SystemExit(f"Missing column: {c}")

    q = df[joint_cols].to_numpy()
    world_xyz = df[["world_x", "world_y", "world_z"]].to_numpy()
    yaw = df["yaw"].to_numpy()
    roll = df["CoMAngle_0"].to_numpy() if "CoMAngle_0" in df.columns else np.zeros_like(yaw)
    pitch = df["CoMAngle_1"].to_numpy() if "CoMAngle_1" in df.columns else np.zeros_like(yaw)

    frame0 = 0
    base_xyz0 = world_xyz[frame0].copy()
    roll0 = float(roll[frame0])
    pitch0 = float(pitch[frame0])
    yaw0 = float(yaw[frame0])
    joints0 = q[frame0].copy()

    model_handler, _data_handler, problem_conf, gravity, force_size, horizon_T, mpc_conf, contact_phase_quadru = build_mpc()
    model = model_handler.getModel()
    nq = model.nq
    nv = model.nv

    q_init = model_handler.getReferenceState()[:nq].copy()
    q_init[0] = 0.0
    q_init[1] = 0.0
    q_init[2] = base_xyz0[2]
    q_init[3:7] = quat_from_rpy(roll0, pitch0, yaw0)
    for joint_name, qv in zip(JOINT_ORDER, joints0):
        joint_id = model.getJointId(joint_name)
        idx_q = model.joints[joint_id].idx_q
        q_init[idx_q : idx_q + model.joints[joint_id].nq] = qv

    x_measured = model_handler.getReferenceState().copy()
    x_measured[:nq] = q_init
    x_measured[nq:] = 0.0

    qmin = model.lowerPositionLimit.copy()
    qmax = model.upperPositionLimit.copy()

    dt = float(problem_conf.get("timestep", 0.01))

    radius_max = float(args.radius_max)
    radius_min = float(args.radius_min)
    if radius_max <= 0.0:
        raise SystemExit("radius_max must be > 0")
    if args.fix_icr_center:
        if abs(args.fixed_wz) < 1e-9:
            radius_vals = np.array([0.0], dtype=float)
        else:
            radius_vals = np.array([float(args.fixed_vx / args.fixed_wz)], dtype=float)
    else:
        log_count = max(int(args.radius_steps), 2)
        log_grid = np.logspace(-3, np.log10(radius_max), log_count)
        log_grid = np.clip(log_grid, 0.0, radius_max)
        log_grid = np.round(log_grid).astype(int)
        radius_vals = np.unique(np.concatenate(([0], log_grid)))
        radius_vals = radius_vals[(radius_vals >= radius_min) & (radius_vals <= radius_max)]
        if radius_vals.size == 0:
            radius_vals = np.array([0], dtype=int)

    if args.weight_exp:
        wmin = float(args.weight_min)
        wmax = float(args.weight_max)
        steps_w = max(int(args.weight_steps), 2)
        if wmax <= 0.0:
            weight_vals = np.array([0.0])
        else:
            if wmin <= 0.0:
                wmin_pos = 1.0
                log_grid = np.logspace(np.log10(wmin_pos), np.log10(wmax), steps_w - 1)
                weight_vals = np.concatenate(([0.0], log_grid))
            else:
                weight_vals = np.logspace(np.log10(wmin), np.log10(wmax), steps_w)
    else:
        weight_vals = np.linspace(args.weight_min, args.weight_max, args.weight_steps)

    final_err_map = np.full((radius_vals.size, args.weight_steps), np.nan, dtype=float)
    weighted_time_map = np.full((radius_vals.size, args.weight_steps), np.nan, dtype=float)
    body_time_map = np.full((radius_vals.size, args.weight_steps), np.nan, dtype=float)

    wheel_joint_names = {
        "FL_wheel_joint",
        "FR_wheel_joint",
        "RL_wheel_joint",
        "RR_wheel_joint",
    }
    joint_pos_indices = []
    for jname in model.names[1:]:
        if jname in wheel_joint_names:
            continue
        jid = model.getJointId(jname)
        if jid <= 0:
            continue
        idx_q = model.joints[jid].idx_q
        nq_j = model.joints[jid].nq
        for k in range(nq_j):
            joint_pos_indices.append(idx_q + k)
    joint_pos_indices = np.array(sorted(set(joint_pos_indices)), dtype=int)

    omega_fixed = float(args.fixed_wz)
    vx_fixed = float(args.fixed_vx)
    if args.single:
        try:
            final_err, body_time, weighted_time, refs, xs, us = compute_point(
                args.radius,
                args.weight,
                model_handler,
                problem_conf,
                gravity,
                force_size,
                horizon_T,
                mpc_conf,
                contact_phase_quadru,
                q_init,
                x_measured,
                qmin,
                qmax,
                roll0,
                pitch0,
                yaw0,
                joints0,
                base_xyz0[2],
                dt,
                joint_pos_indices,
            )
            payload = {
                "final_err": final_err,
                "body_time": body_time,
                "weighted_time": weighted_time,
            }
            if args.output_npz:
                np.savez_compressed(
                    args.output_npz,
                    radius=float(args.radius),
                    weight_base=float(args.weight),
                    omega=omega_fixed,
                    vx=float(vx_fixed if args.fix_icr_center else (omega_fixed * args.radius)),
                    x_measured=x_measured,
                    x_ref_terminal=refs[-1],
                    refs=np.asarray(refs),
                    xs=np.asarray(xs),
                    us=np.asarray(us),
                    q_init=q_init,
                    qmin=qmin,
                    qmax=qmax,
                    roll0=roll0,
                    pitch0=pitch0,
                    yaw0=yaw0,
                    joint_pos_indices=joint_pos_indices,
                )
        except Exception:
            payload = {"final_err": None, "body_time": None, "weighted_time": None}
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload))
        return

    traj_dir = Path(args.output_dir) / "trajectories"
    if args.save_trajectories:
        traj_dir.mkdir(parents=True, exist_ok=True)

    for i, R in enumerate(radius_vals):
        for j, w_base in enumerate(weight_vals):
            try:
                if args.subprocess:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tf:
                        out_json = tf.name
                    out_npz = ""
                    if args.save_trajectories:
                        out_npz = str(traj_dir / f"R{int(R):03d}_W{int(w_base):05d}.npz")
                    cmd = [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--single",
                        "--radius",
                        str(float(R)),
                        "--weight",
                        str(float(w_base)),
                        "--output-json",
                        out_json,
                    ]
                    if out_npz:
                        cmd.extend(["--output-npz", out_npz])
                    subprocess.run(cmd, check=False)
                    payload = json.loads(Path(out_json).read_text())
                    Path(out_json).unlink(missing_ok=True)
                    final_err = payload.get("final_err")
                    body_time = payload.get("body_time")
                    weighted_time = payload.get("weighted_time")
                    if final_err is None or body_time is None or weighted_time is None:
                        raise RuntimeError("single-run failed")
                else:
                    final_err, body_time, weighted_time, refs, xs, us = compute_point(
                        R,
                        w_base,
                        model_handler,
                        problem_conf,
                        gravity,
                        force_size,
                        horizon_T,
                        mpc_conf,
                        contact_phase_quadru,
                        q_init,
                        x_measured,
                        qmin,
                        qmax,
                        roll0,
                        pitch0,
                        yaw0,
                        joints0,
                        base_xyz0[2],
                        dt,
                        joint_pos_indices,
                    )
                    if args.save_trajectories:
                        np.savez_compressed(
                            traj_dir / f"R{int(R):03d}_W{int(w_base):05d}.npz",
                            radius=float(R),
                            weight_base=float(w_base),
                            omega=omega_fixed,
                            vx=float(vx_fixed if args.fix_icr_center else (omega_fixed * R)),
                            x_measured=x_measured,
                            x_ref_terminal=refs[-1],
                            refs=np.asarray(refs),
                            xs=np.asarray(xs),
                            us=np.asarray(us),
                            q_init=q_init,
                            qmin=qmin,
                            qmax=qmax,
                            roll0=roll0,
                            pitch0=pitch0,
                            yaw0=yaw0,
                            joint_pos_indices=joint_pos_indices,
                        )
                final_err_map[i, j] = final_err
                weighted_time_map[i, j] = weighted_time
                body_time_map[i, j] = body_time
            except Exception:
                final_err_map[i, j] = np.nan
                weighted_time_map[i, j] = np.nan
                body_time_map[i, j] = np.nan

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_heatmap(data: np.ndarray, title: str, fname: str, cmap: str = "viridis") -> None:
        plt.figure(figsize=(8, 6))
        plt.imshow(
            data,
            origin="lower",
            aspect="auto",
            extent=[args.weight_min, args.weight_max, radius_vals[0], radius_vals[-1]],
            cmap=cmap,
        )
        plt.colorbar(label=title)
        plt.xlabel("vector_glide_weight_base")
        plt.ylabel("ICR radius (m, left)")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=200)
        plt.close()

    save_heatmap(final_err_map, "Final state error norm", "final_error_heatmap.png")
    save_heatmap(weighted_time_map, "Weighted deviation time (body 50%)", "weighted_deviation_heatmap.png")

    np.save(out_dir / "final_error.npy", final_err_map)
    np.save(out_dir / "weighted_deviation.npy", weighted_time_map)
    np.save(out_dir / "body_deviation.npy", body_time_map)

    print(f"Saved heatmaps to {out_dir}")


if __name__ == "__main__":
    main()
