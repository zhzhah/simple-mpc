#!/usr/bin/env python3
"""Generate MPC error grid data for radius/weight sweeps."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

import pinocchio as pin
import example_robot_data as erd
from example_robot_data.robots_loader import ROBOTS, RobotLoader

from simple_mpc import KinodynamicsOCP, MPC, RobotModelHandler, RobotDataHandler

FIXED_URDF = Path("/home/yzc/MPCtest/traj/BQR3W_go2_description/urdf/BQR3W_go2_relative.urdf")
LOW_FPS_CSV = Path("/home/yzc/MPCtest/traj/processed_low.csv")

ENABLE_ICR_ARC_REF = False
ICR_ARC_POS_WEIGHT = 50.0


JOINT_ORDER = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_wheel_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_wheel_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_wheel_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_wheel_joint",
]


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    R = pin.rpy.rpyToMatrix(roll, pitch, yaw)
    return pin.Quaternion(R).coeffs()


def build_mpc():
    # Model loading (mirrors urdf_joint_viewer_lowfps_mpc_reg_circ.py)
    base_joint_name = "root_joint"
    if "BQR3W_go2name" not in ROBOTS:
        class BQR3WLoader(RobotLoader):
            path = "BQR3W_go2_description"
            urdf_filename = "BQR3W_go2.urdf"
            urdf_subpath = "urdf"
            srdf_filename = "BQR3_350.srdf"
            ref_posture = "standing"
            free_flyer = True

        ROBOTS["BQR3W_go2name"] = BQR3WLoader

    robot_wrapper = erd.load("BQR3W_go2name")
    if "standing" not in robot_wrapper.model.referenceConfigurations:
        robot_wrapper.model.referenceConfigurations["standing"] = pin.neutral(
            robot_wrapper.model
        )

    model_handler = RobotModelHandler(robot_wrapper.model, "standing", base_joint_name)
    model_handler.addPointFoot("FL_wheel", base_joint_name)
    model_handler.addPointFoot("FR_wheel", base_joint_name)
    model_handler.addPointFoot("RL_wheel", base_joint_name)
    model_handler.addPointFoot("RR_wheel", base_joint_name)
    data_handler = RobotDataHandler(model_handler)

    model = model_handler.getModel()
    gravity = np.array([0, 0, -9.81])
    force_size = 3

    dt_mpc = 0.01
    if ENABLE_ICR_ARC_REF:
        w_basepos = [ICR_ARC_POS_WEIGHT, ICR_ARC_POS_WEIGHT, 1, 0.0, 1000.0, 1000]
    else:
        w_basepos = [1, 1, 1, 9000.0, 9000.0, 1000]
    w_jointpos = [1879.64, 249.799, 220.196, 0.0]
    w_basevel = [0, 0, 0, 10, 10, 20]
    w_jointvel = [0.1, 0.1, 0.1, 0.1]

    w_basepos_T = [100000, 100000, 30000, 9000.0, 9000.0, 100000]
    w_jointpos_T = [1879.64, 249.799, 220.196, 0.0]
    w_basevel_T = [100, 100, 10, 10, 10, 100]
    w_jointvel_T = [0.1, 0.1, 0.1, 0.1]

    wheel_joints = [
        "FL_wheel_joint",
        "FR_wheel_joint",
        "RL_wheel_joint",
        "RR_wheel_joint",
    ]
    wheel_radius = 0.1015
    wheel_vel_limit = 3.0 * 100.0 * 2.0 * np.pi / 60.0

    for joint_name in wheel_joints:
        joint_id = model.getJointId(joint_name)
        idx_v = model.joints[joint_id].idx_v
        nv_joint = model.joints[joint_id].nv
        model.velocityLimit[idx_v : idx_v + nv_joint] = wheel_vel_limit

    w_q = np.zeros(model.nv)
    w_v = np.zeros(model.nv)
    w_q[:6] = w_basepos
    w_v[:6] = w_basevel
    w_q[6:] = np.resize(np.tile(w_jointpos, 4), w_q[6:].shape[0])
    w_v[6:] = np.resize(np.tile(w_jointvel, 4), w_v[6:].shape[0])

    w_x = np.diag(np.concatenate((w_q, w_v)))
    w_q_T = np.zeros(model.nv)
    w_v_T = np.zeros(model.nv)
    w_q_T[:6] = w_basepos_T
    w_v_T[:6] = w_basevel_T
    w_q_T[6:] = np.resize(np.tile(w_jointpos_T, 4), w_q_T[6:].shape[0])
    w_v_T[6:] = np.resize(np.tile(w_jointvel_T, 4), w_v_T[6:].shape[0])
    w_x_T = np.diag(np.concatenate((w_q_T, w_v_T)))
    w_linforce = np.array([0.01, 0.01, 0.01])
    w_u_joints = np.ones(model.nv - 6) * 1e-5
    w_u = np.concatenate((w_linforce, w_linforce, w_linforce, w_linforce, w_u_joints))
    w_u = np.diag(w_u)

    w_cent_lin = np.array([0.0, 0.0, 1])
    w_cent_ang = np.array([0.0, 0.1, 10])
    w_cent = np.diag(np.concatenate((w_cent_lin, w_cent_ang)))
    w_centder_lin = np.ones(3) * 0.0
    w_centder_ang = np.ones(3) * 0.1
    w_centder = np.diag(np.concatenate((w_centder_lin, w_centder_ang)))

    qmin = model_handler.getModel().lowerPositionLimit[7:].copy()
    qmax = model_handler.getModel().upperPositionLimit[7:].copy()
    for joint_name in wheel_joints:
        joint_id = model.getJointId(joint_name)
        idx_q = model.joints[joint_id].idx_q
        nq_joint = model.joints[joint_id].nq
        qmin[idx_q - 7 : idx_q - 7 + nq_joint] = -1e9
        qmax[idx_q - 7 : idx_q - 7 + nq_joint] = 1e9

    problem_conf = dict(
        timestep=dt_mpc,
        w_x=w_x,
        w_u=w_u,
        w_cent=w_cent,
        w_centder=w_centder,
        w_x_terminal=w_x_T,
        w_cent_terminal=w_cent * 1.0,
        gravity=gravity,
        force_size=3,
        w_frame=np.array([0.0, 0.0, 0.0]),
        qmin=qmin,
        qmax=qmax,
        mu=0.8,
        Lfoot=0.01,
        Wfoot=0.01,
        kinematics_limits=True,
        force_cone=False,
        land_cstr=False,
        nonholonomic_rolling=False,
        enable_lateral_no_slip=True,
        lateral_no_slip_min_axis_norm=1e-12,
        lateral_no_slip_min_cross_norm=1e-8,
        use_vector_glide_cost=True,
        vector_glide_weight=5000.0,
        vector_glide_weight_base=10992.2,
        vector_glide_disable_on_twist=True,
        vector_glide_min_omega=1e-6,
        min_wheel_distance_cstr=False,
        min_wheel_distance=1.2 * 2.0 * wheel_radius,
        min_wheel_distance_cost=True,
        w_min_wheel_distance=0.05,
        min_wheel_distance_cost_eps=1e-3,
        force_z_variance_cost=False,
        w_force_z_variance=3.0,
        joint_limit_soft_cost=False,
        w_joint_limit_soft=2.0,
        joint_limit_soft_fraction=0.5,
        soft_constraints=False,
        w_soft_contact_vel=20.0,
        w_soft_friction=0.0,
        w_soft_land=0.0,
        track_width_cstr=True,
        w_track_width=6000,
        foot_sum_cstr=False,
        w_foot_sum=6000,
        foot_sum_z_offset=wheel_radius,
        wheel_axle_height_cstr=True,
        wheel_axle_height_min=0.9 * wheel_radius,
        wheel_axle_height_max=1.1 * wheel_radius,
        w_jointpos_base=w_jointpos,
        w_jointpos_T_base=w_jointpos_T,
        icr_arc_cstr=True,
        icr_arc_min_omega=1e-6,
        icr_arc_max_radius=100.0,
        w_hip_sum=1937.92,
        w_icr_axle=56.7218,
        hip_joint_names=[
            "FR_hip_joint",
            "RR_hip_joint",
            "FL_hip_joint",
            "RL_hip_joint",
        ],
    )

    horizon_T = int(round(1.0 / dt_mpc))
    mpc_conf = dict(
        support_force=-model_handler.getMass() * gravity[2],
        TOL=1e-4,
        mu_init=1e-8,
        max_iters=20,
        num_threads=8,
        swing_apex=0.15,
        T_fly=30,
        T_contact=10,
        timestep=dt_mpc,
        update_contact_ref=True,
    )

    contact_phase_quadru = {
        "FL_wheel": True,
        "FR_wheel": True,
        "RL_wheel": True,
        "RR_wheel": True,
    }
    return (
        model_handler,
        data_handler,
        problem_conf,
        gravity,
        force_size,
        horizon_T,
        mpc_conf,
        contact_phase_quadru,
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


def compute_icr_params(radius: float, yaw0: float) -> np.ndarray:
    if radius <= 1e-9:
        return np.array([0.0, 0.0, 0.0, 0.0])
    cx = -math.sin(yaw0) * radius
    cy = math.cos(yaw0) * radius
    return np.array([cx, cy, radius, 1.0])


def compute_point(
    radius: float,
    weight: float,
    vx_cmd: float,
    use_vector_glide_cost: bool,
    model_handler,
    problem_conf,
    gravity,
    force_size,
    horizon_T,
    mpc_conf,
    contact_phase_quadru,
    q_init,
    x_measured,
    yaw0: float,
    roll0: float,
    pitch0: float,
    joints0: np.ndarray,
    base_z: float,
    dt: float,
) -> tuple:
    if radius <= 1e-9:
        omega = 0.0
    else:
        omega = float(vx_cmd / radius)

    problem_conf["icr_arc_cstr"] = True
    problem_conf["icr_arc_min_omega"] = 1e-9
    problem_conf["use_vector_glide_cost"] = bool(use_vector_glide_cost)
    problem_conf["vector_glide_weight_base"] = float(weight)
    problem_conf["vector_glide_weight"] = float(weight)

    dynproblem = KinodynamicsOCP(problem_conf, model_handler)
    dynproblem.createProblem(model_handler.getReferenceState(), horizon_T, force_size, gravity[2], False)
    mpc = MPC(mpc_conf, dynproblem)
    mpc.generateCycleHorizon([contact_phase_quadru])

    v_world_cmd = np.array([
        vx_cmd * math.cos(yaw0),
        vx_cmd * math.sin(yaw0),
        0.0,
        0.0,
        0.0,
        omega,
    ])
    mpc.velocity_base = v_world_cmd

    icr_params = compute_icr_params(radius, yaw0)
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

    if not mpc.xs:
        return None
    x_final = np.asarray(mpc.xs[-1])
    x_ref = np.asarray(refs[-1])

    nq = q_init.shape[0]
    q_final = x_final[:nq]
    q_ref = x_ref[:nq]
    v_final = x_final[nq:]

    pos_err = q_final[:2] - q_ref[:2]
    rpy_final = rpy_from_quat_xyzw(q_final[3:7])
    rpy_ref = rpy_from_quat_xyzw(q_ref[3:7])
    yaw_err = rpy_final[2] - rpy_ref[2]
    yaw_err = (yaw_err + np.pi) % (2.0 * np.pi) - np.pi

    vel_err = v_final[:2] - v_world_cmd[:2]
    yaw_rate_err = float(v_final[5] - v_world_cmd[5])

    q_default = q_init
    joint_dev = q_final[7:] - q_default[7:]
    joint_dev_norm = float(np.linalg.norm(joint_dev))

    return {
        "radius": float(radius),
        "weight": float(weight),
        "vx": float(vx_cmd),
        "omega": float(omega),
        "use_vector_glide_cost": bool(use_vector_glide_cost),
        "pos_err_x": float(pos_err[0]),
        "pos_err_y": float(pos_err[1]),
        "yaw_err": float(yaw_err),
        "vel_err_x": float(vel_err[0]),
        "vel_err_y": float(vel_err[1]),
        "yaw_rate_err": float(yaw_rate_err),
        "joint_dev_norm": float(joint_dev_norm),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius-min", type=float, default=0.0)
    parser.add_argument("--radius-max", type=float, default=3.0)
    parser.add_argument("--radius-steps", type=int, default=31)
    parser.add_argument("--weight-min", type=float, default=0.0)
    parser.add_argument("--weight-max", type=float, default=50000.0)
    parser.add_argument("--weight-steps", type=int, default=51)
    parser.add_argument("--vx", type=float, default=2.0)
    parser.add_argument("--use-vector-glide", action="store_true", default=True)
    parser.add_argument("--no-use-vector-glide", action="store_false", dest="use_vector_glide")
    parser.add_argument("--disable-when-zero", action="store_true", default=True)
    parser.add_argument("--no-disable-when-zero", action="store_false", dest="disable_when_zero")
    parser.add_argument("--output-csv", type=str, default="examples/analysis_outputs/raw_error_grid.csv")
    parser.add_argument("--error-log", type=str, default="examples/analysis_outputs/raw_error_grid_errors.csv")
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

    (
        model_handler,
        _data_handler,
        problem_conf,
        gravity,
        force_size,
        horizon_T,
        mpc_conf,
        contact_phase_quadru,
    ) = build_mpc()
    model = model_handler.getModel()
    nq = model.nq

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

    dt = float(problem_conf.get("timestep", 0.01))

    radius_vals = np.linspace(args.radius_min, args.radius_max, max(args.radius_steps, 2))
    weight_vals = np.linspace(args.weight_min, args.weight_max, max(args.weight_steps, 2))

    rows = []
    err_rows = []
    for radius in radius_vals:
        for weight in weight_vals:
            use_vg = bool(args.use_vector_glide)
            if args.disable_when_zero and weight <= 0.0:
                use_vg = False
            try:
                result = compute_point(
                    radius,
                    weight,
                    float(args.vx),
                    use_vg,
                    model_handler,
                    problem_conf,
                    gravity,
                    force_size,
                    horizon_T,
                    mpc_conf,
                    contact_phase_quadru,
                    q_init,
                    x_measured,
                    yaw0,
                    roll0,
                    pitch0,
                    joints0,
                    base_xyz0[2],
                    dt,
                )
                if result is None:
                    continue
                rows.append(result)
            except Exception as exc:
                err_rows.append(
                    {
                        "radius": float(radius),
                        "weight": float(weight),
                        "use_vector_glide_cost": bool(use_vg),
                        "error": repr(exc),
                    }
                )
                rows.append(
                    {
                        "radius": float(radius),
                        "weight": float(weight),
                        "vx": float(args.vx),
                        "omega": float("nan"),
                        "use_vector_glide_cost": bool(use_vg),
                        "pos_err_x": float("nan"),
                        "pos_err_y": float("nan"),
                        "yaw_err": float("nan"),
                        "vel_err_x": float("nan"),
                        "vel_err_y": float("nan"),
                        "yaw_rate_err": float("nan"),
                        "joint_dev_norm": float("nan"),
                    }
                )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    if err_rows:
        err_path = Path(args.error_log)
        err_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(err_rows).to_csv(err_path, index=False)
    print(f"Saved raw data to {out_path}")


if __name__ == "__main__":
    main()
