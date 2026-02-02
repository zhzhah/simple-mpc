#!/usr/bin/env python3
"""Low-FPS visualization with MPC full-body trajectory prediction."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import csv
import re
import numpy as np
import pandas as pd

try:
    import pybullet as p
    import pybullet_data
except Exception as exc:  # pragma: no cover
    raise SystemExit("pybullet is required. Install with: pip install pybullet") from exc

try:
    import pinocchio as pin
    import example_robot_data as erd
    from example_robot_data.robots_loader import ROBOTS, RobotLoader
    from simple_mpc import (
        RobotModelHandler,
        RobotDataHandler,
        KinodynamicsOCP,
        MPC,
    )
except Exception as exc:  # pragma: no cover
    raise SystemExit("simple_mpc, pinocchio, example_robot_data are required") from exc

FIXED_URDF = Path("/home/yzc/MPCtest/traj/BQR3W_go2_description/urdf/BQR3W_go2_relative.urdf")
LOW_FPS_CSV = Path("/home/yzc/MPCtest/traj/processed_low.csv")

# ---- Visualization toggles (all visual helpers) ----
SHOW_TRAJECTORY = False
SHOW_TARGET_GHOST = False
SHOW_VELOCITY_ARROWS = False
SHOW_TARGET_BALL = False
SHOW_MPC_GHOSTS = True
SHOW_MPC_REF_GHOST = False
SHOW_MPC_TRAJECTORY = False
SHOW_FOOT_TRAJECTORY = False
SHOW_COM_TRAJECTORY = False
SHOW_MPC_STATE_LINE = True
SHOW_MPC_FOOT_LINES = True
SHOW_ICR_POINT = True
PRINT_MPC_DEBUG = True
SAVE_MPC_DEBUG = True
MPC_DEBUG_CSV = Path("/home/yzc/MPCtest/simple-mpc/mpc_debug.csv")
# ----------------------------------------------------

BASE_POS_OFFSET = (0.0, 0.0, 0.25)
MAX_TRAJ_POINTS = 2000

# Arrow style
VEL_SCALE = 0.5
ANG_VEL_SCALE = 0.5
ARROW_WIDTH = 100
ARROW_HEAD_RADIUS = 0.04

# Target ball
TARGET_BALL_RADIUS = 0.06
TARGET_INTEGRATION_T = 1.0  # seconds (also MPC horizon time)

# Instantaneous center of rotation (ICR)
ICR_RADIUS = 0.06

# MPC visualization
MPC_GHOST_STRIDE = 5
MPC_GHOST_ALPHA = 0.25

# Camera
FOLLOW_CAMERA = True
CAMERA_DISTANCE = 2.4
CAMERA_YAW = 110
CAMERA_PITCH = -11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize URDF with low-FPS processed data + MPC prediction.")
    parser.add_argument("--base-height", type=float, default=0.25, help="Base height (default: 0.25)")
    parser.add_argument("--hz", type=float, default=60.0, help="UI update rate (default: 60)")
    parser.add_argument(
        "--data-hz",
        type=float,
        default=100.0,
        help="Data frame rate for horizon indexing (default: 100)",
    )
    parser.add_argument("--video", action="store_true", help="Enable video output")
    parser.add_argument("--video-out", type=str, default="", help="Output video path (mp4)")
    parser.add_argument("--video-fps", type=float, default=30.0, help="Video FPS (default: 30)")
    parser.add_argument("--video-width", type=int, default=1280, help="Video width (default: 1280)")
    parser.add_argument("--video-height", type=int, default=720, help="Video height (default: 720)")
    parser.add_argument("--video-start", type=int, default=0, help="Start frame index (default: 0)")
    parser.add_argument("--video-end", type=int, default=-1, help="End frame index, inclusive (default: -1 = last)")
    parser.add_argument(
        "--video-range",
        type=str,
        default="",
        help="Video range as percentage (e.g. 30%-50%). Overrides video-start/end.",
    )
    return parser.parse_args()


def get_joint_name_to_index(body_id: int) -> dict[str, int]:
    name_to_index: dict[str, int] = {}
    for i in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, i)
        name = info[1].decode("utf-8")
        name_to_index[name] = i
    return name_to_index


def build_mpc():
    # Model loading (mirrors BQR3W_kinodynamics_go2w.py)
    URDF_SUBPATH = "BQR3W_go2_description/urdf/BQR3W_go2.urdf"
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
    nk = model_handler.getFeetNb()
    gravity = np.array([0, 0, -9.81])
    force_size = 3

    dt_mpc = 0.01
    w_basepos = [0, 0, 0, 0.0, 0.0, 0]
    w_jointpos = [10.1, 10.1, 10.1, 0.0]
    w_basevel = [20, 20, 10, 10, 10, 20]
    w_jointvel = [0.1, 0.1, 0.1, 0.0]

    # Terminal weights (separate from stage weights)
    w_basepos_T = [10000, 10000, 10000, 10000.0, 10000.0, 100000]
    w_jointpos_T = [1000.1, 1000.1, 1000.1, 0.0]
    w_basevel_T = [10, 10, 10, 10, 10, 10]
    w_jointvel_T = [0.1, 0.1, 0.1, 0.0]

    wheel_joints = [
        "FL_wheel_joint",
        "FR_wheel_joint",
        "RL_wheel_joint",
        "RR_wheel_joint",
    ]
    wheel_radius = 0.125
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
        vector_glide_weight=1000.0,
        vector_glide_weight_base=1000.0,
        vector_glide_disable_on_twist=True,
        vector_glide_min_omega=1e-6,
        min_wheel_distance_cstr=False,
        min_wheel_distance=1.2 * 2.0 * wheel_radius,
        min_wheel_distance_cost=False,
        w_min_wheel_distance=0.05,
        min_wheel_distance_cost_eps=1e-3,
        force_z_variance_cost=False,
        w_force_z_variance=10.0,
        joint_limit_soft_cost=False,
        w_joint_limit_soft=2.0,
        joint_limit_soft_fraction=0.5,
        soft_constraints=False,
        w_soft_contact_vel=20.0,
        w_soft_friction=0.0,
        w_soft_land=0.0,
        track_width_cstr=False,
        w_track_width=6000,
        foot_sum_cstr=False,
        w_foot_sum=6000,
        foot_sum_z_offset=wheel_radius,
        wheel_axle_height_cstr=False,
        wheel_axle_height_min=0.5 * wheel_radius,
        wheel_axle_height_max=1.0 * wheel_radius,
    )

    horizon_T = int(round(TARGET_INTEGRATION_T / dt_mpc))
    mpc_conf = dict(
        support_force=-model_handler.getMass() * gravity[2],
        TOL=1e-4,
        mu_init=1e-8,
        max_iters=5,
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


def spawn_ghost(urdf_path: Path, color_rgba, model_names: list[str]):
    ghost_id = p.loadURDF(str(urdf_path), [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], useFixedBase=True)
    bullet_joint_names = [
        p.getJointInfo(ghost_id, i)[1].decode() for i in range(p.getNumJoints(ghost_id))
    ]
    joint_indices = []
    for name in model_names[2:]:
        if name in bullet_joint_names:
            joint_indices.append(bullet_joint_names.index(name))
    for link_id in range(-1, p.getNumJoints(ghost_id)):
        p.setCollisionFilterGroupMask(ghost_id, link_id, 0, 0)
        p.changeVisualShape(ghost_id, link_id, rgbaColor=color_rgba)
    return ghost_id, joint_indices


def update_ghost(robot_id, joint_indices, q_ref_full):
    p.resetBasePositionAndOrientation(robot_id, q_ref_full[:3], q_ref_full[3:7])
    for i, j_idx in enumerate(joint_indices):
        p.resetJointState(robot_id, j_idx, q_ref_full[7 + i])


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    R = pin.rpy.rpyToMatrix(roll, pitch, yaw)
    return pin.Quaternion(R).coeffs()


def clamp_mpc_pitch(xs, nq, pitch_min, pitch_max):
    for x in xs:
        q = np.asarray(x[:nq]).copy()
        R = pin.Quaternion(q[3:7]).toRotationMatrix()
        rpy = pin.rpy.matrixToRpy(R)
        pitch_clamped = float(np.clip(rpy[1], pitch_min, pitch_max))
        if abs(pitch_clamped - rpy[1]) > 1e-12:
            R2 = pin.rpy.rpyToMatrix(float(rpy[0]), pitch_clamped, float(rpy[2]))
            q[3:7] = pin.Quaternion(R2).coeffs()
            x[:nq] = q


def update_vector_glide_weight(problem_conf, model_handler, x_measured, yaw_rate, wheel_link_names):
    if not problem_conf.get("vector_glide_disable_on_twist", False):
        return
    if abs(yaw_rate) <= 0.0:
        problem_conf["vector_glide_weight"] = problem_conf.get("vector_glide_weight_base", 0.0)
        return
    model = model_handler.getModel()
    nq = model.nq
    data = pin.Data(model)
    q0 = np.asarray(x_measured[:nq]).copy()
    v0 = np.asarray(x_measured[nq:]).copy()
    pin.forwardKinematics(model, data, q0, v0)
    pin.updateFramePlacements(model, data)
    base_fid = model_handler.getBaseFrameId()
    base_pose = data.oMf[base_fid]
    Rb = base_pose.rotation
    pb = base_pose.translation

    def foot_pos(name):
        fid = model_handler.getFootFrameId(model_handler.getFootNb(name))
        pw = data.oMf[fid].translation.copy()
        return Rb.T @ (pw - pb)

    fl = foot_pos("FL_wheel")
    fr = foot_pos("FR_wheel")
    rl = foot_pos("RL_wheel")
    rr = foot_pos("RR_wheel")

    front_avg = 0.5 * (fl + fr)
    rear_avg = 0.5 * (rl + rr)
    front_left_of_rear = front_avg[1] < rear_avg[1]
    front_right_of_rear = front_avg[1] > rear_avg[1]

    disable = (yaw_rate > 0.0 and front_left_of_rear) or (yaw_rate < 0.0 and front_right_of_rear)
    if disable:
        problem_conf["vector_glide_weight"] = 0.0
    else:
        problem_conf["vector_glide_weight"] = problem_conf.get("vector_glide_weight_base", 0.0)


def capture_frame(width: int, height: int) -> np.ndarray:
    cam = p.getDebugVisualizerCamera()
    view_matrix = cam[2]
    proj_matrix = cam[3]
    img = p.getCameraImage(
        width,
        height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    rgba = np.reshape(img[2], (height, width, 4))
    return rgba[:, :, :3]


def _quat_from_two_vecs(v_from: np.ndarray, v_to: np.ndarray):
    v_from = v_from / max(np.linalg.norm(v_from), 1e-9)
    v_to = v_to / max(np.linalg.norm(v_to), 1e-9)
    axis = np.cross(v_from, v_to)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        return [0, 0, 0, 1]
    axis = axis / axis_norm
    angle = np.arccos(np.clip(np.dot(v_from, v_to), -1.0, 1.0))
    s = np.sin(angle * 0.5)
    return [axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle * 0.5)]


def add_line_geom(p0, p1, color, radius=0.01):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v = p1 - p0
    length = float(np.linalg.norm(v))
    if length < 1e-6:
        return None
    mid = (p0 + p1) * 0.5
    quat = _quat_from_two_vecs(np.array([0.0, 0.0, 1.0]), v)
    vis = p.createVisualShape(p.GEOM_CAPSULE, radius=radius, length=length, rgbaColor=color)
    body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=mid.tolist(), baseOrientation=quat)
    return body


def debug_mpc_horizon(mpc, model_handler, problem_conf, wheel_link_names, base_height):
    model = model_handler.getModel()
    nq = model.nq
    nv = model.nv
    force_size = int(problem_conf.get("force_size", 3))
    gravity = np.asarray(problem_conf.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

    w_x = problem_conf["w_x"]
    w_u = problem_conf["w_u"]

    base_frame_id = model_handler.getBaseFrameId()
    data_ref = pin.Data(model)
    ref_state = model_handler.getReferenceState()
    pin.forwardKinematics(model, data_ref, ref_state[:nq], ref_state[nq:])
    pin.updateFramePlacements(model, data_ref)

    def quad_cost(vec, W, idxs):
        if idxs.size == 0:
            return 0.0
        sub = vec[idxs]
        Wsub = W[np.ix_(idxs, idxs)]
        return float(sub.T @ Wsub @ sub)

    idx_base_pos = np.arange(0, 3)
    idx_base_ori = np.arange(3, 6)
    idx_joint_pos = np.arange(6, nv)
    idx_base_linvel = np.arange(nv + 0, nv + 3)
    idx_base_angvel = np.arange(nv + 3, nv + 6)
    idx_joint_vel = np.arange(nv + 6, 2 * nv)

    rows = []
    for t in range(len(mpc.xs) - 1):
        x0 = np.asarray(mpc.xs[t])
        u0 = np.asarray(mpc.us[t]) if t < len(mpc.us) else None

        try:
            x_ref = np.asarray(mpc.ocp_handler.getReferenceState(t))
            u_ref = np.asarray(mpc.ocp_handler.getReferenceControl(t))
        except Exception:
            x_ref = x0
            u_ref = u0 if u0 is not None else np.zeros_like(mpc.us[0])

        q0 = x0[:nq]
        v0 = x0[nq:]
        q_ref = x_ref[:nq]
        v_ref = x_ref[nq:]
        dx_q = pin.difference(model, q_ref, q0)
        dx_v = v0 - v_ref
        dx = np.concatenate([dx_q, dx_v])
        du = u0 - u_ref if u0 is not None else np.zeros_like(u_ref)

        state_cost = float(dx.T @ w_x @ dx)
        control_cost = float(du.T @ w_u @ du)

        state_cost_base_pos = quad_cost(dx, w_x, idx_base_pos)
        state_cost_base_ori = quad_cost(dx, w_x, idx_base_ori)
        state_cost_joint_pos = quad_cost(dx, w_x, idx_joint_pos)
        state_cost_base_linvel = quad_cost(dx, w_x, idx_base_linvel)
        state_cost_base_angvel = quad_cost(dx, w_x, idx_base_angvel)
        state_cost_joint_vel = quad_cost(dx, w_x, idx_joint_vel)

        data = pin.Data(model)
        pin.forwardKinematics(model, data, x0[:nq], x0[nq:])
        pin.updateFramePlacements(model, data)
        pin.computeCentroidalMomentum(model, data, x0[:nq], x0[nq:])
        com = pin.centerOfMass(model, data, x0[:nq], x0[nq:])

        foot_cost = 0.0
        for name in wheel_link_names:
            frame_id = mpc.getModelHandler().getFootFrameId(
                mpc.getModelHandler().getFootNb(name)
            )
            foot_pos = data.oMf[frame_id].translation
            foot_ref = mpc.getReferencePose(t, name).translation
            err = foot_pos - foot_ref
            foot_cost += float(err.T @ np.diag(problem_conf["w_frame"]) @ err)

        centroidal_cost = 0.0
        centroidal_cost_lin = 0.0
        centroidal_cost_ang = 0.0
        if problem_conf.get("w_cent", None) is not None:
            h = data.hg.vector
            centroidal_cost = float(h.T @ problem_conf["w_cent"] @ h)
            centroidal_cost_lin = float(
                h[:3].T
                @ np.diag(np.diag(problem_conf["w_cent"])[:3])
                @ h[:3]
            )
            centroidal_cost_ang = float(
                h[3:].T
                @ np.diag(np.diag(problem_conf["w_cent"])[3:])
                @ h[3:]
            )

        centroidal_derivative_cost = 0.0
        centroidal_derivative_res = None
        if problem_conf.get("w_centder", None) is not None:
            sum_f = np.zeros(3)
            sum_tau = np.zeros(3)
            for i, name in enumerate(wheel_link_names):
                frame_id = mpc.getModelHandler().getFootFrameId(
                    mpc.getModelHandler().getFootNb(name)
                )
                r_contact = data.oMf[frame_id].translation
                fx, fy, fz = u0[i * force_size : i * force_size + 3]
                f_i = np.array([fx, fy, fz])
                sum_f += f_i
                sum_tau += np.cross(r_contact - com, f_i)
            sum_f += model_handler.getMass() * gravity
            centroidal_derivative_res = np.concatenate([sum_f, sum_tau])
            centroidal_derivative_cost = float(
                centroidal_derivative_res.T
                @ problem_conf["w_centder"]
                @ centroidal_derivative_res
            )

        soft_contact_vel_cost = 0.0
        soft_contact_vel_res = []
        for name in wheel_link_names:
            frame_id = mpc.getModelHandler().getFootFrameId(
                mpc.getModelHandler().getFootNb(name)
            )
            v_local = pin.getFrameVelocity(model, data, frame_id, pin.LOCAL).linear
            if problem_conf.get("enable_lateral_no_slip", False):
                foot_pose = data.oMf[frame_id]
                a_y = foot_pose.rotation[:, 1]
                if np.linalg.norm(a_y) < problem_conf["lateral_no_slip_min_axis_norm"]:
                    a_y = np.array([0.0, 1.0, 0.0])
                else:
                    a_y = a_y / np.linalg.norm(a_y)
                g_z = np.array([0.0, 0.0, 1.0])
                c_x = np.cross(a_y, g_z)
                if np.linalg.norm(c_x) < problem_conf["lateral_no_slip_min_cross_norm"]:
                    a_x = foot_pose.rotation[:, 0]
                    c_x = a_x - g_z * np.dot(a_x, g_z)
                if np.linalg.norm(c_x) < problem_conf["lateral_no_slip_min_cross_norm"]:
                    c_x = np.array([1.0, 0.0, 0.0])
                c_x = c_x / np.linalg.norm(c_x)
                c_y = np.cross(g_z, c_x)
                if np.linalg.norm(c_y) < problem_conf["lateral_no_slip_min_cross_norm"]:
                    c_y = np.array([0.0, 1.0, 0.0])
                else:
                    c_y = c_y / np.linalg.norm(c_y)
                v_world = pin.getFrameVelocity(
                    model, data, frame_id, pin.LOCAL_WORLD_ALIGNED
                ).linear
                res_lat = np.array([float(np.dot(v_world, c_y))])
                res = np.concatenate([res_lat, np.array([v_local[2]])])
            elif problem_conf["nonholonomic_rolling"]:
                res = v_local[1:3]
            else:
                res = v_local[:3]
            soft_contact_vel_res.append(res.copy())
            soft_contact_vel_cost += float(
                problem_conf["w_soft_contact_vel"] * (res.T @ res)
            )

        soft_friction_cost = 0.0
        soft_friction_res = []
        for i in range(4):
            fx, fy, fz = u0[i * force_size : i * force_size + 3]
            r1 = np.sqrt(fx * fx + fy * fy) - problem_conf["mu"] * fz
            r2 = -fz
            res = np.array([max(0.0, r1), max(0.0, r2)])
            soft_friction_res.append(res.copy())
            soft_friction_cost += float(
                problem_conf["w_soft_friction"] * (res.T @ res)
            )

        foot_sum_res = None
        foot_sum_cost = 0.0
        if problem_conf.get("foot_sum_cstr", False):
            base_pose = data.oMf[base_frame_id]
            Rb = base_pose.rotation
            pb = base_pose.translation
            sum_base = np.zeros(3)
            for name in wheel_link_names:
                frame_id = mpc.getModelHandler().getFootFrameId(
                    mpc.getModelHandler().getFootNb(name)
                )
                pw = data.oMf[frame_id].translation
                sum_base += Rb.T @ (pw - pb)
            mean_base = sum_base / 4.0
            target_sum = np.array(
                [
                    0.0,
                    0.0,
                    base_height - problem_conf.get("foot_sum_z_offset", 0.0),
                ]
            )
            foot_sum_res = -mean_base - target_sum
            foot_sum_cost = float(
                problem_conf.get("w_foot_sum", 0.0) * (foot_sum_res.T @ foot_sum_res)
            )

        track_width_res = None
        track_width_cost = 0.0
        if problem_conf.get("track_width_cstr", False) and len(wheel_link_names) >= 4:
            base_pose = data.oMf[base_frame_id]
            Rb = base_pose.rotation
            pb = base_pose.translation

            def y_in_base(fid):
                pw = data.oMf[fid].translation
                pb_rel = Rb.T @ (pw - pb)
                return pb_rel[1]

            fl_id = mpc.getModelHandler().getFootFrameId(
                mpc.getModelHandler().getFootNb(wheel_link_names[0])
            )
            fr_id = mpc.getModelHandler().getFootFrameId(
                mpc.getModelHandler().getFootNb(wheel_link_names[1])
            )
            rl_id = mpc.getModelHandler().getFootFrameId(
                mpc.getModelHandler().getFootNb(wheel_link_names[2])
            )
            rr_id = mpc.getModelHandler().getFootFrameId(
                mpc.getModelHandler().getFootNb(wheel_link_names[3])
            )
            y_fl = y_in_base(fl_id)
            y_fr = y_in_base(fr_id)
            y_rl = y_in_base(rl_id)
            y_rr = y_in_base(rr_id)

            base_ref = data_ref.oMf[base_frame_id]
            Rb_ref = base_ref.rotation
            pb_ref = base_ref.translation

            def y_in_base_ref(fid):
                pw = data_ref.oMf[fid].translation
                pb_rel = Rb_ref.T @ (pw - pb_ref)
                return pb_rel[1]

            y_fl_ref = y_in_base_ref(fl_id)
            y_fr_ref = y_in_base_ref(fr_id)
            y_rl_ref = y_in_base_ref(rl_id)
            y_rr_ref = y_in_base_ref(rr_id)
            target_front = y_fl_ref - y_fr_ref
            target_rear = y_rl_ref - y_rr_ref

            track_width_res = np.array(
                [(y_fl - y_fr) - target_front, (y_rl - y_rr) - target_rear]
            )
            track_width_cost = float(
                problem_conf.get("w_track_width", 0.0)
                * (track_width_res.T @ track_width_res)
            )

        min_wheel_distance_cost = 0.0
        min_wheel_distance_res = None
        if problem_conf.get("min_wheel_distance_cost", False) and len(wheel_link_names) >= 2:
            min_wheel_distance_res = []
            min_dist = float(problem_conf.get("min_wheel_distance", 0.0))
            eps = float(problem_conf.get("min_wheel_distance_cost_eps", 1e-3))
            pairs = [("FL_wheel", "RL_wheel"), ("FR_wheel", "RR_wheel")]
            ref_dist = []
            for a, b in pairs:
                fa = mpc.getModelHandler().getFootFrameId(mpc.getModelHandler().getFootNb(a))
                fb = mpc.getModelHandler().getFootFrameId(mpc.getModelHandler().getFootNb(b))
                pa = data_ref.oMf[fa].translation.copy()
                pb = data_ref.oMf[fb].translation.copy()
                pa[2] = 0.0
                pb[2] = 0.0
                ref_dist.append(np.linalg.norm(pa - pb))
            for idx, (a, b) in enumerate(pairs):
                fa = mpc.getModelHandler().getFootFrameId(mpc.getModelHandler().getFootNb(a))
                fb = mpc.getModelHandler().getFootFrameId(mpc.getModelHandler().getFootNb(b))
                pa = data.oMf[fa].translation.copy()
                pb = data.oMf[fb].translation.copy()
                pa[2] = 0.0
                pb[2] = 0.0
                d = np.linalg.norm(pa - pb)
                denom = max(d - min_dist + eps, eps)
                denom0 = max(ref_dist[idx] - min_dist + eps, eps)
                res = (1.0 / denom) - (1.0 / denom0)
                min_wheel_distance_res.append(res)
                min_wheel_distance_cost += float(
                    problem_conf.get("w_min_wheel_distance", 0.0) * (res * res)
                )

        vector_glide_cost = 0.0
        vector_glide_res = []
        if problem_conf.get("use_vector_glide_cost", False):
            v_des = float(mpc.velocity_base[0])
            omega_des = float(mpc.velocity_base[5])
            if abs(omega_des) >= problem_conf.get("vector_glide_min_omega", 1e-6):
                base_pose = data.oMf[base_frame_id]
                r_base = base_pose.translation
                R_base = base_pose.rotation
                icr_local = np.array([0.0, v_des / omega_des, 0.0])
                r_O_star = r_base + R_base @ icr_local
                for name in wheel_link_names:
                    frame_id = mpc.getModelHandler().getFootFrameId(
                        mpc.getModelHandler().getFootNb(name)
                    )
                    foot_pose = data.oMf[frame_id]
                    a_y = foot_pose.rotation[:, 1]
                    if np.linalg.norm(a_y) < problem_conf["lateral_no_slip_min_axis_norm"]:
                        a_y = np.array([0.0, 1.0, 0.0])
                    else:
                        a_y = a_y / np.linalg.norm(a_y)
                    g_z = np.array([0.0, 0.0, 1.0])
                    c_x = np.cross(a_y, g_z)
                    if np.linalg.norm(c_x) < problem_conf["lateral_no_slip_min_cross_norm"]:
                        a_x = foot_pose.rotation[:, 0]
                        c_x = a_x - g_z * np.dot(a_x, g_z)
                    if np.linalg.norm(c_x) < problem_conf["lateral_no_slip_min_cross_norm"]:
                        c_x = np.array([1.0, 0.0, 0.0])
                    c_x = c_x / np.linalg.norm(c_x)
                    r_contact = foot_pose.translation
                    rho = r_O_star - r_contact
                    rho_norm = np.linalg.norm(rho)
                    if rho_norm > 1e-9:
                        e_i = float(np.dot(rho, c_x))
                        res = e_i / rho_norm
                        vector_glide_res.append(res)
                        vector_glide_cost += float(
                            problem_conf.get("vector_glide_weight", 0.0) * (res * res)
                        )

        joint_limit_soft_cost = 0.0
        joint_limit_soft_res = None
        if problem_conf.get("joint_limit_soft_cost", False):
            qmin = problem_conf["qmin"]
            qmax = problem_conf["qmax"]
            frac = float(problem_conf.get("joint_limit_soft_fraction", 0.5))
            res = np.zeros_like(qmin)
            for i in range(qmin.shape[0]):
                q_i = q0[7 + i]
                qmin_i = qmin[i]
                qmax_i = qmax[i]
                rng = qmax_i - qmin_i
                if rng <= 0.0:
                    continue
                lower = qmin_i + frac * rng
                upper = qmax_i - frac * rng
                if q_i < lower:
                    res[i] = lower - q_i
                elif q_i > upper:
                    res[i] = q_i - upper
            joint_limit_soft_res = res
            joint_limit_soft_cost = float(
                problem_conf.get("w_joint_limit_soft", 0.0) * (res.T @ res)
            )

        contact_states = []
        try:
            contact_states = mpc.ocp_handler.getContactState(t)
        except Exception:
            contact_states = []

        row = {
            "t": t,
            "state_cost": state_cost,
            "control_cost": control_cost,
            "foot_cost": foot_cost,
            "state_cost_base_pos": state_cost_base_pos,
            "state_cost_base_ori": state_cost_base_ori,
            "state_cost_joint_pos": state_cost_joint_pos,
            "state_cost_base_linvel": state_cost_base_linvel,
            "state_cost_base_angvel": state_cost_base_angvel,
            "state_cost_joint_vel": state_cost_joint_vel,
            "centroidal_cost": centroidal_cost,
            "centroidal_cost_lin": centroidal_cost_lin,
            "centroidal_cost_ang": centroidal_cost_ang,
            "centroidal_derivative_cost": centroidal_derivative_cost,
            "soft_contact_vel_cost": soft_contact_vel_cost,
            "soft_friction_cost": soft_friction_cost,
            "track_width_cost": track_width_cost,
            "foot_sum_cost": foot_sum_cost,
            "min_wheel_distance_cost": min_wheel_distance_cost,
            "vector_glide_cost": vector_glide_cost,
            "joint_limit_soft_cost": joint_limit_soft_cost,
        }
        if centroidal_derivative_res is not None:
            row["centroidal_derivative_res"] = centroidal_derivative_res
        if soft_contact_vel_res:
            row["soft_contact_vel_res"] = soft_contact_vel_res
        if soft_friction_res:
            row["soft_friction_res"] = soft_friction_res
        if track_width_res is not None:
            row["track_width_res"] = track_width_res
        if foot_sum_res is not None:
            row["foot_sum_res"] = foot_sum_res
        if min_wheel_distance_res is not None:
            row["min_wheel_distance_res"] = min_wheel_distance_res
        if vector_glide_res:
            row["vector_glide_res"] = vector_glide_res
        if joint_limit_soft_res is not None:
            row["joint_limit_soft_res"] = joint_limit_soft_res
        if contact_states:
            row["contact_states"] = contact_states
        rows.append(row)

        print(f"[MPC] t={t}")
        print(f"  state_cost={state_cost:.6g} control_cost={control_cost:.6g} foot_cost={foot_cost:.6g}")
        if t == len(mpc.xs) - 2:
            try:
                x_term_ref = np.asarray(mpc.ocp_handler.getTerminalReferenceState())
                print(f"  terminal q_ref={x_term_ref[:nq]}")
            except Exception:
                print(f"  terminal q_ref={q_ref}")
        print(
            "  state_cost parts: base_pos={:.3g} base_ori={:.3g} joint_pos={:.3g} base_linvel={:.3g} base_angvel={:.3g} joint_vel={:.3g}".format(
                state_cost_base_pos,
                state_cost_base_ori,
                state_cost_joint_pos,
                state_cost_base_linvel,
                state_cost_base_angvel,
                state_cost_joint_vel,
            )
        )
        print(
            "  centroidal_cost={:.6g} (lin={:.3g}, ang={:.3g}) cent_der_cost={:.6g}".format(
                centroidal_cost,
                centroidal_cost_lin,
                centroidal_cost_ang,
                centroidal_derivative_cost,
            )
        )
        print(
            "  soft_contact_vel_cost={:.6g} soft_friction_cost={:.6g} track_width_cost={:.6g} foot_sum_cost={:.6g}".format(
                soft_contact_vel_cost,
                soft_friction_cost,
                track_width_cost,
                foot_sum_cost,
            )
        )
        print(
            "  min_wheel_distance_cost={:.6g} vector_glide_cost={:.6g} joint_limit_soft_cost={:.6g}".format(
                min_wheel_distance_cost,
                vector_glide_cost,
                joint_limit_soft_cost,
            )
        )
        if contact_states:
            print(f"  contact_states={contact_states}")
        if centroidal_derivative_res is not None:
            print(f"  centroidal_derivative_res={centroidal_derivative_res}")
        if soft_contact_vel_res:
            print(f"  soft_contact_vel_res={soft_contact_vel_res}")
        if soft_friction_res:
            print(f"  soft_friction_res={soft_friction_res}")
        if track_width_res is not None:
            print(f"  track_width_res={track_width_res}")
        if foot_sum_res is not None:
            print(f"  foot_sum_res={foot_sum_res}")
        if min_wheel_distance_res is not None:
            print(f"  min_wheel_distance_res={min_wheel_distance_res}")
        if vector_glide_res:
            print(f"  vector_glide_res={vector_glide_res}")
        if joint_limit_soft_res is not None:
            print(f"  joint_limit_soft_res={joint_limit_soft_res}")

        # Coordinate-frame consistency diagnostics (only t=0)
        if t == 0:
            sum_f_A = np.zeros(3)
            sum_tau_A = np.zeros(3)
            sum_f_B = np.zeros(3)
            sum_tau_B = np.zeros(3)
            print("  [FrameDiag] force interpretation check @ t=0")
            for i, name in enumerate(wheel_link_names):
                frame_id = mpc.getModelHandler().getFootFrameId(
                    mpc.getModelHandler().getFootNb(name)
                )
                R = data.oMf[frame_id].rotation
                r_contact = data.oMf[frame_id].translation
                f_raw = u0[i * force_size : i * force_size + 3]
                f_world_A = f_raw
                f_world_B = R @ f_raw
                sum_f_A += f_world_A
                sum_tau_A += np.cross(r_contact - com, f_world_A)
                sum_f_B += f_world_B
                sum_tau_B += np.cross(r_contact - com, f_world_B)
                print(f"    wheel={name}")
                print(f"      R=\n{R}")
                print(f"      f_raw={f_raw}")
                print(f"      f_world_A={f_world_A}")
                print(f"      f_world_B={f_world_B}")
            sum_f_A += model_handler.getMass() * gravity
            sum_f_B += model_handler.getMass() * gravity
            res_A = np.concatenate([sum_f_A, sum_tau_A])
            res_B = np.concatenate([sum_f_B, sum_tau_B])
            print(f"  [FrameDiag] A: sum_f={sum_f_A}, sum_tau={sum_tau_A}, res={res_A}, |res|={np.linalg.norm(res_A):.6g}")
            print(f"  [FrameDiag] B: sum_f={sum_f_B}, sum_tau={sum_tau_B}, res={res_B}, |res|={np.linalg.norm(res_B):.6g}")
    return rows


def main() -> None:
    args = parse_args()
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
    tgt_world_xyz = df[["tgt_world_x", "tgt_world_y", "tgt_world_z"]].to_numpy() if SHOW_TARGET_GHOST else None
    yaw = df["yaw"].to_numpy()
    roll = df["CoMAngle_0"].to_numpy() if "CoMAngle_0" in df.columns else np.zeros_like(yaw)
    pitch = df["CoMAngle_1"].to_numpy() if "CoMAngle_1" in df.columns else np.zeros_like(yaw)
    pitch_mean = float(np.mean(pitch)) if pitch.size else 0.0
    pitch_limit = 5.0 * np.pi / 180.0
    pitch_min = pitch_mean - pitch_limit
    pitch_max = pitch_mean + pitch_limit

    vel_world = df[["vel_world_x", "vel_world_y"]].to_numpy() if SHOW_VELOCITY_ARROWS else None
    vel_d_world = df[["vel_d_world_x", "vel_d_world_y"]].to_numpy() if SHOW_VELOCITY_ARROWS else None
    wz = df["wz"].to_numpy() if SHOW_VELOCITY_ARROWS else None
    wz_d = df["wz_d"].to_numpy() if SHOW_VELOCITY_ARROWS else None

    num_frames = q.shape[0]

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setAdditionalSearchPath(str(FIXED_URDF.parent))
    meshes_dir = FIXED_URDF.parent.parent / "meshes"
    if meshes_dir.exists():
        p.setAdditionalSearchPath(str(meshes_dir))
    p.resetDebugVisualizerCamera(
        cameraDistance=CAMERA_DISTANCE,
        cameraYaw=CAMERA_YAW,
        cameraPitch=CAMERA_PITCH,
        cameraTargetPosition=[0, 0, 0.2],
    )
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

    # Floor + grid
    floor_half_extents = [50, 50, 0.0005]
    floor_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=floor_half_extents, rgbaColor=[0.2, 0.2, 0.2, 1])
    floor_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=floor_half_extents)
    p.createMultiBody(0, floor_col, floor_vis, [0.0, 0.0, 0.005])
    grid_size = 100
    grid_step = 0.5
    grid_z = 0.006
    for i in range(-grid_size, grid_size + 1):
        a = i * grid_step
        p.addUserDebugLine(
            [-grid_size * grid_step, a, grid_z],
            [grid_size * grid_step, a, grid_z],
            lineColorRGB=[0.35, 0.35, 0.35],
            lineWidth=1,
        )
        p.addUserDebugLine(
            [a, -grid_size * grid_step, grid_z],
            [a, grid_size * grid_step, grid_z],
            lineColorRGB=[0.35, 0.35, 0.35],
            lineWidth=1,
        )

    # Screen
    screen_half_extents = [0.0005, 32.5, 32.5]
    screen_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=screen_half_extents, rgbaColor=[0.25, 0.25, 0.25, 1])
    screen_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=screen_half_extents)
    p.createMultiBody(0, screen_col, screen_vis, [-1.0, 0.0, 1.0])
    # Screen on right side (3m to the right of spawn), facing the robot
    right_screen_quat = p.getQuaternionFromEuler([0.0, 0.0, -np.pi / 2.0])
    p.createMultiBody(0, screen_col, screen_vis, [0.0, -3.0, 1.0], right_screen_quat)

    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

    cam_yaw_slider = p.addUserDebugParameter("cam_yaw", -180, 180, CAMERA_YAW)
    cam_pitch_slider = p.addUserDebugParameter("cam_pitch", -89, 89, CAMERA_PITCH)
    cam_dist_slider = p.addUserDebugParameter("cam_dist", 0.2, 6.0, CAMERA_DISTANCE)

    p.setGravity(0, 0, 0)

    body_id = p.loadURDF(str(FIXED_URDF), [0, 0, args.base_height], useFixedBase=False, flags=p.URDF_IGNORE_COLLISION_SHAPES)

    # Joint order fixed
    joint_order = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_wheel_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_wheel_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_wheel_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_wheel_joint",
    ]
    name_to_index = get_joint_name_to_index(body_id)
    joint_indices = [name_to_index[n] for n in joint_order]

    frame_slider = p.addUserDebugParameter("frame", 0, max(num_frames - 1, 0), 0)

    # Target ghost / ball
    target_ghost_id = None
    if SHOW_TARGET_GHOST and tgt_world_xyz is not None:
        ghost_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.08, rgbaColor=[0.2, 0.8, 1.0, 0.35])
        target_ghost_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=ghost_vis)
    target_ball_id = None
    if SHOW_TARGET_BALL:
        ball_vis = p.createVisualShape(p.GEOM_SPHERE, radius=TARGET_BALL_RADIUS, rgbaColor=[1.0, 0.2, 0.2, 1])
        target_ball_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=ball_vis)
    icr_id = None
    if SHOW_ICR_POINT:
        icr_vis = p.createVisualShape(p.GEOM_SPHERE, radius=ICR_RADIUS, rgbaColor=[1.0, 0.2, 0.2, 1])
        icr_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=icr_vis)

    # Arrow heads (spheres)
    vel_head_id = vel_d_head_id = ang_head_id = ang_d_head_id = -1
    if SHOW_VELOCITY_ARROWS:
        vel_head_vis = p.createVisualShape(p.GEOM_SPHERE, radius=ARROW_HEAD_RADIUS, rgbaColor=[0.2, 0.9, 0.2, 1])
        vel_d_head_vis = p.createVisualShape(p.GEOM_SPHERE, radius=ARROW_HEAD_RADIUS, rgbaColor=[0.2, 0.6, 1.0, 1])
        ang_head_vis = p.createVisualShape(p.GEOM_SPHERE, radius=ARROW_HEAD_RADIUS, rgbaColor=[1.0, 0.6, 0.2, 1])
        ang_d_head_vis = p.createVisualShape(p.GEOM_SPHERE, radius=ARROW_HEAD_RADIUS, rgbaColor=[1.0, 0.2, 0.8, 1])
        vel_head_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vel_head_vis)
        vel_d_head_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vel_d_head_vis)
        ang_head_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=ang_head_vis)
        ang_d_head_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=ang_d_head_vis)

    line_ids = []
    vel_line_id = vel_d_line_id = ang_line_id = ang_d_line_id = -1

    # MPC config/model
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
    nv = model.nv
    model_names = list(model.names)
    wheel_link_names = ["FL_wheel", "FR_wheel", "RL_wheel", "RR_wheel"]
    # Fixed foot height (world) based on initial pose
    q_init = model_handler.getReferenceState()[:nq].copy()
    q_init[0] = world_xyz[0, 0]
    q_init[1] = world_xyz[0, 1]
    q_init[2] = world_xyz[0, 2]
    q_init[3:7] = quat_from_rpy(float(roll[0]), float(pitch[0]), float(yaw[0]))
    for joint_name, qv in zip(joint_order, q[0]):
        joint_id = model.getJointId(joint_name)
        idx_q = model.joints[joint_id].idx_q
        q_init[idx_q : idx_q + model.joints[joint_id].nq] = qv
    data_init = pin.Data(model)
    pin.forwardKinematics(model, data_init, q_init, np.zeros(nv))
    pin.updateFramePlacements(model, data_init)
    foot_axis_z_world = {}
    for name in wheel_link_names:
        fid = model_handler.getFootFrameId(model_handler.getFootNb(name))
        foot_axis_z_world[name] = float(data_init.oMf[fid].translation[2])
    foot_traj_lines = {name: [] for name in wheel_link_names}
    foot_traj_prev = {name: None for name in wheel_link_names}
    com_traj_lines = []
    com_traj_prev = None
    mpc_state_line_ids = []
    mpc_foot_line_ids = {name: [] for name in wheel_link_names}
    mpc_state_line_bodies = []
    mpc_foot_line_bodies = {name: [] for name in wheel_link_names}

    debug_csv_file = None
    debug_csv_writer = None
    if SAVE_MPC_DEBUG:
        debug_csv_file = MPC_DEBUG_CSV.open("w", newline="")
        debug_csv_writer = csv.writer(debug_csv_file)
        header = [
            "frame",
            "t",
            "base_x",
            "base_y",
            "base_z",
            "v_world_cmd_x",
            "v_world_cmd_y",
            "v_world_cmd_yaw",
            "state_cost",
            "control_cost",
            "foot_cost",
            "state_cost_base_pos",
            "state_cost_base_ori",
            "state_cost_joint_pos",
            "state_cost_base_linvel",
            "state_cost_base_angvel",
            "state_cost_joint_vel",
            "centroidal_cost",
            "centroidal_cost_lin",
            "centroidal_cost_ang",
            "centroidal_derivative_cost",
            "soft_contact_vel_cost",
            "soft_friction_cost",
            "track_width_cost",
            "foot_sum_cost",
            "min_wheel_distance_cost",
            "vector_glide_cost",
            "joint_limit_soft_cost",
            "contact_states",
            "centroidal_derivative_res",
            "soft_contact_vel_res",
            "soft_friction_res",
            "track_width_res",
            "foot_sum_res",
            "min_wheel_distance_res",
            "vector_glide_res",
            "joint_limit_soft_res",
        ]
        debug_csv_writer.writerow(header)

    def create_mpc_instance():
        dynproblem = KinodynamicsOCP(problem_conf, model_handler)
        dynproblem.createProblem(
            model_handler.getReferenceState(), horizon_T, force_size, gravity[2], False
        )
        mpc = MPC(mpc_conf, dynproblem)
        mpc.generateCycleHorizon([contact_phase_quadru])
        return mpc

    def reset_mpc_initial_guess(mpc, x_measured):
        # Force the whole horizon initial guess to the current measured state.
        try:
            for i in range(len(mpc.xs)):
                mpc.xs[i][:] = x_measured
        except Exception:
            pass
        try:
            u_ref = np.asarray(mpc.ocp_handler.getReferenceControl(0)).copy()
            for i in range(len(mpc.us)):
                mpc.us[i][:] = u_ref
        except Exception:
            pass

    # Single ghost for MPC terminal state
    mpc_terminal_ghost = None
    if SHOW_MPC_GHOSTS:
        ghost_id, ghost_joints = spawn_ghost(
            FIXED_URDF, [0.9, 0.8, 0.2, MPC_GHOST_ALPHA], model_names
        )
        mpc_terminal_ghost = (ghost_id, ghost_joints)
    # Single ghost for MPC terminal reference
    mpc_ref_ghost = None
    if SHOW_MPC_REF_GHOST:
        ghost_id, ghost_joints = spawn_ghost(
            FIXED_URDF, [0.2, 0.9, 0.2, MPC_GHOST_ALPHA], model_names
        )
        mpc_ref_ghost = (ghost_id, ghost_joints)

    mpc_line_ids = []

    # Prepare q/v buffers
    q_meas = model_handler.getReferenceState()[:nq].copy()
    v_meas = np.zeros(nv)

    def compute_mpc_inputs(frame_idx: int):
        base_xyz_local = world_xyz[frame_idx]
        vx_cmd_local = float(df["CoMVelo_d_0"].to_numpy()[frame_idx])
        wz_cmd_local = float(df["rpy_w_d_2"].to_numpy()[frame_idx])

        if abs(wz_cmd_local) < 1e-6:
            dx_local = vx_cmd_local * TARGET_INTEGRATION_T
            dy_local = 0.0
        else:
            R_local = vx_cmd_local / wz_cmd_local
            dtheta_local = wz_cmd_local * TARGET_INTEGRATION_T
            dx_local = R_local * np.sin(dtheta_local)
            dy_local = R_local * (1.0 - np.cos(dtheta_local))
        cy_local = np.cos(yaw[frame_idx])
        sy_local = np.sin(yaw[frame_idx])
        delta_world_local = np.array([cy_local * dx_local - sy_local * dy_local, sy_local * dx_local + cy_local * dy_local])
        ball_xy_local = base_xyz_local[:2] + delta_world_local

        q_meas_local = model_handler.getReferenceState()[:nq].copy()
        # Single-step MPC: anchor base at origin, keep measured orientation/joints
        pitch_use = float(np.clip(pitch[frame_idx], pitch_min, pitch_max))
        q_meas_local[0] = 0.0
        q_meas_local[1] = 0.0
        q_meas_local[2] = base_xyz_local[2]
        q_meas_local[3:7] = quat_from_rpy(float(roll[frame_idx]), pitch_use, float(yaw[frame_idx]))
        for joint_name, qv in zip(joint_order, q[frame_idx]):
            joint_id = model.getJointId(joint_name)
            idx_q = model.joints[joint_id].idx_q
            q_meas_local[idx_q : idx_q + model.joints[joint_id].nq] = qv

        v_meas_local = np.zeros(nv)
        if "vel_world_x" in df.columns and "vel_world_y" in df.columns:
            v_meas_local[0] = float(df["vel_world_x"].to_numpy()[frame_idx])
            v_meas_local[1] = float(df["vel_world_y"].to_numpy()[frame_idx])
            v_meas_local[2] = float(df["CoMVelo_2"].to_numpy()[frame_idx]) if "CoMVelo_2" in df.columns else 0.0
        if "CoMAngleVelo_W_0" in df.columns:
            v_meas_local[3] = float(df["CoMAngleVelo_W_0"].to_numpy()[frame_idx])
            v_meas_local[4] = float(df["CoMAngleVelo_W_1"].to_numpy()[frame_idx])
            v_meas_local[5] = float(df["CoMAngleVelo_W_2"].to_numpy()[frame_idx])

        x_measured_local = np.concatenate([q_meas_local, v_meas_local])

        v_world_cmd_local = np.zeros(6)
        v_world_cmd_local[0] = vx_cmd_local * np.cos(yaw[frame_idx])
        v_world_cmd_local[1] = vx_cmd_local * np.sin(yaw[frame_idx])
        v_world_cmd_local[5] = wz_cmd_local

        horizon_frames_local = max(1, int(round(TARGET_INTEGRATION_T * args.data_hz)))
        tgt_idx_local = min(frame_idx + horizon_frames_local, num_frames - 1)

        q_term_local = model_handler.getReferenceState()[:nq].copy()
        q_term_local[0] = ball_xy_local[0] - base_xyz_local[0]
        q_term_local[1] = ball_xy_local[1] - base_xyz_local[1]
        q_term_local[2] = base_xyz_local[2]
        yaw_term_local = yaw[frame_idx] + wz_cmd_local * TARGET_INTEGRATION_T
        q_term_local[3:7] = quat_from_rpy(float(roll[frame_idx]), pitch_use, float(yaw_term_local))
        for joint_name, qv in zip(joint_order, q[tgt_idx_local]):
            joint_id = model.getJointId(joint_name)
            idx_q = model.joints[joint_id].idx_q
            q_term_local[idx_q : idx_q + model.joints[joint_id].nq] = qv

        return base_xyz_local, ball_xy_local, vx_cmd_local, wz_cmd_local, x_measured_local, v_world_cmd_local, q_term_local

    def process_frame(frame: int):
        nonlocal vel_line_id, vel_d_line_id, ang_line_id, ang_d_line_id, line_ids, mpc_line_ids
        nonlocal foot_traj_lines, foot_traj_prev, com_traj_lines, com_traj_prev, mpc_state_line_ids, mpc_foot_line_ids
        nonlocal mpc_state_line_bodies, mpc_foot_line_bodies
        base_xyz = world_xyz[frame]
        base_pos = np.array([base_xyz[0], base_xyz[1], base_xyz[2]]) + np.array(BASE_POS_OFFSET)
        quat = quat_from_rpy(float(roll[frame]), float(pitch[frame]), float(yaw[frame]))
        p.resetBasePositionAndOrientation(body_id, base_pos, quat)

        if SHOW_TARGET_GHOST and target_ghost_id is not None and tgt_world_xyz is not None:
            tgt_pos = [
                tgt_world_xyz[frame][0] + BASE_POS_OFFSET[0],
                tgt_world_xyz[frame][1] + BASE_POS_OFFSET[1],
                tgt_world_xyz[frame][2] + BASE_POS_OFFSET[2],
            ]
            p.resetBasePositionAndOrientation(target_ghost_id, tgt_pos, [0, 0, 0, 1])

        base_xyz, ball_xy, vx_cmd, wz_cmd, x_measured, v_world_cmd, q_term = compute_mpc_inputs(frame)
        if SHOW_TARGET_BALL and target_ball_id is not None:
            ball_pos = [ball_xy[0] + BASE_POS_OFFSET[0], ball_xy[1] + BASE_POS_OFFSET[1], args.base_height + BASE_POS_OFFSET[2]]
            p.resetBasePositionAndOrientation(target_ball_id, ball_pos, [0, 0, 0, 1])
        if SHOW_ICR_POINT and icr_id is not None:
            omega = float(v_world_cmd[5])
            if abs(omega) > 1e-6:
                icr_x = base_xyz[0] - v_world_cmd[1] / omega
                icr_y = base_xyz[1] + v_world_cmd[0] / omega
                icr_pos = [icr_x + BASE_POS_OFFSET[0], icr_y + BASE_POS_OFFSET[1], base_xyz[2] + BASE_POS_OFFSET[2]]
                p.resetBasePositionAndOrientation(icr_id, icr_pos, [0, 0, 0, 1])
            else:
                p.resetBasePositionAndOrientation(icr_id, [0.0, 0.0, -100.0], [0, 0, 0, 1])

        # Joints
        for idx, qv in zip(joint_indices, q[frame]):
            p.resetJointState(body_id, idx, float(qv))

        # Trajectory
        if SHOW_TRAJECTORY and world_xyz.shape[0] > 1:
            for lid in line_ids:
                p.removeUserDebugItem(lid)
            line_ids = []
            for i in range(1, frame + 1):
                p0 = [world_xyz[i - 1][0], world_xyz[i - 1][1], args.base_height]
                p1 = [world_xyz[i][0], world_xyz[i][1], args.base_height]
                line_ids.append(p.addUserDebugLine(p0, p1, lineColorRGB=[1, 0.1, 0.9], lineWidth=2))

        # Arrows
        if SHOW_VELOCITY_ARROWS:
            base_z = base_pos[2]
            if vel_world is not None:
                end = [base_xyz[0] + vel_world[frame][0] * VEL_SCALE, base_xyz[1] + vel_world[frame][1] * VEL_SCALE, base_z]
                vel_line_id = p.addUserDebugLine([base_xyz[0], base_xyz[1], base_z], end, lineColorRGB=[0.2, 0.9, 0.2], lineWidth=ARROW_WIDTH, replaceItemUniqueId=vel_line_id)
                if vel_head_id != -1:
                    p.resetBasePositionAndOrientation(vel_head_id, end, [0, 0, 0, 1])
            if vel_d_world is not None:
                end = [base_xyz[0] + vel_d_world[frame][0] * VEL_SCALE, base_xyz[1] + vel_d_world[frame][1] * VEL_SCALE, base_z + 0.02]
                vel_d_line_id = p.addUserDebugLine([base_xyz[0], base_xyz[1], base_z + 0.02], end, lineColorRGB=[0.2, 0.6, 1.0], lineWidth=ARROW_WIDTH, replaceItemUniqueId=vel_d_line_id)
                if vel_d_head_id != -1:
                    p.resetBasePositionAndOrientation(vel_d_head_id, end, [0, 0, 0, 1])
            if wz is not None:
                end = [base_xyz[0], base_xyz[1], base_z + 0.05 + wz[frame] * ANG_VEL_SCALE]
                ang_line_id = p.addUserDebugLine([base_xyz[0], base_xyz[1], base_z + 0.05], end, lineColorRGB=[1.0, 0.6, 0.2], lineWidth=ARROW_WIDTH, replaceItemUniqueId=ang_line_id)
                if ang_head_id != -1:
                    p.resetBasePositionAndOrientation(ang_head_id, end, [0, 0, 0, 1])
            if wz_d is not None:
                end = [base_xyz[0], base_xyz[1], base_z + 0.07 + wz_d[frame] * ANG_VEL_SCALE]
                ang_d_line_id = p.addUserDebugLine([base_xyz[0], base_xyz[1], base_z + 0.07], end, lineColorRGB=[1.0, 0.2, 0.8], lineWidth=ARROW_WIDTH, replaceItemUniqueId=ang_d_line_id)
                if ang_d_head_id != -1:
                    p.resetBasePositionAndOrientation(ang_d_head_id, end, [0, 0, 0, 1])

        # Adapt vector glide weight based on twist direction vs. foot alignment
        if problem_conf.get("use_vector_glide_cost", False):
            update_vector_glide_weight(problem_conf, model_handler, x_measured, v_world_cmd[5], wheel_link_names)

        # Fresh MPC instance each frame to reset initial guess
        mpc = create_mpc_instance()
        # reference velocity command (world)
        mpc.velocity_base = v_world_cmd

        # Terminal target: target position from local integration, joint angles from future frame
        x_term = model_handler.getReferenceState().copy()
        x_term[:nq] = q_term
        x_term[nq:] = 0.0
        mpc.ocp_handler.setTerminalReferenceState(x_term)

        reset_mpc_initial_guess(mpc, x_measured)
        mpc.iterate(x_measured)
        clamp_mpc_pitch(mpc.xs, nq, pitch_min, pitch_max)

        # Foot + COM trajectories (world frame)
        if SHOW_FOOT_TRAJECTORY or SHOW_COM_TRAJECTORY:
            data_traj = pin.Data(model)
            pin.forwardKinematics(model, data_traj, x_measured[:nq], x_measured[nq:])
            pin.updateFramePlacements(model, data_traj)
            if SHOW_FOOT_TRAJECTORY:
                foot_colors = {
                    "FL_wheel": [1.0, 0.0, 0.0],  # red
                    "FR_wheel": [0.0, 1.0, 0.0],  # green
                    "RL_wheel": [0.0, 0.4, 1.0],  # blue
                    "RR_wheel": [1.0, 0.9, 0.0],  # yellow
                }
                for name in wheel_link_names:
                    fid = model_handler.getFootFrameId(model_handler.getFootNb(name))
                    p_now = data_traj.oMf[fid].translation
                    prev = foot_traj_prev[name]
                    if prev is not None:
                        lid = p.addUserDebugLine(
                            [prev[0], prev[1], prev[2]],
                            [p_now[0], p_now[1], p_now[2]],
                            lineColorRGB=foot_colors[name],
                            lineWidth=2,
                        )
                        foot_traj_lines[name].append(lid)
                        if len(foot_traj_lines[name]) > MAX_TRAJ_POINTS:
                            p.removeUserDebugItem(foot_traj_lines[name].pop(0))
                    foot_traj_prev[name] = p_now.copy()
            if SHOW_COM_TRAJECTORY:
                com = pin.centerOfMass(model, data_traj, x_measured[:nq], x_measured[nq:])
                if com_traj_prev is not None:
                    lid = p.addUserDebugLine(
                        [com_traj_prev[0], com_traj_prev[1], com_traj_prev[2]],
                        [com[0], com[1], com[2]],
                        lineColorRGB=[0.0, 0.85, 0.85],
                        lineWidth=2,
                    )
                    com_traj_lines.append(lid)
                    if len(com_traj_lines) > MAX_TRAJ_POINTS:
                        p.removeUserDebugItem(com_traj_lines.pop(0))
                com_traj_prev = com.copy()

        # MPC state line: from current state to terminal ghost (sample ~10 points)
        if SHOW_MPC_STATE_LINE and mpc.xs:
            if use_geom_lines:
                for bid in mpc_state_line_bodies:
                    p.removeBody(bid)
                mpc_state_line_bodies = []
            else:
                for lid in mpc_state_line_ids:
                    p.removeUserDebugItem(lid)
                mpc_state_line_ids = []
            samples = 10
            idxs = np.linspace(0, len(mpc.xs) - 1, samples, dtype=int).tolist()
            pts = []
            for idx in idxs:
                xs = mpc.xs[int(idx)]
                pts.append([
                    xs[0] + base_xyz[0] + BASE_POS_OFFSET[0],
                    xs[1] + base_xyz[1] + BASE_POS_OFFSET[1],
                    xs[2] + BASE_POS_OFFSET[2],
                ])
            for i in range(1, len(pts)):
                if use_geom_lines:
                    bid = add_line_geom(pts[i - 1], pts[i], [0.85, 0.0, 0.85, 1.0], radius=0.02)
                    if bid is not None:
                        mpc_state_line_bodies.append(bid)
                else:
                    mpc_state_line_ids.append(
                        p.addUserDebugLine(
                            pts[i - 1],
                            pts[i],
                            lineColorRGB=[0.85, 0.0, 0.85],
                            lineWidth=6,
                        )
                    )

        # MPC foot trajectories: sample ~10 points along horizon
        if SHOW_MPC_FOOT_LINES and mpc.xs:
            foot_colors = {
                "FL_wheel": [1.0, 0.0, 0.0],  # red
                "FR_wheel": [0.0, 1.0, 0.0],  # green
                "RL_wheel": [0.0, 0.4, 1.0],  # blue
                "RR_wheel": [1.0, 0.9, 0.0],  # yellow
            }
            for name in wheel_link_names:
                if use_geom_lines:
                    for bid in mpc_foot_line_bodies[name]:
                        p.removeBody(bid)
                    mpc_foot_line_bodies[name] = []
                else:
                    for lid in mpc_foot_line_ids[name]:
                        p.removeUserDebugItem(lid)
                    mpc_foot_line_ids[name] = []
            samples = 10
            idxs = np.linspace(0, len(mpc.xs) - 1, samples, dtype=int).tolist()
            data_k = pin.Data(model)
            prev_pts = {name: None for name in wheel_link_names}
            for idx in idxs:
                xs = np.asarray(mpc.xs[int(idx)])
                q_world = xs[:nq].copy()
                q_world[0] += base_xyz[0]
                q_world[1] += base_xyz[1]
                q_world[2] += base_xyz[2]
                pin.forwardKinematics(model, data_k, q_world, xs[nq:])
                pin.updateFramePlacements(model, data_k)
                for name in wheel_link_names:
                    fid = model_handler.getFootFrameId(model_handler.getFootNb(name))
                    pw = data_k.oMf[fid].translation
                    z_fixed = foot_axis_z_world.get(name, float(pw[2]))
                    p_vis = [pw[0] + BASE_POS_OFFSET[0], pw[1] + BASE_POS_OFFSET[1], z_fixed + BASE_POS_OFFSET[2]]
                    prev = prev_pts[name]
                    if prev is not None:
                        if use_geom_lines:
                            bid = add_line_geom(prev, p_vis, foot_colors[name] + [1.0], radius=0.015)
                            if bid is not None:
                                mpc_foot_line_bodies[name].append(bid)
                        else:
                            lid = p.addUserDebugLine(
                                prev,
                                p_vis,
                                lineColorRGB=foot_colors[name],
                                lineWidth=4,
                            )
                            mpc_foot_line_ids[name].append(lid)
                    prev_pts[name] = p_vis
        rows = None
        if PRINT_MPC_DEBUG:
            print("=" * 80)
            print(f"[MPC] frame={frame} horizon_steps={len(mpc.xs) - 1}")
            print(f"  base_xyz={base_xyz}")
            print(f"  v_world_cmd={v_world_cmd}")
            print(f"  terminal_q={q_term}")
            rows = debug_mpc_horizon(mpc, model_handler, problem_conf, wheel_link_names, base_xyz[2])
        elif SAVE_MPC_DEBUG:
            rows = debug_mpc_horizon(mpc, model_handler, problem_conf, wheel_link_names, base_xyz[2])

        if SAVE_MPC_DEBUG and rows is not None and debug_csv_writer is not None:
            for row in rows:
                debug_csv_writer.writerow(
                    [
                        frame,
                        row["t"],
                        float(base_xyz[0]),
                        float(base_xyz[1]),
                        float(base_xyz[2]),
                        float(v_world_cmd[0]),
                        float(v_world_cmd[1]),
                        float(v_world_cmd[5]),
                        row["state_cost"],
                        row["control_cost"],
                        row["foot_cost"],
                        row["state_cost_base_pos"],
                        row["state_cost_base_ori"],
                        row["state_cost_joint_pos"],
                        row["state_cost_base_linvel"],
                        row["state_cost_base_angvel"],
                        row["state_cost_joint_vel"],
                        row["centroidal_cost"],
                        row["centroidal_cost_lin"],
                        row["centroidal_cost_ang"],
                        row["centroidal_derivative_cost"],
                        row["soft_contact_vel_cost"],
                        row["soft_friction_cost"],
                        row["track_width_cost"],
                        row["foot_sum_cost"],
                        row["min_wheel_distance_cost"],
                        row["vector_glide_cost"],
                        row["joint_limit_soft_cost"],
                        row.get("contact_states"),
                        row.get("centroidal_derivative_res"),
                        row.get("soft_contact_vel_res"),
                        row.get("soft_friction_res"),
                        row.get("track_width_res"),
                        row.get("foot_sum_res"),
                        row.get("min_wheel_distance_res"),
                        row.get("vector_glide_res"),
                        row.get("joint_limit_soft_res"),
                    ]
                )
            debug_csv_file.flush()

        if SHOW_MPC_GHOSTS and mpc.xs and mpc_terminal_ghost is not None:
            ghost_id, ghost_joints = mpc_terminal_ghost
            q_pred = np.asarray(mpc.xs[-1][:nq]).copy()
            q_pred_vis = q_pred.copy()
            # Display terminal prediction offset to current robot position
            q_pred_vis[0] += base_xyz[0] + BASE_POS_OFFSET[0]
            q_pred_vis[1] += base_xyz[1] + BASE_POS_OFFSET[1]
            q_pred_vis[2] = q_pred[2] + BASE_POS_OFFSET[2]
            update_ghost(ghost_id, ghost_joints, q_pred_vis)
        if SHOW_MPC_REF_GHOST and mpc_ref_ghost is not None:
            ghost_id, ghost_joints = mpc_ref_ghost
            try:
                x_term_ref = np.asarray(mpc.ocp_handler.getTerminalReferenceState())
                q_ref = x_term_ref[:nq].copy()
            except Exception:
                q_ref = q_term.copy()
            q_ref_vis = q_ref.copy()
            # Display terminal reference offset to current robot position
            q_ref_vis[0] += base_xyz[0] + BASE_POS_OFFSET[0]
            q_ref_vis[1] += base_xyz[1] + BASE_POS_OFFSET[1]
            q_ref_vis[2] = q_ref[2] + BASE_POS_OFFSET[2]
            update_ghost(ghost_id, ghost_joints, q_ref_vis)

        if SHOW_MPC_TRAJECTORY and mpc.xs:
            for lid in mpc_line_ids:
                p.removeUserDebugItem(lid)
            mpc_line_ids = []
            pts = []
            for xs in mpc.xs:
                pts.append([xs[0], xs[1], args.base_height])
            for i in range(1, len(pts)):
                mpc_line_ids.append(
                    p.addUserDebugLine(
                        [pts[i - 1][0] + BASE_POS_OFFSET[0], pts[i - 1][1] + BASE_POS_OFFSET[1], pts[i - 1][2]],
                        [pts[i][0] + BASE_POS_OFFSET[0], pts[i][1] + BASE_POS_OFFSET[1], pts[i][2]],
                        lineColorRGB=[0.9, 0.8, 0.2],
                        lineWidth=2,
                    )
                )

        if FOLLOW_CAMERA:
            cam_yaw = p.readUserDebugParameter(cam_yaw_slider)
            cam_pitch = p.readUserDebugParameter(cam_pitch_slider)
            cam_dist = p.readUserDebugParameter(cam_dist_slider)
            p.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, base_pos.tolist())

    video_out = args.video_out
    if args.video and not video_out:
        video_out = "out.mp4"

    if video_out:
        use_geom_lines = True
        try:
            import imageio.v2 as imageio
        except Exception as exc:
            raise SystemExit("imageio is required for video output. Install with: pip install imageio imageio-ffmpeg") from exc
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        start_frame = max(args.video_start, 0)
        end_frame = args.video_end if args.video_end >= 0 else num_frames - 1
        end_frame = min(end_frame, num_frames - 1)
        if args.video_range:
            try:
                range_str = args.video_range.strip()
                nums = re.findall(r"[0-9]*\.?[0-9]+", range_str)
                if len(nums) != 2:
                    raise ValueError
                start_pct = float(nums[0])
                end_pct = float(nums[1])
                if not (0.0 <= start_pct <= 100.0 and 0.0 <= end_pct <= 100.0):
                    raise ValueError
                if end_pct < start_pct:
                    start_pct, end_pct = end_pct, start_pct
                total = max(num_frames - 1, 0)
                start_frame = int(round(total * (start_pct / 100.0)))
                end_frame = int(round(total * (end_pct / 100.0)))
            except Exception:
                raise SystemExit("Invalid --video-range. Use format like 30%-50%")
        step = max(1, int(round(args.data_hz / max(args.video_fps, 1e-6))))
        frames = list(range(start_frame, end_frame + 1, step))
        writer = imageio.get_writer(video_out, fps=args.video_fps)
        for frame in frames:
            process_frame(frame)
            p.stepSimulation()
            rgb = capture_frame(args.video_width, args.video_height)
            writer.append_data(rgb)
        writer.close()
        return

    use_geom_lines = False
    last_frame = -1
    while True:
        frame = int(p.readUserDebugParameter(frame_slider))
        if frame != last_frame:
            last_frame = frame
            process_frame(frame)

        p.stepSimulation()
        time.sleep(1.0 / max(args.hz, 1.0))


if __name__ == "__main__":
    main()
