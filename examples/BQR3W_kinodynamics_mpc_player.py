#!/usr/bin/env python3
"""MPC-only player: roll the first MPC solution without ID or dynamics feedback."""
from __future__ import annotations

import argparse
import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import pinocchio as pin
import example_robot_data as erd
from example_robot_data.robots_loader import ROBOTS, RobotLoader
from simple_mpc import RobotModelHandler, RobotDataHandler, KinodynamicsOCP, MPC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BQR3W MPC player (no ID, no dynamics).")
    parser.add_argument("--vx", type=float, default=0.2, help="Body-frame forward velocity command (m/s).")
    parser.add_argument("--wz", type=float, default=0.0, help="Body-frame yaw rate command (rad/s).")
    parser.add_argument("--init-x", type=float, default=0.0, help="Initial base x (m).")
    parser.add_argument("--init-y", type=float, default=0.0, help="Initial base y (m).")
    parser.add_argument("--init-z", type=float, default=0.25, help="Initial base z (m).")
    parser.add_argument("--init-roll", type=float, default=None, help="Initial base roll (rad).")
    parser.add_argument("--init-pitch", type=float, default=None, help="Initial base pitch (rad).")
    parser.add_argument("--init-yaw", type=float, default=None, help="Initial base yaw (rad).")
    parser.add_argument("--steps", type=int, default=2000, help="Number of MPC steps to play.")
    parser.add_argument("--hz", type=float, default=60.0, help="UI update rate.")
    return parser.parse_args()


def get_joint_name_to_index(body_id: int) -> dict[str, int]:
    name_to_index: dict[str, int] = {}
    for i in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, i)
        name = info[1].decode("utf-8")
        name_to_index[name] = i
    return name_to_index


def spawn_ghost(urdf_path: str, color_rgba, model_names: list[str]):
    ghost_id = p.loadURDF(urdf_path, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], useFixedBase=True)
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


def build_mpc():
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
    gravity = np.array([0, 0, -9.81])
    force_size = 3

    dt_mpc = 0.01
    w_basepos = [0, 0, 0, 0.0, 1000.0, 1000.0]
    w_jointpos = [100.1, 100.1, 100.1, 0.0]
    w_basevel = [20, 20, 10, 10, 10, 20]
    w_jointvel = [0.1, 0.1, 0.1, 0.0]

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
        wheel_axle_height_min=0.9 * wheel_radius,
        wheel_axle_height_max=1.1 * wheel_radius,
    )

    horizon_T = int(round(1.0 / dt_mpc))
    dynproblem = KinodynamicsOCP(problem_conf, model_handler)
    dynproblem.createProblem(
        model_handler.getReferenceState(), horizon_T, force_size, gravity[2], False
    )

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

    mpc = MPC(mpc_conf, dynproblem)
    contact_phase_quadru = {
        "FL_wheel": True,
        "FR_wheel": True,
        "RL_wheel": True,
        "RR_wheel": True,
    }
    mpc.generateCycleHorizon([contact_phase_quadru])

    return model_handler, data_handler, mpc, problem_conf, dt_mpc, horizon_T


def main() -> None:
    args = parse_args()
    model_handler, _data_handler, mpc, _problem_conf, dt_mpc, horizon_T = build_mpc()
    model = model_handler.getModel()
    nq = model.nq
    nv = model.nv

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setAdditionalSearchPath(erd.getModelPath(""))
    p.setGravity(0, 0, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

    urdf_path = os.path.join(erd.getModelPath(""), "BQR3W_go2_description/urdf/BQR3W_go2.urdf")
    robot_id = p.loadURDF(urdf_path, [0, 0, 0.25], useFixedBase=False, flags=p.URDF_IGNORE_COLLISION_SHAPES)
    name_to_index = get_joint_name_to_index(robot_id)

    joint_order = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_wheel_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_wheel_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_wheel_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_wheel_joint",
    ]
    joint_indices = [name_to_index[n] for n in joint_order]
    joint_q_index = {}
    for name in joint_order:
        joint_id = model.getJointId(name)
        joint_q_index[name] = model.joints[joint_id].idx_q

    # Terminal prediction ghost + terminal reference ghost
    model_names = list(model.names)
    ghost_id, ghost_joints = spawn_ghost(urdf_path, [0.9, 0.6, 0.2, 0.35], model_names)
    ref_ghost_id, ref_ghost_joints = spawn_ghost(urdf_path, [0.2, 0.9, 0.2, 0.35], model_names)

    vx_slider = p.addUserDebugParameter("vx_body", -1.0, 1.0, args.vx)
    wz_slider = p.addUserDebugParameter("wz_body", -1.5, 1.5, args.wz)

    x_measured = model_handler.getReferenceState().copy()
    ref_state_anchor = model_handler.getReferenceState().copy()
    R_init = pin.Quaternion(x_measured[3:7]).toRotationMatrix()
    rpy_init = pin.rpy.matrixToRpy(R_init)
    init_roll = rpy_init[0] if args.init_roll is None else args.init_roll
    init_pitch = rpy_init[1] if args.init_pitch is None else args.init_pitch
    init_yaw = rpy_init[2] if args.init_yaw is None else args.init_yaw
    R_init = pin.rpy.rpyToMatrix(init_roll, init_pitch, init_yaw)
    q_init = pin.Quaternion(R_init).coeffs()
    x_measured[0] = args.init_x
    x_measured[1] = args.init_y
    x_measured[2] = args.init_z
    x_measured[3:7] = q_init
    ref_state_anchor[0] = args.init_x
    ref_state_anchor[1] = args.init_y
    ref_state_anchor[2] = args.init_z
    ref_state_anchor[3:7] = q_init
    ref_x = float(ref_state_anchor[0])
    ref_y = float(ref_state_anchor[1])
    ref_yaw = float(init_yaw)

    def _print_full_state(step_idx: int, q_state: np.ndarray, v_state: np.ndarray):
        R = pin.Quaternion(q_state[3:7]).toRotationMatrix()
        rpy = pin.rpy.matrixToRpy(R)
        print("=" * 80)
        print(f"[STATE] step={step_idx}")
        print("  base rpy", rpy)
        print("  base q", q_state[:7])
        print("  base v", v_state[:6])
        print("  joints q", q_state[7:nq])
        print("  joints v", v_state[6:nv])

    paused = True
    _print_full_state(0, x_measured[:nq], x_measured[nq:])
    print("[UI] paused = True (press 'p' to start, 'n' single-step)")
    while paused:
        events = p.getKeyboardEvents()
        if ord('p') in events and events[ord('p')] & p.KEY_WAS_TRIGGERED:
            paused = False
            print("[UI] paused = False")
            break
        if ord('n') in events and events[ord('n')] & p.KEY_WAS_TRIGGERED:
            paused = True
            break
        time.sleep(0.01)

    for step in range(args.steps):
        events = p.getKeyboardEvents()
        if ord('p') in events and events[ord('p')] & p.KEY_WAS_TRIGGERED:
            paused = not paused
            print("[UI] paused =", paused)
        step_once = False
        if ord('n') in events and events[ord('n')] & p.KEY_WAS_TRIGGERED:
            step_once = True
            paused = True
        if paused and not step_once:
            time.sleep(0.01)
            continue

        vx_cmd = float(p.readUserDebugParameter(vx_slider))
        wz_cmd = float(p.readUserDebugParameter(wz_slider))

        v_world_cmd = np.zeros(6)
        v_world_cmd[0] = vx_cmd * np.cos(ref_yaw)
        v_world_cmd[1] = vx_cmd * np.sin(ref_yaw)
        v_world_cmd[5] = wz_cmd
        mpc.velocity_base = v_world_cmd

        x0 = float(ref_x)
        y0 = float(ref_y)
        yaw0 = float(ref_yaw)
        dt_ref = horizon_T * dt_mpc
        yaw_t = yaw0 + wz_cmd * dt_ref
        x_t = x0 + vx_cmd * np.cos(yaw_t) * dt_ref
        y_t = y0 + vx_cmd * np.sin(yaw_t) * dt_ref
        R_ref = pin.rpy.rpyToMatrix(init_roll, init_pitch, yaw_t)
        q_ref = pin.Quaternion(R_ref).coeffs()
        x_ref_last = ref_state_anchor.copy()
        x_ref_last[0] = x_t
        x_ref_last[1] = y_t
        x_ref_last[3:7] = q_ref
        x_ref_last[nq : nq + 6] = v_world_cmd
        for t_idx in range(horizon_T):
            mpc.ocp_handler.setReferenceState(t_idx, x_ref_last)

        ref_yaw = yaw0 + wz_cmd * dt_mpc
        ref_x = x0 + vx_cmd * np.cos(ref_yaw) * dt_mpc
        ref_y = y0 + vx_cmd * np.sin(ref_yaw) * dt_mpc
        ref_state_anchor[0] = ref_x
        ref_state_anchor[1] = ref_y
        R_anchor = pin.rpy.rpyToMatrix(init_roll, init_pitch, ref_yaw)
        ref_state_anchor[3:7] = pin.Quaternion(R_anchor).coeffs()

        mpc.iterate(x_measured)
        if step % 50 == 0 and x_ref_last is not None:
            print("=" * 80)
            print(f"[MPC] step={step} horizon={horizon_T}")
            print(f"  base_xyz={x_measured[:3]}")
            print(f"  v_world_cmd={v_world_cmd}")
            print(f"  terminal_ref_xyz={x_ref_last[:3]}")
        if len(mpc.xs) > 1:
            x_measured = np.asarray(mpc.xs[1]).copy()
        _print_full_state(step + 1, x_measured[:nq], x_measured[nq:])

        q_vis = x_measured[:nq]
        p.resetBasePositionAndOrientation(robot_id, q_vis[:3], q_vis[3:7])
        for jname, j_idx in zip(joint_order, joint_indices):
            q_idx = joint_q_index[jname]
            p.resetJointState(robot_id, j_idx, float(q_vis[q_idx]))

        if mpc.xs:
            q_pred = np.asarray(mpc.xs[-1][:nq]).copy()
            update_ghost(ghost_id, ghost_joints, q_pred)
        if x_ref_last is not None:
            q_ref = np.asarray(x_ref_last[:nq]).copy()
            update_ghost(ref_ghost_id, ref_ghost_joints, q_ref)

        p.stepSimulation()
        time.sleep(1.0 / max(args.hz, 1.0))


if __name__ == "__main__":
    main()
