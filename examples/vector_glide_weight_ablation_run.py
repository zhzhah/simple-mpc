#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import time
import sys

import numpy as np
import pinocchio as pin
import pybullet as p
import example_robot_data as erd
from example_robot_data.robots_loader import ROBOTS, RobotLoader

from simple_mpc import (
    RobotModelHandler,
    RobotDataHandler,
    KinodynamicsOCP,
    MPC,
    Interpolator,
    KinodynamicsID,
    KinodynamicsIDSettings,
)
from bullet_robot import BulletRobot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vector-glide weight ablation for go2w.")
    parser.add_argument("--out-dir", type=str, default="out/vector_glide_ablation")
    parser.add_argument("--r-min", type=float, default=0.1)
    parser.add_argument("--r-max", type=float, default=2.0)
    parser.add_argument("--r-step", type=float, default=0.5)
    parser.add_argument("--w-min", type=float, default=0.0)
    parser.add_argument("--w-max", type=float, default=400.0)
    parser.add_argument("--w-step", type=float, default=100.0)
    parser.add_argument("--dt-mpc", type=float, default=0.01)
    parser.add_argument("--horizon-steps", type=int, default=50)
    parser.add_argument("--init-yaws", type=str, default="0,1.5708,3.1416,-1.5708")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-factor", type=float, default=1.3)
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI")
    return parser.parse_args()


def ensure_go2w_robot():
    if "go2w" not in ROBOTS:
        class Go2WLoader(RobotLoader):
            path = "go2w_description"
            urdf_filename = "go2w.urdf"
            urdf_subpath = "urdf"
            srdf_filename = "go2w.srdf"
            ref_posture = "standing"
            free_flyer = True

        ROBOTS["go2w"] = Go2WLoader


def build_problem(dt_mpc: float):
    ensure_go2w_robot()
    robot_wrapper = erd.load("go2w")
    if "standing" not in robot_wrapper.model.referenceConfigurations:
        robot_wrapper.model.referenceConfigurations["standing"] = pin.neutral(robot_wrapper.model)

    model_handler = RobotModelHandler(robot_wrapper.model, "standing", "root_joint")
    model_handler.addPointFoot("FL_wheel", "root_joint")
    model_handler.addPointFoot("FR_wheel", "root_joint")
    model_handler.addPointFoot("RL_wheel", "root_joint")
    model_handler.addPointFoot("RR_wheel", "root_joint")
    data_handler = RobotDataHandler(model_handler)

    model = model_handler.getModel()
    gravity = np.array([0.0, 0.0, -9.81])
    force_size = 3

    w_basepos = [0.0, 0.0, 100.0, 20.0, 20.0, 0.0]
    w_legpos = [10.1, 10.1, 10.1, 0.0]
    w_basevel = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    w_legvel = [10.1, 10.1, 10.1, 10.1]

    w_q = np.zeros(model.nv)
    w_v = np.zeros(model.nv)
    w_q[:6] = w_basepos
    w_v[:6] = w_basevel
    w_q[6:] = np.resize(np.tile(w_legpos, 4), w_q[6:].shape[0])
    w_v[6:] = np.resize(np.tile(w_legvel, 4), w_v[6:].shape[0])

    w_x = np.diag(np.concatenate((w_q, w_v)))
    w_linforce = np.array([0.01, 0.01, 0.01])
    w_u_joints = np.ones(model.nv - 6) * 1e-5
    w_u = np.concatenate((w_linforce, w_linforce, w_linforce, w_linforce, w_u_joints))
    w_u = np.diag(w_u)
    w_cent_lin = np.array([0.0, 0.0, 1.0])
    w_cent_ang = np.array([0.0, 0.1, 10.0])
    w_cent = np.diag(np.concatenate((w_cent_lin, w_cent_ang)))
    w_centder_lin = np.ones(3) * 0.0
    w_centder_ang = np.ones(3) * 0.1
    w_centder = np.diag(np.concatenate((w_centder_lin, w_centder_ang)))

    qmin = model.lowerPositionLimit[7:].copy()
    qmax = model.upperPositionLimit[7:].copy()
    for joint_name in ["FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint"]:
        jid = model.getJointId(joint_name)
        idx_q = model.joints[jid].idx_q
        nq_joint = model.joints[jid].nq
        qmin[idx_q - 7 : idx_q - 7 + nq_joint] = -1e9
        qmax[idx_q - 7 : idx_q - 7 + nq_joint] = 1e9

    problem_conf = dict(
        timestep=dt_mpc,
        w_x=w_x,
        w_u=w_u,
        w_cent=w_cent,
        w_centder=w_centder,
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
        vector_glide_weight=0.0,
        vector_glide_min_omega=1e-6,
        min_wheel_distance_cstr=False,
        min_wheel_distance=0.2,
        min_wheel_distance_cost=False,
        w_min_wheel_distance=0.05,
        min_wheel_distance_cost_eps=1e-3,
        force_z_variance_cost=False,
        w_force_z_variance=10.0,
        joint_limit_soft_cost=False,
        w_joint_limit_soft=2.0,
        joint_limit_soft_fraction=0.5,
        soft_constraints=True,
        w_soft_contact_vel=20.0,
        w_soft_friction=0.0,
        w_soft_land=0.0,
        track_width_cstr=True,
        w_track_width=6000.0,
        foot_sum_cstr=False,
        w_foot_sum=6000.0,
        foot_sum_z_offset=0.0762,
    )

    return model_handler, data_handler, problem_conf


def build_id(model_handler, dt_simu: float):
    settings = KinodynamicsIDSettings()
    settings.kp_base = 15.0
    settings.kp_posture = 30.0
    settings.kp_contact = 0.0
    settings.w_base = 100.0
    settings.w_posture = 10.0
    settings.w_contact_force = 1.0
    settings.w_contact_motion = 0.0
    settings.kp_posture_wheel = 0.001
    settings.w_posture_wheel = 0.001
    settings.wheel_radius = 0.0762
    settings.ff_wheel_scale = 1.0
    settings.enable_nonholonomic = False
    settings.enable_lateral_no_slip = False
    settings.lateral_no_slip_use_bounds = True
    settings.lateral_no_slip_lower = -3
    settings.lateral_no_slip_upper = 3
    settings.use_vector_glide_cost = False
    settings.vector_glide_weight = 1.0
    kino_ID = KinodynamicsID(model_handler, dt_simu, settings)
    for name in ["FL_wheel", "FR_wheel", "RL_wheel", "RR_wheel"]:
        kino_ID.addLateralNoSlipConstraint(name)
    return kino_ID


def compute_command_for_radius(radius: float) -> tuple[float, float]:
    vx = 0.4
    wz = vx / radius
    if abs(wz) > 1.5:
        wz = math.copysign(1.5, wz)
        vx = abs(wz) * radius
    return vx, wz


def run_one(
    model_handler,
    data_handler,
    problem_conf,
    radius: float,
    weight: float,
    yaw0: float,
    dt_mpc: float,
    horizon_steps: int,
    max_factor: float,
    out_dir: Path,
):
    model = model_handler.getModel()
    nq = model.nq
    nv = model.nv
    force_size = int(problem_conf["force_size"])
    gravity = np.array(problem_conf["gravity"], dtype=float)

    vx_cmd, wz_cmd = compute_command_for_radius(radius)
    expected_time = 2.0 * math.pi / max(abs(wz_cmd), 1e-6)
    max_time = max_factor * expected_time

    # Force headless mode for batch runs unless GUI requested
    if not problem_conf.get("_gui_enabled", False):
        os.environ.pop("DISPLAY", None)

    URDF_SUBPATH = "/go2w_description/urdf/go2w.urdf"
    device = BulletRobot(
        model_handler.getModel().names,
        erd.getModelPath(URDF_SUBPATH),
        URDF_SUBPATH,
        dt_mpc,
        model_handler.getModel(),
        model_handler.getReferenceState()[:3],
    )

    q_init = model_handler.getReferenceState()[:nq].copy()
    q_init[:3] = np.array([0.0, 0.0, 0.305])
    R0 = pin.rpy.rpyToMatrix(0.0, 0.0, yaw0)
    q_init[3:7] = pin.Quaternion(R0).coeffs()
    device.initializeJoints(q_init)

    dynproblem = KinodynamicsOCP(problem_conf, model_handler)
    dynproblem.createProblem(model_handler.getReferenceState(), horizon_steps, force_size, gravity[2], False)

    mpc_conf = dict(
        support_force=-model_handler.getMass() * gravity[2],
        TOL=1e-4,
        mu_init=1e-8,
        max_iters=1,
        num_threads=8,
        swing_apex=0.15,
        T_fly=30,
        T_contact=10,
        timestep=dt_mpc,
        update_contact_ref=True,
    )
    mpc = MPC(mpc_conf, dynproblem)
    mpc.ocp_handler.setVectorGlideWeight(weight)
    contact_phase_quadru = {
        "FL_wheel": True,
        "FR_wheel": True,
        "RL_wheel": True,
        "RR_wheel": True,
    }
    mpc.generateCycleHorizon([contact_phase_quadru])

    N_simu = 10
    dt_simu = dt_mpc / N_simu
    interpolator = Interpolator(model_handler.getModel())
    kino_ID = build_id(model_handler, dt_simu)

    v_body_cmd = np.zeros(6)
    v_body_cmd[0] = vx_cmd
    v_body_cmd[5] = wz_cmd

    q_meas, v_meas = device.measureState()
    x_measured = np.concatenate([q_meas, v_meas])

    yaw_accum = yaw0
    last_yaw = yaw0

    t = 0.0
    wall_start = time.time()
    wall_limit = max_time * 2.0
    lin_errs = []
    yaw_errs = []
    completed = False

    try:
        for step in range(1000000):
            if (time.time() - wall_start) > wall_limit:
                break
            if step % 200 == 0:
                print(f"  [progress] t={t:.2f}s step={step}")
                sys.stdout.flush()
            q_meas, v_meas = device.measureState()
            x_measured = np.concatenate([q_meas, v_meas])
            base_R_world = pin.Quaternion(q_meas[3:7]).toRotationMatrix()
            yaw = math.atan2(base_R_world[1, 0], base_R_world[0, 0])
            dyaw = yaw - last_yaw
            if dyaw > math.pi:
                dyaw -= 2.0 * math.pi
            elif dyaw < -math.pi:
                dyaw += 2.0 * math.pi
            yaw_accum += dyaw
            last_yaw = yaw

            v_world_cmd = np.zeros(6)
            v_world_cmd[0] = v_body_cmd[0] * math.cos(yaw_accum)
            v_world_cmd[1] = v_body_cmd[0] * math.sin(yaw_accum)
            v_world_cmd[5] = v_body_cmd[5]
            mpc.velocity_base = v_world_cmd

            # reference: fixed terminal state only (standing pose)
            x_ref = model_handler.getReferenceState().copy()
            for t_idx in range(horizon_steps):
                mpc.ocp_handler.setReferenceState(t_idx, x_ref)
            mpc.ocp_handler.setTerminalReferenceState(x_ref)

            if step == 0:
                print("  [debug] before mpc.iterate")
                sys.stdout.flush()
            mpc.iterate(x_measured)
            if step == 0:
                print("  [debug] after mpc.iterate")
                sys.stdout.flush()

            forces0 = mpc.us[0][: model_handler.getFeetNb() * force_size]
            forces1 = mpc.us[1][: model_handler.getFeetNb() * force_size]
            a0 = mpc.getStateDerivative(0)[nv:].copy()
            a1 = mpc.getStateDerivative(1)[nv:].copy()
            a0[6:] = mpc.us[0][model_handler.getFeetNb() * force_size :]
            a1[6:] = mpc.us[1][model_handler.getFeetNb() * force_size :]
            xss = [mpc.xs[0], mpc.xs[1]]
            forces = [forces0, forces1]
            ddqs = [a0, a1]
            contact_states = mpc.ocp_handler.getContactState(0)

            for sub in range(N_simu):
                delay = sub / float(N_simu) * dt_mpc
                xs_interp = interpolator.interpolateState(delay, dt_mpc, xss)
                acc_interp = interpolator.interpolateLinear(delay, dt_mpc, ddqs)
                force_interp = interpolator.interpolateLinear(delay, dt_mpc, forces).reshape((4, 3))
                q_interp = xs_interp[:nq]
                v_interp = xs_interp[nq:]
                force_interp = [force_interp[i, :] for i in range(4)]

                kino_ID.setTarget(q_interp, v_interp, acc_interp, contact_states, force_interp)
                if step == 0 and sub == 0:
                    print("  [debug] before kino_ID.solve")
                    sys.stdout.flush()
                tau_cmd = kino_ID.solve(t, q_meas, v_meas)
                if step == 0 and sub == 0:
                    print("  [debug] after kino_ID.solve")
                    sys.stdout.flush()
                device.execute(tau_cmd)
                t += dt_simu

            # tracking errors (world frame)
            base_R_world = pin.Quaternion(q_meas[3:7]).toRotationMatrix()
            v_meas_world = np.zeros(6)
            v_meas_world[:3] = base_R_world @ v_meas[:3]
            v_meas_world[3:6] = base_R_world @ v_meas[3:6]
            lin_err = math.hypot(v_meas_world[0] - v_world_cmd[0], v_meas_world[1] - v_world_cmd[1])
            yaw_err = abs(v_meas_world[5] - v_world_cmd[5])
            lin_errs.append(lin_err)
            yaw_errs.append(yaw_err)

            if abs(yaw_accum - yaw0) >= 2.0 * math.pi:
                completed = True
                break
            if t >= max_time:
                break
    except Exception:
        completed = False
    finally:
        p.disconnect()

    lin_mean = float(np.mean(lin_errs)) if lin_errs else float("nan")
    yaw_mean = float(np.mean(yaw_errs)) if yaw_errs else float("nan")

    meta = {
        "radius": radius,
        "vx_cmd": vx_cmd,
        "wz_cmd": wz_cmd,
        "weight": weight,
        "yaw0": yaw0,
        "expected_time": expected_time,
        "max_time": max_time,
        "actual_time": t,
        "completed": completed,
        "lin_err_mean": lin_mean,
        "yaw_err_mean": yaw_mean,
    }

    fname = f"vg_r{radius:.3f}_w{weight:.1f}_vx{vx_cmd:.3f}_wz{wz_cmd:.3f}_yaw{yaw0:.3f}_ok{int(completed)}.csv"
    out_path = out_dir / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for k, v in meta.items():
            f.write(f"# {k}: {v}\n")
        f.write("lin_err,yaw_err\n")
        for le, ye in zip(lin_errs, yaw_errs):
            f.write(f"{le},{ye}\n")
    print(f"  [done] r={radius:.3f} w={weight:.1f} yaw0={yaw0:.3f} ok={int(completed)} time={t:.2f}s")
    sys.stdout.flush()
    return meta


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_handler, data_handler, problem_conf = build_problem(args.dt_mpc)
    problem_conf["use_vector_glide_cost"] = True
    problem_conf["_gui_enabled"] = bool(args.gui)

    radii = np.arange(args.r_min, args.r_max + 1e-9, args.r_step)
    weights = np.arange(args.w_min, args.w_max + 1e-9, args.w_step)
    init_yaws = [float(x) for x in args.init_yaws.split(",") if x.strip() != ""]

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8") as fsum:
        fsum.write(
            "radius,weight,vx_cmd,wz_cmd,yaw0,expected_time,max_time,actual_time,completed,lin_err_mean,yaw_err_mean\n"
        )
        fsum.flush()
        for r in radii:
            for w in weights:
                for yaw0 in init_yaws:
                    print(f"[run] r={r:.3f} w={w:.1f} yaw0={yaw0:.3f}")
                    sys.stdout.flush()
                    meta = run_one(
                        model_handler,
                        data_handler,
                        problem_conf,
                        float(r),
                        float(w),
                        float(yaw0),
                        args.dt_mpc,
                        args.horizon_steps,
                        args.max_factor,
                        out_dir,
                    )
                    fsum.write(
                        f"{meta['radius']},{meta['weight']},{meta['vx_cmd']},{meta['wz_cmd']},{meta['yaw0']},"
                        f"{meta['expected_time']},{meta['max_time']},{meta['actual_time']},"
                        f"{int(meta['completed'])},{meta['lin_err_mean']},{meta['yaw_err_mean']}\n"
                    )
                    fsum.flush()


if __name__ == "__main__":
    main()
