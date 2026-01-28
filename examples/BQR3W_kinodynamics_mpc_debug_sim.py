import numpy as np
import time
import pinocchio as pin
import pybullet as p
import example_robot_data as erd

from bullet_robot import BulletRobot
from simple_mpc import Interpolator, KinodynamicsID, KinodynamicsIDSettings

import BQR3W_kinodynamics_mpc_debug as dbg


def main():
    dt_mpc = dbg.dt_mpc
    dt_simu = 0.001
    N_simu = max(1, int(round(dt_mpc / dt_simu)))
    dt_simu = dt_mpc / N_simu

    model_handler = dbg.model_handler
    nq = model_handler.getModel().nq
    nv = model_handler.getModel().nv
    force_size = 3
    nk = model_handler.getFeetNb()

    mpc = dbg._build_mpc()
    interpolator = Interpolator(model_handler.getModel())

    kino_ID_settings = KinodynamicsIDSettings()
    kino_ID_settings.kp_base = 15.0
    kino_ID_settings.kp_posture = 10.0
    kino_ID_settings.kp_contact = 0
    kino_ID_settings.w_base = 100.0
    kino_ID_settings.w_posture = 10.0
    # Separate wheel posture weight/gain from leg joints
    kino_ID_settings.kp_posture_wheel = 0.0
    kino_ID_settings.w_posture_wheel = 1.0
    kino_ID_settings.w_contact_force = 1.0
    kino_ID_settings.w_contact_motion = 0.0
    kino_ID_settings.wheel_radius = dbg.wheel_radius
    kino_ID_settings.enable_nonholonomic = False
    kino_ID_settings.enable_lateral_no_slip = False
    kino_ID_settings.use_vector_glide_cost = False
    kino_ID = KinodynamicsID(model_handler, dt_simu, kino_ID_settings)

    device = BulletRobot(
        model_handler.getModel().names,
        erd.getModelPath(dbg.URDF_SUBPATH),
        dbg.URDF_SUBPATH,
        dt_simu,
        model_handler.getModel(),
        model_handler.getReferenceState()[:3],
    )
    device.initializeJoints(model_handler.getReferenceState()[: model_handler.getModel().nq])
    device.changeCamera(1.0, 60, -15, [0.6, -0.2, 0.5])

    # MPC ghost robot (terminal state visualization)
    target_model_path = erd.getModelPath(dbg.URDF_SUBPATH)
    p.setAdditionalSearchPath(target_model_path)
    ghost_id = p.loadURDF(
        dbg.URDF_SUBPATH, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], useFixedBase=True
    )
    ghost_local_inertia = np.array(p.getDynamicsInfo(ghost_id, -1)[3])
    ghost_joint_names = [
        p.getJointInfo(ghost_id, i)[1].decode() for i in range(p.getNumJoints(ghost_id))
    ]
    ghost_joint_indices = [
        ghost_joint_names.index(model_handler.getModel().names[i])
        for i in range(2, model_handler.getModel().njoints)
    ]
    for link_id in range(-1, p.getNumJoints(ghost_id)):
        p.setCollisionFilterGroupMask(ghost_id, link_id, 0, 0)
        p.changeVisualShape(ghost_id, link_id, rgbaColor=[0.95, 0.5, 0.2, 0.35])

    v_body_cmd = np.zeros(6)
    v_body_cmd[0] = 0.2
    forward_speed_slider = p.addUserDebugParameter("forward_speed_body", -1.0, 1.0, v_body_cmd[0])
    yaw_rate_slider = p.addUserDebugParameter("yaw_rate_body", -1.0, 1.0, v_body_cmd[5])

    q_meas, v_meas = device.measureState()
    x_measured = np.concatenate([q_meas, v_meas])

    while True:
        v_body_cmd[0] = p.readUserDebugParameter(forward_speed_slider)
        v_body_cmd[5] = p.readUserDebugParameter(yaw_rate_slider)

        base_R_world = pin.Quaternion(x_measured[3:7]).toRotationMatrix()
        yaw = np.arctan2(base_R_world[1, 0], base_R_world[0, 0])
        v_world_cmd = np.zeros(6)
        v_world_cmd[0] = v_body_cmd[0] * np.cos(yaw)
        v_world_cmd[1] = v_body_cmd[0] * np.sin(yaw)
        v_world_cmd[5] = v_body_cmd[5]

        mpc.velocity_base = v_world_cmd

        x_t = float(x_measured[0])
        y_t = float(x_measured[1])
        yaw_t = float(yaw)
        for t_idx in range(dbg.T):
            yaw_t = yaw_t + v_body_cmd[5] * dt_mpc
            x_t = x_t + v_body_cmd[0] * np.cos(yaw_t) * dt_mpc
            y_t = y_t + v_body_cmd[0] * np.sin(yaw_t) * dt_mpc
            R_ref = pin.rpy.rpyToMatrix(0.0, 0.0, yaw_t)
            q_ref = pin.Quaternion(R_ref).coeffs()
            x_ref = model_handler.getReferenceState().copy()
            x_ref[0] = x_t
            x_ref[1] = y_t
            x_ref[3:7] = q_ref
            x_ref[nq : nq + 6] = v_world_cmd
            mpc.ocp_handler.setReferenceState(t_idx, x_ref)

        mpc.iterate(x_measured)

        forces0 = mpc.us[0][: nk * force_size]
        forces1 = mpc.us[1][: nk * force_size]
        a0 = mpc.getStateDerivative(0)[nv:].copy()
        a1 = mpc.getStateDerivative(1)[nv:].copy()
        a0[6:] = mpc.us[0][nk * force_size :]
        a1[6:] = mpc.us[1][nk * force_size :]

        forces = [forces0, forces1]
        ddqs = [a0, a1]
        xss = [mpc.xs[0], mpc.xs[1]]

        contact_states = mpc.ocp_handler.getContactState(0)

        if len(mpc.xs) > 0:
            q_ghost = np.asarray(mpc.xs[-1][:nq]).copy()
            R_ref = pin.Quaternion(q_ghost[3:7]).toRotationMatrix()
            offset = R_ref @ ghost_local_inertia
            pos = [q_ghost[0] + offset[0], q_ghost[1] + offset[1], q_ghost[2] + offset[2]]
            p.resetBasePositionAndOrientation(ghost_id, pos, q_ghost[3:7])
            for i, j_idx in enumerate(ghost_joint_indices):
                p.resetJointState(ghost_id, j_idx, q_ghost[7 + i])

        for sub_step in range(N_simu):
            delay = sub_step / float(N_simu) * dt_mpc
            xs_interp = interpolator.interpolateState(delay, dt_mpc, xss)
            acc_interp = interpolator.interpolateLinear(delay, dt_mpc, ddqs)
            force_interp = interpolator.interpolateLinear(delay, dt_mpc, forces).reshape(
                (4, 3)
            )
            q_interp = xs_interp[:nq]
            v_interp = xs_interp[nq:]
            force_interp = [force_interp[i, :] for i in range(4)]

            q_meas, v_meas = device.measureState()
            x_measured = np.concatenate([q_meas, v_meas])

            # Use MPC only for joint position reference and initial ground reaction forces.
            # Base pose comes from measurement; base velocity uses MPC input command.
            q_interp[:7] = q_meas[:7]
            v_interp[:6] = v_world_cmd

            kino_ID.setTarget(q_interp, v_interp, acc_interp, contact_states, force_interp)
            tau_cmd = kino_ID.solve(time.time(), q_meas, v_meas)
            device.execute(tau_cmd)


if __name__ == "__main__":
    main()
