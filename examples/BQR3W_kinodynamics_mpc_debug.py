import os
import sys
import time
import numpy as np
import pinocchio as pin
import pybullet as p
import example_robot_data as erd

from simple_mpc import RobotModelHandler, RobotDataHandler, KinodynamicsOCP, MPC

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from example_robot_data.robots_loader import ROBOTS, RobotLoader


# ####### CONFIGURATION  ############
URDF_SUBPATH = "/BQR3W_description/urdf/BQR3_350_go2name.urdf"
base_joint_name = "root_joint"

if "BQR3W_go2name" not in ROBOTS:
    class BQR3WLoader(RobotLoader):
        path = "BQR3W_description"
        urdf_filename = "BQR3_350_go2name.urdf"
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

# Initial pose
base_pos = np.array([0.0, 0.0, 0.605])
base_rpy = np.array([0.0, 0.0, 0.0])
R_base = pin.rpy.rpyToMatrix(base_rpy[0], base_rpy[1], base_rpy[2])
q_base = pin.Quaternion(R_base)
q_init = robot_wrapper.model.referenceConfigurations["standing"].copy()
q_init[:3] = base_pos
q_init[3:7] = q_base.coeffs()
robot_wrapper.model.referenceConfigurations["standing"] = q_init

# Model handler
model_handler = RobotModelHandler(robot_wrapper.model, "standing", base_joint_name)
model_handler.addPointFoot("FL_wheel", base_joint_name)
model_handler.addPointFoot("FR_wheel", base_joint_name)
model_handler.addPointFoot("RL_wheel", base_joint_name)
model_handler.addPointFoot("RR_wheel", base_joint_name)
data_handler = RobotDataHandler(model_handler)

nq = model_handler.getModel().nq
nv = model_handler.getModel().nv
force_size = 3
nk = model_handler.getFeetNb()
gravity = np.array([0, 0, -9.81])

fref = np.zeros(force_size)
fref[2] = -model_handler.getMass() / nk * gravity[2]

dt_mpc = 0.01

w_basepos = [0, 0, 100, 20.0, 0.0, 30]
w_jointpos = [10.1, 10.1, 10.1, 0.0]
w_basevel = [100, 10, 10, 10, 10, 100]
w_jointvel = [0.1, 0.1, 0.1, 0.0]

model = model_handler.getModel()
wheel_joints = [
    "FL_wheel_joint",
    "FR_wheel_joint",
    "RL_wheel_joint",
    "RR_wheel_joint",
]
wheel_radius = 0.125

w_q = np.zeros(model.nv)
w_v = np.zeros(model.nv)
w_q[:6] = w_basepos
w_v[:6] = w_basevel
w_q[6:] = np.resize(np.tile(w_jointpos, 4), w_q[6:].shape[0])
w_v[6:] = np.resize(np.tile(w_jointvel, 4), w_v[6:].shape[0])

w_x = np.diag(np.concatenate((w_q, w_v)))
print("[weights] w_x diag:")
print(np.diag(w_x))
w_linforce = np.array([0.01, 0.01, 0.01])
w_u_joints = np.ones(model.nv - 6) * 1e-5
w_u = np.concatenate((w_linforce, w_linforce, w_linforce, w_linforce, w_u_joints))
w_u = np.diag(w_u)
w_cent = np.zeros((6, 6))
w_centder = np.zeros((6, 6))

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
    gravity=gravity,
    force_size=3,
    w_frame=np.array([0.0, 0.0, 0.0]),
    qmin=qmin,
    qmax=qmax,
    mu=0.8,
    Lfoot=0.01,
    Wfoot=0.01,
    kinematics_limits=False,
    force_cone=False,
    land_cstr=False,
    nonholonomic_rolling=False,
    enable_lateral_no_slip=True,
    lateral_no_slip_min_axis_norm=1e-12,
    lateral_no_slip_min_cross_norm=1e-8,
    use_vector_glide_cost=True,
    vector_glide_weight=100.0,
    vector_glide_min_omega=1e-6,
    min_wheel_distance_cstr=False,
    min_wheel_distance=1.2 * 2.0 * wheel_radius,
    min_wheel_distance_cost=False,
    w_min_wheel_distance=100.5,
    min_wheel_distance_cost_eps=1e-2,
    force_z_variance_cost=False,
    w_force_z_variance=10.0,
    joint_limit_soft_cost=False,
    w_joint_limit_soft=2.0,
    joint_limit_soft_fraction=0.5,
    soft_constraints=True,
    w_soft_contact_vel=0.0,
    w_soft_friction=0.0,
    w_soft_land=0.0,
    track_width_cstr=False,
    w_track_width=6000,
    foot_sum_cstr=False,
    w_foot_sum=6000,
    foot_sum_z_offset=wheel_radius,
    foot_height_cstr=False,
    foot_height=0.0,
)

T = 60

mpc_conf = dict(
    support_force=-model_handler.getMass() * gravity[2],
    TOL=1e-4,
    mu_init=1e-8,
    max_iters=3,
    num_threads=1,
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


def _build_mpc():
    dynproblem = KinodynamicsOCP(problem_conf, model_handler)
    dynproblem.createProblem(
        model_handler.getReferenceState(), T, force_size, gravity[2], False
    )
    mpc = MPC(mpc_conf, dynproblem)
    mpc.generateCycleHorizon([contact_phase_quadru])
    return mpc


def _spawn_robot(color_rgba):
    robot_id = p.loadURDF(
        URDF_SUBPATH, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], useFixedBase=True
    )
    local_inertia_pos = np.array(p.getDynamicsInfo(robot_id, -1)[3])
    bullet_joint_names = [
        p.getJointInfo(robot_id, i)[1].decode()
        for i in range(p.getNumJoints(robot_id))
    ]
    joint_indices = [
        bullet_joint_names.index(model_handler.getModel().names[i])
        for i in range(2, model_handler.getModel().njoints)
    ]
    for link_id in range(-1, p.getNumJoints(robot_id)):
        p.setCollisionFilterGroupMask(robot_id, link_id, 0, 0)
        p.changeVisualShape(robot_id, link_id, rgbaColor=color_rgba)
    return robot_id, local_inertia_pos, joint_indices


def _update_robot(robot_id, local_inertia_pos, joint_indices, q_ref_full):
    R_ref = pin.Quaternion(q_ref_full[3:7]).toRotationMatrix()
    offset = R_ref @ local_inertia_pos
    pos = [
        q_ref_full[0] + offset[0],
        q_ref_full[1] + offset[1],
        q_ref_full[2] + offset[2],
    ]
    p.resetBasePositionAndOrientation(robot_id, pos, q_ref_full[3:7])
    for i, j_idx in enumerate(joint_indices):
        p.resetJointState(robot_id, j_idx, q_ref_full[7 + i])


def main():
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(1.2, 60, -15, [0.6, -0.2, 0.5])
    p.setAdditionalSearchPath(erd.getModelPath(URDF_SUBPATH))

    current_id, current_local_inertia, current_joints = _spawn_robot([0.2, 0.8, 0.2, 0.7])
    terminal_id, terminal_local_inertia, terminal_joints = _spawn_robot([0.95, 0.5, 0.2, 0.35])

    # Camera control pad (sliders)
    cam_distance = 1.2
    cam_yaw = 60.0
    cam_pitch = -15.0
    cam_target = np.array([0.6, -0.2, 0.5], dtype=float)
    cam_dist_slider = p.addUserDebugParameter("cam_dist", 0.2, 5.0, cam_distance)
    cam_yaw_slider = p.addUserDebugParameter("cam_yaw", -180.0, 180.0, cam_yaw)
    cam_pitch_slider = p.addUserDebugParameter("cam_pitch", -89.0, 89.0, cam_pitch)
    cam_tx_slider = p.addUserDebugParameter("cam_target_x", -5.0, 5.0, cam_target[0])
    cam_ty_slider = p.addUserDebugParameter("cam_target_y", -5.0, 5.0, cam_target[1])
    cam_tz_slider = p.addUserDebugParameter("cam_target_z", -1.0, 3.0, cam_target[2])

    v_body_cmd = np.zeros(6)
    v_body_cmd[0] = 0.2
    forward_speed_slider = p.addUserDebugParameter("forward_speed_body", -1.0, 1.0, v_body_cmd[0])
    yaw_rate_slider = p.addUserDebugParameter("yaw_rate_body", -1.0, 1.0, v_body_cmd[5])

    x0_fixed = model_handler.getReferenceState().copy()
    x_refs = []
    v_world_ref = np.zeros(6)
    last_update = 0.0

    while True:
        now = time.time()
        if now - last_update < 0.1:
            continue
        last_update = now

        v_body_cmd[0] = p.readUserDebugParameter(forward_speed_slider)
        v_body_cmd[5] = p.readUserDebugParameter(yaw_rate_slider)

        cam_dist_new = p.readUserDebugParameter(cam_dist_slider)
        cam_yaw_new = p.readUserDebugParameter(cam_yaw_slider)
        cam_pitch_new = p.readUserDebugParameter(cam_pitch_slider)
        cam_target_new = np.array(
            [
                p.readUserDebugParameter(cam_tx_slider),
                p.readUserDebugParameter(cam_ty_slider),
                p.readUserDebugParameter(cam_tz_slider),
            ],
            dtype=float,
        )
        if (
            abs(cam_dist_new - cam_distance) > 1e-6
            or abs(cam_yaw_new - cam_yaw) > 1e-6
            or abs(cam_pitch_new - cam_pitch) > 1e-6
            or np.linalg.norm(cam_target_new - cam_target) > 1e-6
        ):
            cam_distance = cam_dist_new
            cam_yaw = cam_yaw_new
            cam_pitch = cam_pitch_new
            cam_target = cam_target_new
            p.resetDebugVisualizerCamera(
                cam_distance, cam_yaw, cam_pitch, cam_target.tolist()
            )

        base_R_world = pin.Quaternion(x0_fixed[3:7]).toRotationMatrix()
        yaw = np.arctan2(base_R_world[1, 0], base_R_world[0, 0])
        v_world_ref[:] = 0.0
        v_world_ref[0] = v_body_cmd[0] * np.cos(yaw)
        v_world_ref[1] = v_body_cmd[0] * np.sin(yaw)
        v_world_ref[5] = v_body_cmd[5]

        x_refs.clear()
        x_t = float(x0_fixed[0])
        y_t = float(x0_fixed[1])
        yaw_t = float(yaw)
        for t_idx in range(T):
            yaw_t = yaw_t + v_body_cmd[5] * dt_mpc
            x_t = x_t + v_body_cmd[0] * np.cos(yaw_t) * dt_mpc
            y_t = y_t + v_body_cmd[0] * np.sin(yaw_t) * dt_mpc
            R_ref = pin.rpy.rpyToMatrix(0.0, 0.0, yaw_t)
            q_ref = pin.Quaternion(R_ref).coeffs()
            x_ref = model_handler.getReferenceState().copy()
            x_ref[0] = x_t
            x_ref[1] = y_t
            x_ref[3:7] = q_ref
            x_ref[nq : nq + 6] = v_world_ref
            x_refs.append(x_ref)

        mpc = _build_mpc()
        mpc.velocity_base = v_world_ref
        for t_idx, x_ref in enumerate(x_refs):
            mpc.ocp_handler.setReferenceState(t_idx, x_ref)

        mpc.iterate(x0_fixed)

        if len(mpc.xs) > 0:
            _update_robot(current_id, current_local_inertia, current_joints, mpc.xs[0][:nq])
            _update_robot(terminal_id, terminal_local_inertia, terminal_joints, mpc.xs[-1][:nq])

        # Keep the MPC prediction anchored at the initial state.


if __name__ == "__main__":
    main()
