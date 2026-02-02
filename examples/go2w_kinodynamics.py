import numpy as np
import os
import sys
from bullet_robot import BulletRobot
from simple_mpc import (
    RobotModelHandler,
    RobotDataHandler,
    KinodynamicsOCP,
    MPC,
    Interpolator,
    KinodynamicsID,
    KinodynamicsIDSettings,
)
import example_robot_data as erd
import pinocchio as pin
import pybullet as p
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry_solver import GeometryAdaptor


def format_joints_4x4(vec):
    if vec.size == 16:
        return vec.reshape(4, 4)
    return vec
from example_robot_data.robots_loader import ROBOTS, RobotLoader
import time
import copy

# ####### CONFIGURATION  ############
# Load robot
URDF_SUBPATH = "/go2w_description/urdf/go2w.urdf"
base_joint_name = "root_joint"
if "go2w" not in ROBOTS:
    class Go2WLoader(RobotLoader):
        path = "go2w_description"
        urdf_filename = "go2w.urdf"
        urdf_subpath = "urdf"
        srdf_filename = "go2w.srdf"
        ref_posture = "standing"
        free_flyer = True

    ROBOTS["go2w"] = Go2WLoader
robot_wrapper = erd.load("go2w")
if "standing" not in robot_wrapper.model.referenceConfigurations:
    robot_wrapper.model.referenceConfigurations["standing"] = pin.neutral(
        robot_wrapper.model
    )
# 手动设置初始位姿：用欧拉角计算四元数
base_pos = np.array([0.0, 0.0, 0.305])
base_rpy = np.array([0.0, 0.0, -0.0])  # roll, pitch, yaw
R_base = pin.rpy.rpyToMatrix(base_rpy[0], base_rpy[1], base_rpy[2])
q_base = pin.Quaternion(R_base)
q_init = robot_wrapper.model.referenceConfigurations["standing"].copy()
q_init[:3] = base_pos
q_init[3:7] = q_base.coeffs()
robot_wrapper.model.referenceConfigurations["standing"] = q_init

# Create Model and Data handler
model_handler = RobotModelHandler(robot_wrapper.model, "standing", base_joint_name)
model_handler.addPointFoot("FL_wheel", base_joint_name)
model_handler.addPointFoot("FR_wheel", base_joint_name)
model_handler.addPointFoot("RL_wheel", base_joint_name)
model_handler.addPointFoot("RR_wheel", base_joint_name)
data_handler = RobotDataHandler(model_handler)

nq = model_handler.getModel().nq
nv = model_handler.getModel().nv
nu = nv - 6
nf = 12
force_size = 3
nk = model_handler.getFeetNb()
gravity = np.array([0, 0, -9.81])
fref = np.zeros(force_size)
fref[2] = -model_handler.getMass() / nk * gravity[2]
u0 = np.concatenate((fref, fref, fref, fref, np.zeros(model_handler.getModel().nv - 6)))
dt_mpc = 0.01

w_basepos = [0, 0, 100, 20.0, 20.0, 0]
w_legpos = [10.1, 10.1, 10.1]

w_basevel = [10, 10, 10, 10, 10, 10]
w_legvel = [1.1, 1.1, 1.1]
model = model_handler.getModel()
wheel_joints = [
    "FL_wheel_joint",
    "FR_wheel_joint",
    "RL_wheel_joint",
    "RR_wheel_joint",
]
wheel_q_indices = []
wheel_v_indices = []
wheel_u_indices = []
wheel_meas_indices = []
wheel_vel_limit = 3.0 * 100.0 * 2.0 * np.pi / 60.0
wheel_target_vel = 10.0 * 2.0 * np.pi / 60.0
wheel_radius = 0.0762
joint_names_complete = list(model.names)[2:]
for joint_name in wheel_joints:
    joint_id = model.getJointId(joint_name)
    idx_q = model.joints[joint_id].idx_q
    idx_v = model.joints[joint_id].idx_v
    nq_joint = model.joints[joint_id].nq
    nv_joint = model.joints[joint_id].nv
    wheel_q_indices.extend(range(idx_q, idx_q + nq_joint))
    wheel_v_indices.extend(range(idx_v, idx_v + nv_joint))
    wheel_u_indices.extend(range(idx_v - 6, idx_v - 6 + nv_joint))
    wheel_meas_indices.append(6 + joint_names_complete.index(joint_name))
    model.velocityLimit[idx_v : idx_v + nv_joint] = wheel_vel_limit

w_q = np.zeros(model.nv)
w_v = np.zeros(model.nv)
w_q[:6] = w_basepos
w_v[:6] = w_basevel
w_q[6:] = w_legpos[0]
w_v[6:] = w_legvel[0]
for joint_name in wheel_joints:
    joint_id = model.getJointId(joint_name)
    idx_v = model.joints[joint_id].idx_v
    nv_joint = model.joints[joint_id].nv
    w_q[idx_v : idx_v + nv_joint] = 0.0

w_x = np.diag(np.concatenate((w_q, w_v)))
w_linforce = np.array([0.01, 0.01, 0.01])
w_u_joints = np.ones(model.nv - 6) * 1e-5
for joint_name in wheel_joints:
    joint_id = model.getJointId(joint_name)
    idx_v = model.joints[joint_id].idx_v
    nv_joint = model.joints[joint_id].nv
w_u = np.concatenate((w_linforce, w_linforce, w_linforce, w_linforce, w_u_joints))
w_u = np.diag(w_u)
w_LFRF = 6000
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
    vector_glide_weight=100.0,
    vector_glide_min_omega=1e-6,
    min_wheel_distance_cstr=False,
    min_wheel_distance=1.2 * 2.0 * wheel_radius,
    min_wheel_distance_cost=True,
    w_min_wheel_distance=0.05,
    min_wheel_distance_cost_eps=1e-3,
    force_z_variance_cost=True,
    w_force_z_variance=10.0,
    joint_limit_soft_cost=True,
    w_joint_limit_soft=2.0,
    joint_limit_soft_fraction=0.5,
    soft_constraints=True,
    w_soft_contact_vel=20.0,
    w_soft_friction=0.0,
    w_soft_land=0.0,
    track_width_cstr=True,
    w_track_width=w_LFRF,
    foot_sum_cstr=False,
    w_foot_sum=w_LFRF,
    foot_sum_z_offset=wheel_radius,
)
T = 50

dynproblem = KinodynamicsOCP(problem_conf, model_handler)
dynproblem.createProblem(
    model_handler.getReferenceState(), T, force_size, gravity[2], False
)

T_ds = 10
T_ss = 30

mpc_conf = dict(
    support_force=-model_handler.getMass() * gravity[2],
    TOL=1e-4,
    mu_init=1e-8,
    max_iters=1,
    num_threads=8,
    swing_apex=0.15,
    T_fly=T_ss,
    T_contact=T_ds,
    timestep=dt_mpc,
    update_contact_ref=True,
)

mpc = MPC(mpc_conf, dynproblem)

debug_id = False
debug_mpc_costs = True

""" Define contact sequence throughout horizon"""
contact_phase_quadru = {
    "FL_wheel": True,
    "FR_wheel": True,
    "RL_wheel": True,
    "RR_wheel": True,
}
contact_phase_lift_FL = {
    "FL_wheel": False,
    "FR_wheel": True,
    "RL_wheel": True,
    "RR_wheel": False,
}
contact_phase_lift_FR = {
    "FL_wheel": True,
    "FR_wheel": False,
    "RL_wheel": False,
    "RR_wheel": True,
}
contact_phase_lift = {
    "FL_wheel": False,
    "FR_wheel": False,
    "RL_wheel": False,
    "RR_wheel": False,
}
# contact_phases = [contact_phase_quadru] * T_ds
# contact_phases += [contact_phase_lift_FL] * T_ss
# contact_phases += [contact_phase_quadru] * T_ds
# contact_phases += [contact_phase_lift_FR] * T_ss

contact_phases = [contact_phase_quadru]

mpc.generateCycleHorizon(contact_phases)

""" Interpolation """
N_simu = 10  # Number of substep the simulation does between two MPC computation
dt_simu = dt_mpc / N_simu
interpolator = Interpolator(model_handler.getModel())

""" Inverse Dynamics """
kino_ID_settings = KinodynamicsIDSettings()
kino_ID_settings.kp_base = 15.0
kino_ID_settings.kp_posture = 30.0
kino_ID_settings.kp_contact = 0
.0
kino_ID_settings.w_base = 100.0
kino_ID_settings.w_posture = 10.0
kino_ID_settings.w_contact_force = 1.0
kino_ID_settings.w_contact_motion = 0.0
# Wheel-specific posture gains/weight (if equal to the above, behavior remains unchanged)
kino_ID_settings.kp_posture_wheel = 0.001
kino_ID_settings.w_posture_wheel = 0.001
# Wheel physical settings
kino_ID_settings.wheel_radius = wheel_radius
kino_ID_settings.ff_wheel_scale = 1.0
# Disable legacy non-holonomic rolling constraints (kept for compatibility)
kino_ID_settings.enable_nonholonomic = False
kino_ID_settings.enable_lateral_no_slip = False
kino_ID_settings.lateral_no_slip_use_bounds = True
kino_ID_settings.lateral_no_slip_lower = -3
kino_ID_settings.lateral_no_slip_upper = 3
kino_ID_settings.use_vector_glide_cost = False
kino_ID_settings.vector_glide_weight = 1.0
kino_ID = KinodynamicsID(model_handler, dt_simu, kino_ID_settings)
kino_ID.addLateralNoSlipConstraint("FL_wheel")
kino_ID.addLateralNoSlipConstraint("FR_wheel")
kino_ID.addLateralNoSlipConstraint("RL_wheel")
kino_ID.addLateralNoSlipConstraint("RR_wheel")

""" Initialize simulation"""
device = BulletRobot(
    model_handler.getModel().names,
    erd.getModelPath(URDF_SUBPATH),
    URDF_SUBPATH,
    dt_simu,
    model_handler.getModel(),
    model_handler.getReferenceState()[:3],
)

device.initializeJoints(
    model_handler.getReferenceState()[: model_handler.getModel().nq]
)
device.changeCamera(1.0, 60, -15, [0.6, -0.2, 0.5])

# MPC terminal state visualization (ghost robot)
target_model_path = erd.getModelPath(URDF_SUBPATH)
p.setAdditionalSearchPath(target_model_path)


def _spawn_ghost(color_rgba):
    ghost_id = p.loadURDF(
        URDF_SUBPATH, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], useFixedBase=True
    )
    local_inertia_pos = np.array(p.getDynamicsInfo(ghost_id, -1)[3])
    bullet_joint_names = [
        p.getJointInfo(ghost_id, i)[1].decode()
        for i in range(p.getNumJoints(ghost_id))
    ]
    joint_indices = [
        bullet_joint_names.index(model_handler.getModel().names[i])
        for i in range(2, model_handler.getModel().njoints)
    ]
    for link_id in range(-1, p.getNumJoints(ghost_id)):
        p.setCollisionFilterGroupMask(ghost_id, link_id, 0, 0)
        p.changeVisualShape(ghost_id, link_id, rgbaColor=color_rgba)
    return ghost_id, local_inertia_pos, joint_indices


def _update_ghost(robot_id, local_inertia_pos, joint_indices, q_ref_full):
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


target_robot_id, target_local_inertia_pos, target_joint_indices = _spawn_ghost([0.2, 0.9, 0.2, 0.35])
terminal_target_robot_id, terminal_target_local_inertia_pos, terminal_target_joint_indices = _spawn_ghost([0.1, 0.6, 0.1, 0.35])
terminal_robot_id, terminal_local_inertia_pos, terminal_joint_indices = _spawn_ghost([0.95, 0.5, 0.2, 0.35])

# 机身系速度指令（来自可视化滑条）
v_body_cmd = np.zeros(6)
v_world_cmd = np.zeros(6)
v_body_cmd[0] = wheel_target_vel * wheel_radius

wheel_link_names = ["FL_wheel", "FR_wheel", "RL_wheel", "RR_wheel"]
wheel_link_ids = {}
for link_id in range(p.getNumJoints(device.robotId)):
    link_name = p.getJointInfo(device.robotId, link_id)[12].decode()
    if link_name in wheel_link_names:
        wheel_link_ids[link_name] = link_id

x_measured = None
q_meas, v_meas = device.measureState()
base_R_world = pin.Quaternion(q_meas[3:7]).toRotationMatrix()
yaw = np.arctan2(base_R_world[1, 0], base_R_world[0, 0])
yaw_accum = yaw
last_yaw = yaw
cy = np.cos(yaw)
sy = np.sin(yaw)
R_yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
v_world_cmd[:3] = R_yaw @ v_body_cmd[:3]
v_world_cmd[3:6] = R_yaw @ v_body_cmd[3:6]
mpc.velocity_base = v_world_cmd
x_measured = np.concatenate([q_meas, v_meas])

device.showQuadrupedFeet(
    mpc.getDataHandler().getFootPose(mpc.getModelHandler().getFootNb("FL_wheel")),
    mpc.getDataHandler().getFootPose(mpc.getModelHandler().getFootNb("FR_wheel")),
    mpc.getDataHandler().getFootPose(mpc.getModelHandler().getFootNb("RL_wheel")),
    mpc.getDataHandler().getFootPose(mpc.getModelHandler().getFootNb("RR_wheel")),
)

force_FL = []
force_FR = []
force_RL = []
force_RR = []
FL_measured = []
FR_measured = []
RL_measured = []
RR_measured = []
FL_references = []
FR_references = []
RL_references = []
RR_references = []
x_multibody = []
u_multibody = []
u_riccati = []
com_measured = []
solve_time = []
L_measured = []

mpc.velocity_base = v_world_cmd
forward_speed_slider = p.addUserDebugParameter("forward_speed_body", -1.0, 1.0, v_body_cmd[0])
yaw_rate_slider = p.addUserDebugParameter("yaw_rate_body", -1.0, 1.0, 0.0)
# Simulation control: use keyboard keys instead of sliders
# 'p' toggles pause/resume, 'n' single-steps one outer iteration
paused = False  # start paused as requested
prev_step_pressed = False
wheel_target_vel_prev = v_body_cmd[0] / wheel_radius
wheel_pos_ref = None
base_pos_ref = None
q_meas, v_meas = device.measureState()
current_ref_pos_x = q_meas[0]
current_ref_pos_y = q_meas[1]
initial_height = q_meas[2]
base_R_world = pin.Quaternion(q_meas[3:7]).toRotationMatrix()
current_ref_yaw = np.arctan2(base_R_world[1, 0], base_R_world[0, 0])
# Geometry solver (Ackermann-like kinematic alignment)
foot_pos_base = np.array(
    [
        [0.3, 0.15, -initial_height],
        [0.3, -0.15, -initial_height],
        [-0.3, 0.15, -initial_height],
        [-0.3, -0.15, -initial_height],
    ]
)
axle_dir_base = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
)
geo_solver = GeometryAdaptor(foot_pos_base, axle_dir_base)
for step in range(300000):
    v_body_cmd[0] = p.readUserDebugParameter(forward_speed_slider)
    v_body_cmd[5] = p.readUserDebugParameter(yaw_rate_slider)
    # 只允许机身系 x 方向速度与 yaw 角速度，禁止横向/竖向/翻滚俯仰速度
    v_body_cmd[1] = 0.0
    v_body_cmd[2] = 0.0
    v_body_cmd[3] = 0.0
    v_body_cmd[4] = 0.0
    # 用当前机身姿态把机身系前进速度投影到世界系（只取 yaw）
    q_meas, v_meas = device.measureState()
    base_R_world = pin.Quaternion(q_meas[3:7]).toRotationMatrix()
    yaw = np.arctan2(base_R_world[1, 0], base_R_world[0, 0])
    dyaw = yaw - last_yaw
    if dyaw > np.pi:
        dyaw -= 2.0 * np.pi
    elif dyaw < -np.pi:
        dyaw += 2.0 * np.pi
    yaw_accum += dyaw
    last_yaw = yaw
    yaw_use = yaw_accum

    v_world_cmd[:] = 0.0
    v_world_cmd[0] = v_body_cmd[0] * np.cos(yaw_use)
    v_world_cmd[1] = v_body_cmd[0] * np.sin(yaw_use)
    v_world_cmd[5] = v_body_cmd[5]
    mpc.velocity_base = v_world_cmd
    print("base q", q_meas[:7])
    print("yaw(deg)", yaw_use * 180.0 / np.pi)
    print("cmd body", v_body_cmd)
    print("cmd yaw-only world", v_world_cmd)
    wheel_target_vel = v_body_cmd[0] / wheel_radius
    # Keyboard control: 'p' toggle pause, 'n' single-step
    events = p.getKeyboardEvents()
    if ord('p') in events and events[ord('p')] & p.KEY_WAS_TRIGGERED:
        paused = not paused
        print("[UI] paused =", paused)
    step_pressed = False
    if ord('n') in events and events[ord('n')] & p.KEY_WAS_TRIGGERED:
        step_pressed = True
    while paused and not step_pressed:
        events = p.getKeyboardEvents()
        if ord('p') in events and events[ord('p')] & p.KEY_WAS_TRIGGERED:
            paused = not paused
            print("[UI] paused =", paused)
        if ord('n') in events and events[ord('n')] & p.KEY_WAS_TRIGGERED:
            step_pressed = True
        time.sleep(0.01)
    if paused and not step_pressed:
        continue
    # print("Time " + str(step))
    land_LF = mpc.getFootLandCycle("FL_wheel")
    land_RF = mpc.getFootLandCycle("RL_wheel")
    takeoff_LF = mpc.getFootTakeoffCycle("FL_wheel")
    takeoff_RF = mpc.getFootTakeoffCycle("RL_wheel")
    print(
        "takeoff_RF = " + str(takeoff_RF) + ", landing_RF = ",
        str(land_RF) + ", takeoff_LF = " + str(takeoff_LF) + ", landing_LF = ",
        str(land_LF),
    )
    start = time.time()
    mpc.iterate(x_measured)
    end = time.time()
    solve_time.append(end - start)

    try:
        x_ref0 = np.asarray(mpc.ocp_handler.getReferenceState(0)).copy()
    except Exception:
        x_ref0 = model_handler.getReferenceState().copy()
    try:
        x_refT = np.asarray(mpc.ocp_handler.getReferenceState(T - 1)).copy()
    except Exception:
        x_refT = x_ref0
    _update_ghost(
        target_robot_id, target_local_inertia_pos, target_joint_indices, x_ref0[:nq]
    )
    _update_ghost(
        terminal_target_robot_id,
        terminal_target_local_inertia_pos,
        terminal_target_joint_indices,
        x_refT[:nq],
    )

    if len(mpc.xs) > 0:
        q_last = np.asarray(mpc.xs[-1][:nq]).copy()
        _update_ghost(
            terminal_robot_id, terminal_local_inertia_pos, terminal_joint_indices, q_last
        )

    force_FL.append(mpc.us[0][:3])
    force_FR.append(mpc.us[0][3:6])
    force_RL.append(mpc.us[0][6:9])
    force_RR.append(mpc.us[0][9:12])

    FL_measured.append(
        mpc.getDataHandler()
        .getFootPose(mpc.getModelHandler().getFootNb("FL_wheel"))
        .translation
    )
    FR_measured.append(
        mpc.getDataHandler()
        .getFootPose(mpc.getModelHandler().getFootNb("FR_wheel"))
        .translation
    )
    RL_measured.append(
        mpc.getDataHandler()
        .getFootPose(mpc.getModelHandler().getFootNb("RL_wheel"))
        .translation
    )
    RR_measured.append(
        mpc.getDataHandler()
        .getFootPose(mpc.getModelHandler().getFootNb("RR_wheel"))
        .translation
    )
    FL_references.append(mpc.getReferencePose(0, "FL_wheel").translation)
    FR_references.append(mpc.getReferencePose(0, "FR_wheel").translation)
    RL_references.append(mpc.getReferencePose(0, "RL_wheel").translation)
    RR_references.append(mpc.getReferencePose(0, "RR_wheel").translation)
    com_measured.append(mpc.getDataHandler().getData().com[0].copy())
    L_measured.append(mpc.getDataHandler().getData().hg.angular.copy())

    a0 = mpc.getStateDerivative(0)[nv:].copy()
    a1 = mpc.getStateDerivative(1)[nv:].copy()

    a0[6:] = mpc.us[0][nk * force_size :]
    a1[6:] = mpc.us[1][nk * force_size :]
    forces0 = mpc.us[0][: nk * force_size]
    forces1 = mpc.us[1][: nk * force_size]
    contact_states = mpc.ocp_handler.getContactState(0)

    forces = [forces0, forces1]
    ddqs = [a0, a1]
    xss = [mpc.xs[0], mpc.xs[1]]
    uss = [mpc.us[0], mpc.us[1]]

    device.moveQuadrupedFeet(
        mpc.getReferencePose(0, "FL_wheel").translation,
        mpc.getReferencePose(0, "FR_wheel").translation,
        mpc.getReferencePose(0, "RL_wheel").translation,
        mpc.getReferencePose(0, "RR_wheel").translation,
    )

    for sub_step in range(N_simu):
        t = step * dt_mpc + sub_step * dt_simu

        delay = sub_step / float(N_simu) * dt_mpc
        xs_interp = interpolator.interpolateState(delay, dt_mpc, xss)
        acc_interp = interpolator.interpolateLinear(delay, dt_mpc, ddqs)
        force_interp = interpolator.interpolateLinear(delay, dt_mpc, forces).reshape(
            (4, 3)
        )

        q_interp = xs_interp[: mpc.getModelHandler().getModel().nq]
        v_interp = xs_interp[mpc.getModelHandler().getModel().nq :]
        force_interp = [force_interp[i, :] for i in range(4)]

        user_v_cmd = v_body_cmd[0]
        user_w_cmd = v_body_cmd[5]
        current_ref_yaw += user_w_cmd * dt_simu
        v_world_x = user_v_cmd * np.cos(current_ref_yaw)
        v_world_y = user_v_cmd * np.sin(current_ref_yaw)
        current_ref_pos_x += v_world_x * dt_simu
        current_ref_pos_y += v_world_y * dt_simu

        v_interp[:6] = 0.0
        v_interp[0] = v_world_x
        v_interp[1] = v_world_y
        v_interp[2] = 0.0
        v_interp[3] = 0.0
        v_interp[4] = 0.0
        v_interp[5] = user_w_cmd

        q_interp[0] = current_ref_pos_x
        q_interp[1] = current_ref_pos_y
        q_interp[2] = initial_height
        geo = geo_solver.solve(user_v_cmd, user_w_cmd, yaw=current_ref_yaw)
        q_roll = geo.roll
        q_pitch = geo.pitch
        q_yaw = current_ref_yaw
        qx = np.sin(q_roll * 0.5) * np.cos(q_pitch * 0.5) * np.cos(q_yaw * 0.5) - np.cos(q_roll * 0.5) * np.sin(q_pitch * 0.5) * np.sin(q_yaw * 0.5)
        qy = np.cos(q_roll * 0.5) * np.sin(q_pitch * 0.5) * np.cos(q_yaw * 0.5) + np.sin(q_roll * 0.5) * np.cos(q_pitch * 0.5) * np.sin(q_yaw * 0.5)
        qz = np.cos(q_roll * 0.5) * np.cos(q_pitch * 0.5) * np.sin(q_yaw * 0.5) - np.sin(q_roll * 0.5) * np.sin(q_pitch * 0.5) * np.cos(q_yaw * 0.5)
        qw = np.cos(q_roll * 0.5) * np.cos(q_pitch * 0.5) * np.cos(q_yaw * 0.5) + np.sin(q_roll * 0.5) * np.sin(q_pitch * 0.5) * np.sin(q_yaw * 0.5)
        q_interp[3] = qx
        q_interp[4] = qy
        q_interp[5] = qz
        q_interp[6] = qw

        # Optional: apply toe offsets to hip joints (verify indices for your URDF)
        hip_joint_indices = [7, 10, 13, 16]
        for i, idx in enumerate(hip_joint_indices):
            if idx < q_interp.shape[0]:
                q_interp[idx] += geo.toe_offsets[i]

        q_meas, v_meas = device.measureState()
        base_R_world = pin.Quaternion(q_meas[3:7]).toRotationMatrix()
        x_measured = np.concatenate([q_meas, v_meas])

        # ID input uses MPC-interpolated state/force directly
        kino_ID.setTarget(q_interp, v_interp, acc_interp, contact_states, force_interp)
        tau_cmd = kino_ID.solve(t, q_meas, v_meas)

        contact_forces = {name: np.zeros(3) for name in wheel_link_names}
        for cp in p.getContactPoints(bodyA=device.robotId):
            link_a = cp[3]
            if link_a in wheel_link_ids.values():
                normal = np.array(cp[7])
                normal_force = cp[9]
                fric1 = cp[10]
                fric_dir1 = np.array(cp[11])
                fric2 = cp[12]
                fric_dir2 = np.array(cp[13])
                f = normal * normal_force + fric_dir1 * fric1 + fric_dir2 * fric2
                for name, lid in wheel_link_ids.items():
                    if lid == link_a:
                        contact_forces[name] += f
                        break

        nq = mpc.getModelHandler().getModel().nq
        nv = mpc.getModelHandler().getModel().nv

        if debug_id:
            print("ID input:")
            print("base q", q_interp[:7])
            print("base v", v_interp[:6])
            print("base a", acc_interp[:6])
            print("joints q", format_joints_4x4(q_interp[7:nq]))
            print("joints v", format_joints_4x4(v_interp[6:nv]))
            print("joints a", format_joints_4x4(acc_interp[6:nv]))
            print("meas base q", q_meas[:7])
            print("meas base v", v_meas[:6])
            print("meas joints q", format_joints_4x4(q_meas[7:nq]))
            print("meas joints v", format_joints_4x4(v_meas[6:nv]))
            print("")
            print("MPC output:")
            print("x0 base q", xss[0][:7])
            print("x0 base v", xss[0][nq : nq + 6])
            print("x0 joints q", format_joints_4x4(xss[0][7:nq]))
            print("x0 joints v", format_joints_4x4(xss[0][nq + 6 : nq + nv]))
            print("u0 force", forces0.reshape(4, 3))
            print("u0 joints acc", format_joints_4x4(a0[6:]))
            print("tau_cmd", tau_cmd)
            print("contact force meas", [contact_forces[name] for name in wheel_link_names])
            print("contact force mpc", force_interp)
            print("")

        if debug_mpc_costs and sub_step == 0:
            x0 = xss[0]
            u0 = mpc.us[0]
            try:
                x_ref = mpc.ocp_handler.getReferenceState(0)
                u_ref = mpc.ocp_handler.getReferenceControl(0)
            except Exception:
                x_ref = x0
                u_ref = u0

            q0 = x0[:nq]
            v0 = x0[nq:]
            q_ref = x_ref[:nq]
            v_ref = x_ref[nq:]
            dx_q = pin.difference(model, q_ref, q0)
            dx_v = v0 - v_ref
            dx = np.concatenate([dx_q, dx_v])
            du = u0 - u_ref
            state_cost = float(dx.T @ w_x @ dx)
            control_cost = float(du.T @ w_u @ du)

            ndx = dx.shape[0]
            nv = model.nv
            idx_base_pos = np.arange(0, 3)
            idx_base_ori = np.arange(3, 6)
            idx_joint_pos = np.arange(6, nv)
            idx_base_linvel = np.arange(nv + 0, nv + 3)
            idx_base_angvel = np.arange(nv + 3, nv + 6)
            idx_joint_vel = np.arange(nv + 6, 2 * nv)

            def quad_cost(vec, W, idxs):
                if idxs.size == 0:
                    return 0.0
                sub = vec[idxs]
                Wsub = W[np.ix_(idxs, idxs)]
                return float(sub.T @ Wsub @ sub)

            state_cost_base_pos = quad_cost(dx, w_x, idx_base_pos)
            state_cost_base_ori = quad_cost(dx, w_x, idx_base_ori)
            state_cost_joint_pos = quad_cost(dx, w_x, idx_joint_pos)
            state_cost_base_linvel = quad_cost(dx, w_x, idx_base_linvel)
            state_cost_base_angvel = quad_cost(dx, w_x, idx_base_angvel)
            state_cost_joint_vel = quad_cost(dx, w_x, idx_joint_vel)

            model = mpc.getModelHandler().getModel()
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
                foot_ref = mpc.getReferencePose(0, name).translation
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
                base_frame_id = mpc.getModelHandler().getBaseFrameId()
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
                        initial_height - problem_conf.get("foot_sum_z_offset", 0.0),
                    ]
                )
                foot_sum_res = -mean_base - target_sum
                foot_sum_cost = float(
                    problem_conf.get("w_foot_sum", 0.0) * (foot_sum_res.T @ foot_sum_res)
                )

            track_width_res = None
            track_width_cost = 0.0
            if problem_conf.get("track_width_cstr", False) and len(wheel_link_names) >= 4:
                base_frame_id = mpc.getModelHandler().getBaseFrameId()
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

                ref_state = mpc.getModelHandler().getReferenceState()
                data_ref = pin.Data(model)
                pin.forwardKinematics(model, data_ref, ref_state[:nq], ref_state[nq:])
                pin.updateFramePlacements(model, data_ref)
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
                ref_state = mpc.getModelHandler().getReferenceState()
                data_ref = pin.Data(model)
                pin.forwardKinematics(model, data_ref, ref_state[:nq], ref_state[nq:])
                pin.updateFramePlacements(model, data_ref)
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
                    base_frame_id = mpc.getModelHandler().getBaseFrameId()
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

            force_z_variance_cost = 0.0
            force_z_variance_res = None
            if problem_conf.get("force_z_variance_cost", False) and force_size >= 3:
                fz = []
                for i in range(len(wheel_link_names)):
                    if contact_states[i]:
                        fz.append(u0[i * force_size + 2])
                if fz:
                    mean_fz = float(np.mean(fz))
                    var_fz = float(np.mean((np.array(fz) - mean_fz) ** 2))
                    force_z_variance_res = var_fz
                    force_z_variance_cost = float(
                        problem_conf.get("w_force_z_variance", 0.0) * var_fz
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
                    mid = 0.5 * (qmin_i + qmax_i)
                    half = 0.5 * rng
                    dead = frac * half
                    dist = abs(q_i - mid)
                    if dist <= dead:
                        continue
                    s = (dist - dead) / max(half - dead, 1e-9)
                    s = min(1.0, max(0.0, s))
                    res[i] = s * s
                joint_limit_soft_res = res
                joint_limit_soft_cost = float(
                    problem_conf.get("w_joint_limit_soft", 0.0) * (res.T @ res)
                )

            print("[MPC costs]")
            print("  state_cost", state_cost)
            print("  state_cost_base_pos", state_cost_base_pos)
            print("  state_cost_base_ori", state_cost_base_ori)
            print("  state_cost_joint_pos", state_cost_joint_pos)
            print("  state_cost_base_linvel", state_cost_base_linvel)
            print("  state_cost_base_angvel", state_cost_base_angvel)
            print("  state_cost_joint_vel", state_cost_joint_vel)
            print("  control_cost", control_cost)
            print("  foot_cost", foot_cost)
            print("  centroidal_cost", centroidal_cost)
            print("  centroidal_cost_lin", centroidal_cost_lin)
            print("  centroidal_cost_ang", centroidal_cost_ang)
            print("  centroidal_derivative_cost", centroidal_derivative_cost)
            if centroidal_derivative_res is not None:
                print("  centroidal_derivative_res", centroidal_derivative_res)
            print("  soft_contact_vel_cost", soft_contact_vel_cost)
            print("  soft_contact_vel_res", soft_contact_vel_res)
            print("  soft_friction_cost", soft_friction_cost)
            print("  soft_friction_res", soft_friction_res)
            if foot_sum_res is not None:
                print("  foot_sum_cost", foot_sum_cost)
                print("  foot_sum_res", foot_sum_res)
            if track_width_res is not None:
                print("  track_width_cost", track_width_cost)
                print("  track_width_res", track_width_res)
            if min_wheel_distance_res is not None:
                print("  min_wheel_distance_cost", min_wheel_distance_cost)
                print("  min_wheel_distance_res", min_wheel_distance_res)
            if vector_glide_res:
                print("  vector_glide_cost", vector_glide_cost)
                print("  vector_glide_res", vector_glide_res)
            if force_z_variance_res is not None:
                print("  force_z_variance_cost", force_z_variance_cost)
                print("  force_z_variance_res", force_z_variance_res)
            if joint_limit_soft_res is not None:
                print("  joint_limit_soft_cost", joint_limit_soft_cost)
                print("  joint_limit_soft_res", joint_limit_soft_res)
            print("")

        device.execute(tau_cmd)
        u_multibody.append(copy.deepcopy(tau_cmd))
        x_multibody.append(x_measured)


force_FL = np.array(force_FL)
force_FR = np.array(force_FR)
force_RL = np.array(force_RL)
force_RR = np.array(force_RR)
solve_time = np.array(solve_time)
FL_measured = np.array(FL_measured)
FR_measured = np.array(FR_measured)
RL_measured = np.array(RL_measured)
RR_measured = np.array(RR_measured)
FL_references = np.array(FL_references)
FR_references = np.array(FR_references)
RL_references = np.array(RL_references)
RR_references = np.array(RR_references)
com_measured = np.array(com_measured)
L_measured = np.array(L_measured)

""" save_trajectory(x_multibody, u_multibody, com_measured, force_FL, force_FR, force_RL, force_RR, solve_time,
                FL_measured, FR_measured, RL_measured, RR_measured,
                FL_references, FR_references, RL_references, RR_references, L_measured, "kinodynamics") """
