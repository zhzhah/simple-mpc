import numpy as np
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

w_basepos = [0, 0, 100, 10, 10, 0]
w_legpos = [1, 1, 1]

w_basevel = [10, 10, 10, 10, 10, 10]
w_legvel = [0.1, 0.1, 0.1]
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
wheel_vel_limit = 100.0 * 2.0 * np.pi / 60.0
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

w_x = np.diag(np.concatenate((w_q, w_v)))
w_linforce = np.array([0.01, 0.01, 0.01])
w_u_joints = np.ones(model.nv - 6) * 1e-5
for joint_name in wheel_joints:
    joint_id = model.getJointId(joint_name)
    idx_v = model.joints[joint_id].idx_v
    nv_joint = model.joints[joint_id].nv
w_u = np.concatenate((w_linforce, w_linforce, w_linforce, w_linforce, w_u_joints))
w_u = np.diag(w_u)
w_LFRF = 2000
w_cent_lin = np.array([0.0, 0.0, 1])
w_cent_ang = np.array([0.1, 0.1, 10])
w_cent = np.diag(np.concatenate((w_cent_lin, w_cent_ang)))
w_centder_lin = np.ones(3) * 0.0
w_centder_ang = np.ones(3) * 0.1
w_centder = np.diag(np.concatenate((w_centder_lin, w_centder_ang)))

qmin = model_handler.getModel().lowerPositionLimit[7:]
qmax = model_handler.getModel().upperPositionLimit[7:]

problem_conf = dict(
    timestep=dt_mpc,
    w_x=w_x,
    w_u=w_u,
    w_cent=w_cent,
    w_centder=w_centder,
    gravity=gravity,
    force_size=3,
    w_frame=np.eye(3) * w_LFRF,
    qmin=qmin,
    qmax=qmax,
    mu=0.8,
    Lfoot=0.01,
    Wfoot=0.01,
    kinematics_limits=True,
    force_cone=False,
    land_cstr=False,
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
)

mpc = MPC(mpc_conf, dynproblem)

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
kino_ID_settings.kp_base = 7.0
kino_ID_settings.kp_posture = 10.0
kino_ID_settings.kp_contact = 0.0
kino_ID_settings.w_base = 100.0
kino_ID_settings.w_posture = 10.0
kino_ID_settings.w_contact_force = 1.0
kino_ID_settings.w_contact_motion = 0.0
# Wheel-specific posture gains/weight (if equal to the above, behavior remains unchanged)
kino_ID_settings.kp_posture_wheel = 0.0
kino_ID_settings.w_posture_wheel = 0.0
# Wheel physical settings
kino_ID_settings.wheel_radius = wheel_radius
kino_ID_settings.ff_wheel_scale = 1.0
# Temporarily disable non-holonomic rolling constraints for testing
# (set to False to check whether they cause an abrupt exit)
kino_ID_settings.enable_nonholonomic = False
kino_ID = KinodynamicsID(model_handler, dt_simu, kino_ID_settings)


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

wheel_link_names = ["FL_wheel", "FR_wheel", "RL_wheel", "RR_wheel"]
wheel_link_ids = {}
for link_id in range(p.getNumJoints(device.robotId)):
    link_name = p.getJointInfo(device.robotId, link_id)[12].decode()
    if link_name in wheel_link_names:
        wheel_link_ids[link_name] = link_id

q_meas, v_meas = device.measureState()
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

v = np.zeros(6)
v[0] = wheel_target_vel * wheel_radius
mpc.velocity_base = v
forward_speed_slider = p.addUserDebugParameter("forward_speed", -1.0, 1.0, v[0])
wheel_target_vel_prev = v[0] / wheel_radius
wheel_pos_ref = None
base_pos_ref = None
for step in range(3000):
    v[0] = p.readUserDebugParameter(forward_speed_slider)
    mpc.velocity_base = v
    wheel_target_vel = mpc.velocity_base[0] / wheel_radius
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
        q_interp = model_handler.getReferenceState()[:nq].copy()
        v_interp = np.zeros(nv)
        acc_interp = np.zeros(nv)
        contact_states = [True] * nk
        fz = -0.25 * model_handler.getMass() * gravity[2]
        force_interp = [np.array([0.0, 0.0, fz])] * nk

        q_meas, v_meas = device.measureState()
        x_measured = np.concatenate([q_meas, v_meas])

        # v_interp[wheel_v_indices] = wheel_target_vel
        v_interp[:6] = mpc.velocity_base
        if base_pos_ref is None:
            base_pos_ref = q_interp[:3].copy()
        else:
            base_pos_ref = base_pos_ref + v_interp[:3] * dt_simu
        q_interp[:3] = base_pos_ref
 
        # if wheel_pos_ref is None:
        #     wheel_pos_ref = q_interp[wheel_q_indices].copy()
        # else:
        #     wheel_pos_ref = wheel_pos_ref + wheel_target_vel * dt_simu
        # q_interp[wheel_q_indices] = wheel_pos_ref

        # Temporary test: override MPC targets with fixed ID targets and gravity compensation
        

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
