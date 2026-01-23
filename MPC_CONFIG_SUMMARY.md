# MPC Configuration Summary (go2w_kinodynamics.py)

This document summarizes the active MPC setup in `examples/go2w_kinodynamics.py` and the corresponding implementation in `src/kinodynamics.cpp` / related headers.

## A. Stage Costs (per time step)

### 1) State cost `state_cost`
- **Form:** \(J_x = (x - x_{ref})^T W_x (x - x_{ref})\)
- **Active:** Yes
- **Weights:**
  - `w_q[:6] = w_basepos = [0, 0, 100, 0, 0, 100]`
  - `w_q[6:] = w_legpos[0]` → currently 0 for all joints
  - `w_v[:6] = w_basevel = [0, 0, 10, 10, 10, 10]`
  - `w_v[6:] = w_legvel[0]` → currently 10.1 for all joints
- **Files:** `examples/go2w_kinodynamics.py`, `src/kinodynamics.cpp`

### 2) Control cost `control_cost`
- **Form:** \(J_u = (u - u_{ref})^T W_u (u - u_{ref})\)
- **Active:** Yes
- **Weights:**
  - Foot forces: `w_linforce = [0.01, 0.01, 0.01]`
  - Joint accelerations: `w_u_joints = 1e-5`
- **Files:** `examples/go2w_kinodynamics.py`, `src/kinodynamics.cpp`

### 3) Foot pose cost `*_pose_cost`
- **Form (force_size=3):** \(J_{pose,i}=(p_i - p_{ref,i})^T W_{frame} (p_i - p_{ref,i})\)
- **Active:** Yes, but **weight is zero**
- **Weights:** `w_frame = [0,0,0]` → effectively disabled
- **Files:** `examples/go2w_kinodynamics.py`, `src/kinodynamics.cpp`

### 4) Centroidal momentum cost `centroidal_cost`
- **Form:** \(J_h = h^T W_{cent} h\)
- **Active:** Yes
- **Weights:** `W_cent = diag([0,0,1, 0.1,0.1,10])`
- **Files:** `examples/go2w_kinodynamics.py`, `src/kinodynamics.cpp`

### 5) Centroidal momentum derivative cost `centroidal_derivative_cost`
- **Form:** \(J_{\dot h} = \dot h^T W_{centder} \dot h\)
- **Active:** Yes
- **Weights:** `W_centder = diag([0,0,0, 0.1,0.1,0.1])`
- **Files:** `examples/go2w_kinodynamics.py`, `src/kinodynamics.cpp`

### 6) VectorGlide cost `*_vector_glide_cost`
- **Form (per wheel):** \(r_i = (\rho_i \cdot c_{x,i})/||\rho_i||\), \(J = \sum w \cdot r_i^2\)
- **Active:** Yes
- **Weights:** `vector_glide_weight = 100.0`
- **Notes:** Disabled if `|omega_des| < vector_glide_min_omega`.
- **Files:** `include/simple-mpc/vector-glide.hpp`, `src/kinodynamics.cpp`, `examples/go2w_kinodynamics.py`

### 7) Min wheel distance soft cost `min_wheel_distance_cost`
- **Form (same-side pairs only):**
  - \(r = 1/(d - d_{min} + \epsilon) - 1/(d_0 - d_{min} + \epsilon)\)
  - \(J = \sum w \cdot r^2\)
- **Active:** Yes
- **Parameters:**
  - `min_wheel_distance = 1.2 * 2 * wheel_radius`
  - `w_min_wheel_distance = 0.05`
  - `min_wheel_distance_cost_eps = 1e-1`
- **Pairs:** `FL-RL`, `FR-RR` (same-side only)
- **Files:** `include/simple-mpc/wheel-distance.hpp`, `src/kinodynamics.cpp`, `examples/go2w_kinodynamics.py`

### 8) Vertical force variance cost `force_z_variance_cost`
- **Form:** \(J = w \cdot Var(f_{z,i})\) over **active contacts**
- **Active:** Yes
- **Weights:** `w_force_z_variance = 10.0`
- **Files:** `include/simple-mpc/mpc-extra-costs.hpp`, `src/kinodynamics.cpp`, `examples/go2w_kinodynamics.py`

### 9) Joint limit soft cost `joint_limit_soft_cost`
- **Form (per joint):**
  - Middle `n=50%` range unpenalized
  - Outside → rapidly increasing penalty, max at limit
  - \(r_i = clamp(s,0,1)^2\), \(J = w \sum r_i^2\)
- **Active:** Yes
- **Parameters:**
  - `joint_limit_soft_fraction = 0.5`
  - `w_joint_limit_soft = 2.0`
- **Files:** `include/simple-mpc/mpc-extra-costs.hpp`, `src/kinodynamics.cpp`, `examples/go2w_kinodynamics.py`

### 10) Soft contact velocity cost `*_contact_vel_soft`
- **Form:** \(J = w \cdot ||v_{slice}||^2\)
- **Active:** Yes (because `soft_constraints=True` and `w_soft_contact_vel=20`)
- **Notes:** In lateral-no-slip mode, this soft cost applies to **normal velocity** (`v_local.z`) only.
- **Files:** `src/kinodynamics.cpp`

### 11) Other costs (currently inactive)
- `soft_friction_cost` → `w_soft_friction=0`
- `soft_land_cost` → `w_soft_land=0`
- `track_width_cost` → `track_width_cstr=False`
- `foot_sum_cost` → `foot_sum_cstr=False`

---

## B. Stage Constraints

### 1) Dynamics constraint
- Always enforced by the integrator (`IntegratorSemiImplEuler`).

### 2) Joint position box constraints
- **Active:** Yes (`kinematics_limits=True`)
- **Form:** \(q_{min} \le q \le q_{max}\)
- **Notes:** wheel joints have qmin/qmax expanded to ±1e9.

### 3) Lateral no-slip constraint
- **Active:** Yes (`enable_lateral_no_slip=True`)
- **Form:** \(v_{contact} \cdot c_y = 0\)
- **Hard constraint.**

### 4) Normal contact velocity
- **Active:** Soft (via `w_soft_contact_vel`)
- **Form:** \(v_{local,z} = 0\) (soft penalty)

### 5) Min wheel distance hard constraint (same-side pairs)
- **Active:** Yes
- **Form:** \(||p_i - p_j|| \ge d_{min}\)
- **Pairs:** `FL-RL`, `FR-RR`

### 6) Other constraints (inactive)
- Friction cone (`force_cone=False`)
- Land constraint (`land_cstr=False`)

---

## C. Terminal Cost

- `state_cost` (same form as stage cost)
- `centroidal_cost` with weight `w_cent * 10`

---

## D. MPC High-level Settings

- `force_size = 3`
- `soft_constraints = True`
- `T = 50`, `dt_mpc = 0.01`
- `update_contact_ref = True`

---

# ID Controller Summary (KinodynamicsID)

This section summarizes the current inverse-dynamics controller in `src/inverse-dynamics/kinodynamics-id.cpp` and settings in `include/simple-mpc/inverse-dynamics/kinodynamics-id.hpp`.

## A. Core Tasks / Costs (TSID)

### 1) Base task
- **Type:** `TaskSE3Equality`
- **Reference:** base pose from target state, velocities from `targetVelBase_` / `targetAccBase_`
- **Weights:** `w_base`
- **Gains:** `kp_base`

### 2) Posture task
- **Type:** `TaskJointPosture`
- **Reference:** joint targets from `q_target_`, `v_target_`, `a_target_`
- **Weights:** `w_posture`
- **Gains:** `kp_posture`

### 3) Contact motion / force tasks
- **Contact types:** point or 6D
- **Weights:**
  - `w_contact_motion` (motion)
  - `w_contact_force` (force)
- **Hard/soft:** `contact_motion_equality` chooses constraint vs cost

### 4) Optional VectorGlide debug metric
- **Not a TSID cost.** Only prints `J_kin` when enabled.
- **Switch:** `use_vector_glide_cost`
- **Weight:** `vector_glide_weight`
- **Implementation:** `src/inverse-dynamics/kinodynamics-id.cpp`

---

## B. Constraints (ID)

### 1) Lateral no-slip constraints (new)
- **Active:** `enable_lateral_no_slip` (default True)
- **Form:** \(v_{contact} \cdot c_y = 0\)
- **Hard equality by default**, or **bounded inequality** if `lateral_no_slip_use_bounds=True`:
  \(lower \le v_{contact}\cdot c_y \le upper\)
- **Parameters:**
  - `lateral_no_slip_min_axis_norm`, `lateral_no_slip_min_cross_norm`
  - `lateral_no_slip_use_bounds`
  - `lateral_no_slip_lower`, `lateral_no_slip_upper`

### 2) Legacy nonholonomic constraint
- **Present but disabled by default** (`enable_nonholonomic=False`)

---

## C. Torque Feedforward (wheel)
- **Only used if** `rolling_constraints_` is empty
- **Params:** `wheel_radius`, `ff_wheel_scale`
- **Implementation:** Part B in `KinodynamicsID::solve()`

---

## D. ID Settings (from kinodynamics-id.hpp)

- `kp_base`, `kp_posture`, `kp_posture_wheel`, `kp_contact`
- `w_base`, `w_posture`, `w_posture_wheel`, `w_contact_motion`, `w_contact_force`
- `wheel_radius`, `ff_wheel_scale`
- Lateral no-slip settings (listed above)
- `use_vector_glide_cost`, `vector_glide_weight`

