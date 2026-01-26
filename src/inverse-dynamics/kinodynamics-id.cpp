#include <simple-mpc/inverse-dynamics/kinodynamics-id.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <cmath>

namespace {
inline Eigen::Vector3d rpyFromMatrix(const Eigen::Matrix3d & R) {
  const double roll = std::atan2(R(2,1), R(2,2));
  const double pitch = std::atan2(-R(2,0), std::sqrt(R(0,0)*R(0,0) + R(1,0)*R(1,0)));
  const double yaw = std::atan2(R(1,0), R(0,0));
  return Eigen::Vector3d(roll, pitch, yaw);
}

inline Eigen::Matrix3d rpyToMatrix(double roll, double pitch, double yaw) {
  const double cr = std::cos(roll);
  const double sr = std::sin(roll);
  const double cp = std::cos(pitch);
  const double sp = std::sin(pitch);
  const double cy = std::cos(yaw);
  const double sy = std::sin(yaw);
  Eigen::Matrix3d R;
  R << cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr,
       sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr,
       -sp,   cp*sr,            cp*cr;
  return R;
}
} // namespace
#include <iostream>
#include <tsid/contacts/contact-6d.hpp>
#include <tsid/contacts/contact-point.hpp>
#include <tsid/math/constraint-bound.hpp>

using namespace simple_mpc;

KinodynamicsID::KinodynamicsID(const RobotModelHandler & model_handler, double control_dt, const Settings settings)
: settings_(settings)
, model_handler_(model_handler)
, data_handler_(model_handler_)
, robot_(model_handler_.getModel())
, formulation_("tsid", robot_)
, solver_("solver-proxqp")
{
  const pinocchio::Model & model = model_handler.getModel();
  const int nq = model.nq;
  const int nq_actuated = robot_.nq_actuated();
  const int nv = model.nv;
  const int nu = nv - 6;

  // Prepare foot contact tasks
  const size_t n_contacts = model_handler_.getFeetNb();
  const Eigen::Vector3d normal{0, 0, 1};
  const double weight = model_handler_.getMass() * 9.81;
  const double max_f = settings_.contact_weight_ratio_max * weight;
  const double min_f = settings_.contact_weight_ratio_min * weight;
  for (size_t i = 0; i < n_contacts; i++)
  {
    const std::string frame_name = model_handler.getFootFrameName(i);
    switch (model_handler.getFootType(i))
    {
    case RobotModelHandler::FootType::POINT: {
      auto contact_point = new tsid::contacts::ContactPoint(
        frame_name, robot_, frame_name, normal, settings_.friction_coefficient, min_f, max_f);
      contact_point->Kp(settings_.kp_contact * Eigen::VectorXd::Ones(3));
      contact_point->Kd(2.0 * contact_point->Kp().cwiseSqrt());
      contact_point->useLocalFrame(false);
      tsid_contacts.emplace_back(contact_point);
      break;
    }
    case RobotModelHandler::FootType::QUAD: {
      auto contact_6D = new tsid::contacts::Contact6d(
        frame_name, robot_, frame_name, model_handler_.getQuadFootContactPoints(i).transpose(), normal,
        settings_.friction_coefficient, min_f, max_f);
      contact_6D->Kp(settings_.kp_contact * Eigen::VectorXd::Ones(6));
      contact_6D->Kd(2.0 * contact_6D->Kp().cwiseSqrt());
      tsid_contacts.emplace_back(contact_6D);
      break;
    }
    default: {
      assert(false);
    }
    }
    // By default contact is not active (will be by setTarget)
    active_tsid_contacts_.push_back(false);
  }

  // Add the posture task
  postureTask_ = std::make_unique<tsid::tasks::TaskJointPosture>("task-posture", robot_);

  // Build a per-actuator Kp vector. By default use settings_.kp_posture for all actuated joints,
  // but override wheel actuators with kp_posture_wheel when provided. To emulate a wheel-specific
  // posture "weight" without adding a separate task, we optionally scale the wheel Kp by
  // settings_.w_posture_wheel if it is positive.
  
  Eigen::VectorXd Kp_all = settings_.kp_posture * Eigen::VectorXd::Ones(nu);
  if (settings_.kp_posture_wheel > 0.)
  {
    // detect wheel joints by frame name containing "wheel"
    for (std::size_t foot_nb = 0; foot_nb < model_handler_.getFeetNb(); ++foot_nb)
    {
      const std::string & frame_name = model_handler_.getFootFrameName(foot_nb);
      if (frame_name.find("wheel") == std::string::npos)
        continue;
      const pinocchio::FrameIndex frame_id = model_handler_.getFootFrameId(foot_nb);
      const pinocchio::JointIndex joint_id = model.frames[frame_id].parentJoint;
      const int idx_v = model.joints[joint_id].idx_v();
      const int actuator_index = idx_v - 6;
      if (actuator_index >= 0 && actuator_index < nu)
      {
        double kp_val = settings_.kp_posture_wheel;
        if (settings_.w_posture_wheel > 0. && settings_.w_posture > 0.)
        {
          // scale Kp to reflect wheel-specific weight relative to base posture weight
          kp_val *= (settings_.w_posture_wheel / settings_.w_posture);
        }
        Kp_all(actuator_index) = kp_val;
      }
    }
  }


  postureTask_->Kp(Kp_all);
  postureTask_->Kd(2.0 * postureTask_->Kp().cwiseSqrt());

  if (settings_.w_posture > 0.)
    formulation_.addMotionTask(*postureTask_, settings_.w_posture, 1);

  samplePosture_.resize(nq_actuated, nu);

  // Add the base task
  baseTask_ = std::make_unique<tsid::tasks::TaskSE3Equality>("task-base", robot_, model_handler_.getBaseFrameName());
  baseTask_->Kp(settings_.kp_base * Eigen::VectorXd::Ones(6));
  baseTask_->Kd(2.0 * baseTask_->Kp().cwiseSqrt());
  if (settings_.w_base > 0.)
    formulation_.addMotionTask(*baseTask_, settings_.w_base, 1);

  sampleBase_.resize(12, 6);

  // Add joint limit task
  boundsTask_ = std::make_unique<tsid::tasks::TaskJointPosVelAccBounds>("task-joint-limits", robot_, control_dt);
  boundsTask_->setPositionBounds(
    model_handler_.getModel().lowerPositionLimit.tail(nq_actuated),
    model_handler_.getModel().upperPositionLimit.tail(nq_actuated));
  boundsTask_->setVelocityBounds(model_handler_.getModel().velocityLimit.tail(nu));
  boundsTask_->setImposeBounds(
    true, true, true, false); // For now do not impose acceleration bound as it is not provided in URDF
  formulation_.addMotionTask(*boundsTask_, 1.0, 0); // No weight needed as it is set as constraint

  // Add actuation limit task
  actuationTask_ = std::make_unique<tsid::tasks::TaskActuationBounds>("actuation-limits", robot_);
  const Eigen::VectorXd tau_limit = 2.0 * model_handler_.getModel().effortLimit.tail(nu);
  actuationTask_->setBounds(
    -tau_limit, tau_limit);
  formulation_.addActuationTask(*actuationTask_, 1.0, 0); // No weight needed as it is set as constraint

  // Create an HQP solver
  solver_.resize(formulation_.nVar(), formulation_.nEq(), formulation_.nIn());

  // By default initialize target in reference state
  const Eigen::VectorXd q_ref = model_handler.getReferenceState().head(nq);
  const Eigen::VectorXd v_ref = model_handler.getReferenceState().tail(nv);
  std::vector<bool> c_ref(n_contacts);
  std::vector<TargetContactForce> f_ref;
  for (size_t i = 0; i < n_contacts; i++)
  {
    // By default initialize all foot in contact with same amount of force
    c_ref[i] = true;
    const RobotModelHandler::FootType foot_type = model_handler.getFootType(i);
    if (foot_type == RobotModelHandler::POINT)
      f_ref.push_back(TargetContactForce::Zero(3));
    else if (foot_type == RobotModelHandler::QUAD)
      f_ref.push_back(TargetContactForce::Zero(6));
    else
      assert(false);
    f_ref[i][2] = weight / static_cast<double>(n_contacts); // Weight on Z axis
  }
  setTarget(q_ref, v_ref, v_ref, c_ref, f_ref);

  // Dry run to initialize solver data & output
  const tsid::solvers::HQPData & solver_data = formulation_.computeProblemData(0, q_ref, v_ref);
  last_solution_ = solver_.solve(solver_data);
}

void KinodynamicsID::setTarget(
  const Eigen::Ref<const Eigen::VectorXd> & q_target,
  const Eigen::Ref<const Eigen::VectorXd> & v_target,
  const Eigen::Ref<const Eigen::VectorXd> & a_target,
  const std::vector<bool> & contact_state_target,
  const std::vector<TargetContactForce> & f_target)
{
  q_target_ = q_target;
  v_target_ = v_target;
  a_target_ = a_target;
  data_handler_.updateInternalData(q_target, v_target, false);

  // Posture task
  samplePosture_.setValue(q_target.tail(robot_.nq_actuated()));
  samplePosture_.setDerivative(v_target.tail(robot_.na()));
  samplePosture_.setSecondDerivative(a_target.tail(robot_.na()));
  postureTask_->setReference(samplePosture_);

  // Base task
  tsid::math::SE3ToVector(data_handler_.getBaseFramePose(), sampleBase_.pos);
  /* Velocity and acceleration not set here since the actual position of the robot is needed to transform from local to
   * world-aligned frame. Will be done in solve()
   */
  targetVelBase_ = v_target.head<6>();
  targetAccBase_ = a_target.head<6>();

  // Foot contacts
  for (std::size_t foot_nb = 0; foot_nb < model_handler_.getFeetNb(); foot_nb++)
  {
    const std::string & name{model_handler_.getFootFrameName(foot_nb)};
    if (contact_state_target[foot_nb])
    {
      // Add contact to tsid if necessary
      if (!active_tsid_contacts_[foot_nb])
      {
        formulation_.addRigidContact(
          *tsid_contacts[foot_nb], settings_.w_contact_force, settings_.w_contact_motion,
          settings_.contact_motion_equality ? 0 : 1);
        active_tsid_contacts_[foot_nb] = true;
      }
      // Set contact target force
      switch (model_handler_.getFootType(foot_nb))
      {
      case RobotModelHandler::FootType::POINT: {
        assert(f_target.at(foot_nb).size() == 3);
        static_cast<tsid::contacts::ContactPoint &>(*tsid_contacts[foot_nb]).setForceReference(f_target.at(foot_nb));
        break;
      }
      case RobotModelHandler::FootType::QUAD: {
        assert(f_target.at(foot_nb).size() == 6);
        static_cast<tsid::contacts::Contact6d &>(*tsid_contacts[foot_nb]).setForceReference(f_target.at(foot_nb));
        break;
      }
      default: {
        assert(false);
      }
      }
    }
    else
    {
      // Remove contact from tsid if necessary
      if (active_tsid_contacts_[foot_nb])
      {
        formulation_.removeRigidContact(name, 0);
        active_tsid_contacts_[foot_nb] = false;
      }
    }
  }

  solver_.resize(formulation_.nVar(), formulation_.nEq(), formulation_.nIn());
}

void KinodynamicsID::addActuatedRollingConstraint(const std::string &contact_frame_name, double radius, double inertia)
{
  rolling_constraints_.emplace_back(contact_frame_name, radius, inertia);
}

void KinodynamicsID::addNonHolonomicRollingConstraint(const std::string &contact_frame_name, double radius)
{
  nonholonomic_constraints_.emplace_back(contact_frame_name, radius);
}

void KinodynamicsID::addLateralNoSlipConstraint(const std::string &contact_frame_name)
{
  lateral_no_slip_constraints_.emplace_back(contact_frame_name);
}

void KinodynamicsID::solve(
  const double t,
  const Eigen::Ref<const Eigen::VectorXd> & q_meas,
  const Eigen::Ref<const Eigen::VectorXd> & v_meas,
  Eigen::Ref<Eigen::VectorXd> tau_res)
{
  const pinocchio::Model & model = model_handler_.getModel();
  
  // Update targets
  pinocchio::Data data_target(model);
  pinocchio::forwardKinematics(model, data_target, q_target_);
  pinocchio::updateFramePlacements(model, data_target);

  // Update contact position based on real robot state
  data_handler_.updateInternalData(q_meas, v_meas, false);
  for (std::size_t foot_nb = 0; foot_nb < model_handler_.getFeetNb(); foot_nb++)
  {
    if (active_tsid_contacts_[foot_nb])
    {
      auto foot_type = model_handler_.getFootType(foot_nb);
      const auto & foot_pose = data_handler_.getFootPose(foot_nb);
      
      if (foot_type == RobotModelHandler::FootType::POINT) {
        static_cast<tsid::contacts::ContactPoint &>(*tsid_contacts[foot_nb]).setReference(foot_pose);
      }
      else if (foot_type == RobotModelHandler::FootType::QUAD) {
        static_cast<tsid::contacts::Contact6d &>(*tsid_contacts[foot_nb]).setReference(foot_pose);
      }
    }
  }

  // Task Reference Updates
  const pinocchio::SE3 oMb_rotation(data_handler_.getBaseFramePose().rotation(), Eigen::Vector3d::Zero());
  const int nq_actuated = robot_.nq_actuated();
  const int na = robot_.na();
  
  const Eigen::VectorXd q_ref = q_target_.tail(nq_actuated);
  const Eigen::VectorXd v_ref = v_target_.tail(na);
  const Eigen::VectorXd a_ref = a_target_.tail(na);
  const Eigen::VectorXd q_meas_act = q_meas.tail(nq_actuated);
  const Eigen::VectorXd v_meas_act = v_meas.tail(na);

  // Debug print commented out for performance
  // const Eigen::VectorXd ddq_des = a_ref + postureTask_->Kp().cwiseProduct(q_ref - q_meas_act) + postureTask_->Kd().cwiseProduct(v_ref - v_meas_act);
  // std::cout << "ID ddq_des: " << ddq_des.transpose() << std::endl;

  const pinocchio::Motion v_world_aligned{oMb_rotation.act(pinocchio::Motion(targetVelBase_))};
  const pinocchio::Motion a_world_aligned{oMb_rotation.act(pinocchio::Motion(targetAccBase_))};
  sampleBase_.setDerivative(v_world_aligned.toVector());
  sampleBase_.setSecondDerivative(a_world_aligned.toVector()); // Fixed: setSecondDerivative for acceleration
  if (settings_.enable_pendulum_coupling)
  {
    const pinocchio::SE3 & base_pose = data_handler_.getBaseFramePose();
    const Eigen::Vector3d base_pos = base_pose.translation();
    const Eigen::Matrix3d base_R = base_pose.rotation();
    const double yaw = std::atan2(base_R(1, 0), base_R(0, 0));
    const Eigen::Vector3d com = data_handler_.getData().com[0];
    double v_com_x = 0.0;
    if (data_handler_.getData().vcom.size() > 0)
      v_com_x = data_handler_.getData().vcom[0].x();

    Eigen::Vector3d front_avg = Eigen::Vector3d::Zero();
    Eigen::Vector3d rear_avg = Eigen::Vector3d::Zero();
    int front_n = 0;
    int rear_n = 0;
    for (std::size_t foot_nb = 0; foot_nb < model_handler_.getFeetNb(); ++foot_nb)
    {
      const std::string & name = model_handler_.getFootFrameName(foot_nb);
      const Eigen::Vector3d p = data_handler_.getFootPose(foot_nb).translation();
      if (name.find("FL") != std::string::npos || name.find("FR") != std::string::npos)
      {
        front_avg += p;
        front_n++;
      }
      else if (name.find("RL") != std::string::npos || name.find("RR") != std::string::npos)
      {
        rear_avg += p;
        rear_n++;
      }
    }
    if (front_n > 0) front_avg /= front_n;
    if (rear_n > 0) rear_avg /= rear_n;

    const double front_rear_dist = std::abs(front_avg.x() - rear_avg.x());
    const double denom = std::max(settings_.pendulum_max_front_rear_dist - settings_.pendulum_min_front_rear_dist, 1e-6);
    double scale = (front_rear_dist - settings_.pendulum_min_front_rear_dist) / denom;
    if (scale < 0.0) scale = 0.0;
    if (scale > 1.0) scale = 1.0;

    const double com_z = std::max(com.z(), 1e-3);
    const double omega0 = std::sqrt(9.81 / com_z);
    double x_rel_des = 0.0;
    if (std::abs(omega0) > settings_.pendulum_omega_eps)
      x_rel_des = v_com_x / omega0;

    const double avg_wheel_x = 0.5 * (front_avg.x() + rear_avg.x());
    const double e = (avg_wheel_x - com.x()) - x_rel_des;

    // Adjust base reference x and pitch using pendulum coupling
    const Eigen::Vector3d rpy_cur = rpyFromMatrix(base_R);
    const double roll_cur = rpy_cur.x();
    const double pitch_cur = rpy_cur.y();
    const double pitch_des = std::atan2(x_rel_des, com_z);

    pinocchio::SE3 base_des = base_pose;
    base_des.translation().x() += e;
    const Eigen::Matrix3d R_des = rpyToMatrix(roll_cur, pitch_des, yaw);
    base_des.rotation() = R_des;
    tsid::math::SE3ToVector(base_des, sampleBase_.pos);

    Eigen::VectorXd Kp = settings_.kp_base * Eigen::VectorXd::Ones(6);
    Kp[0] *= scale;
    Kp[4] *= settings_.pendulum_weight;
    baseTask_->Kp(Kp);
    baseTask_->Kd(2.0 * baseTask_->Kp().cwiseSqrt());
  }
  baseTask_->setReference(sampleBase_);

  // 1. Compute Standard Problem Data
  const tsid::solvers::HQPData & solver_data = formulation_.computeProblemData(t, q_meas, v_meas);
  
  // Make mutable copy for custom constraints
  tsid::solvers::HQPData hqp = solver_data;

  // ===========================================================================
  // [PART A] Lateral no-slip constraints (Kinematics)
  // Implements: v_contact dot c_y = 0
  // ===========================================================================
  const bool need_lateral_no_slip = settings_.enable_lateral_no_slip && !lateral_no_slip_constraints_.empty();
  const bool need_vector_glide = settings_.use_vector_glide_cost && !lateral_no_slip_constraints_.empty();
  if (need_lateral_no_slip || need_vector_glide)
  {
    data_handler_.updateInternalData(q_meas, v_meas, true); // Update Jacobians
  }

  if (need_vector_glide)
  {
    double J_kin = 0.0;
    bool compute_vector_glide = true;
    const pinocchio::SE3 & base_pose = data_handler_.getBaseFramePose();
    const Eigen::Vector3d r_base = base_pose.translation();
    const Eigen::Matrix3d R_base = base_pose.rotation();
    const double v_des = targetVelBase_.linear().x();
    const double omega_des = targetVelBase_.angular().z();
    Eigen::Vector3d r_O_star = r_base;
    if (std::abs(omega_des) > 1e-6)
    {
      const Eigen::Vector3d icr_local(0.0, v_des / omega_des, 0.0);
      r_O_star = r_base + R_base * icr_local;
    }
    else
    {
      compute_vector_glide = false;
    }

    for (unsigned int ic = 0; ic < lateral_no_slip_constraints_.size(); ++ic)
    {
      const auto &nh = lateral_no_slip_constraints_[ic];
      std::size_t foot_nb = model_handler_.getFootNb(nh.contact_frame_name);
      const pinocchio::SE3 & foot_pose = data_handler_.getFootPose(foot_nb);
      Eigen::Vector3d a_y = foot_pose.rotation().col(1);
      if (a_y.norm() < settings_.lateral_no_slip_min_axis_norm)
        a_y = Eigen::Vector3d::UnitY();
      else
        a_y.normalize();

      const Eigen::Vector3d g_z = Eigen::Vector3d::UnitZ();
      Eigen::Vector3d c_x = a_y.cross(g_z);
      if (c_x.norm() < settings_.lateral_no_slip_min_cross_norm)
      {
        Eigen::Vector3d a_x = foot_pose.rotation().col(0);
        c_x = a_x - g_z * a_x.dot(g_z);
      }
      if (c_x.norm() < settings_.lateral_no_slip_min_cross_norm)
        c_x = Eigen::Vector3d::UnitX();
      c_x.normalize();

      if (compute_vector_glide)
      {
        const Eigen::Vector3d r_contact = foot_pose.translation();
        const Eigen::Vector3d rho = r_O_star - r_contact;
        const double rho_norm = rho.norm();
        if (rho_norm > 1e-9)
        {
          const double e_i = rho.dot(c_x);
          const double term = (e_i / rho_norm);
          J_kin += settings_.vector_glide_weight * term * term;
        }
      }
    }

    if (compute_vector_glide)
    {
      std::cout << "[VectorGlide] J_kin = " << J_kin << std::endl;
    }
    else
    {
      std::cout << "[VectorGlide] omega_des ~ 0, skip J_kin" << std::endl;
    }
  }

  if (need_lateral_no_slip)
  {
    const unsigned int nvar = formulation_.nVar();
    const unsigned int rows = static_cast<unsigned int>(lateral_no_slip_constraints_.size());

    std::shared_ptr<tsid::math::ConstraintBase> nh_constraint;
    Eigen::VectorXd b_nh = Eigen::VectorXd::Zero(rows);
    if (settings_.lateral_no_slip_use_bounds)
    {
      auto bound = std::make_shared<tsid::math::ConstraintBound>("lateral-no-slip", rows);
      bound->matrix().resize(rows, nvar);
      bound->lowerBound().setConstant(rows, settings_.lateral_no_slip_lower);
      bound->upperBound().setConstant(rows, settings_.lateral_no_slip_upper);
      nh_constraint = bound;
    }
    else
    {
      nh_constraint = std::make_shared<tsid::math::ConstraintEquality>("lateral-no-slip", rows, nvar);
    }

    for (unsigned int ic = 0; ic < lateral_no_slip_constraints_.size(); ++ic)
    {
      const auto &nh = lateral_no_slip_constraints_[ic];
      std::size_t foot_nb = model_handler_.getFootNb(nh.contact_frame_name);

      const pinocchio::FrameIndex frame_id = model_handler_.getFootFrameId(foot_nb);

      const pinocchio::Data & data_const = data_handler_.getData();
      pinocchio::Data & data_nc = const_cast<pinocchio::Data &>(data_const);

      Eigen::MatrixXd J6 = Eigen::MatrixXd::Zero(6, model.nv);
      pinocchio::getFrameJacobian(model, data_nc, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J6);

      const pinocchio::SE3 & foot_pose = data_handler_.getFootPose(foot_nb);
      Eigen::Vector3d a_y = foot_pose.rotation().col(1);
      if (a_y.norm() < settings_.lateral_no_slip_min_axis_norm)
        a_y = Eigen::Vector3d::UnitY();
      else
        a_y.normalize();

      const Eigen::Vector3d g_z = Eigen::Vector3d::UnitZ();
      Eigen::Vector3d c_x = a_y.cross(g_z);
      if (c_x.norm() < settings_.lateral_no_slip_min_cross_norm)
      {
        Eigen::Vector3d a_x = foot_pose.rotation().col(0);
        c_x = a_x - g_z * a_x.dot(g_z);
      }
      if (c_x.norm() < settings_.lateral_no_slip_min_cross_norm)
        c_x = Eigen::Vector3d::UnitX();
      c_x.normalize();

      Eigen::Vector3d c_y = g_z.cross(c_x);
      if (c_y.norm() < settings_.lateral_no_slip_min_cross_norm)
        c_y = Eigen::Vector3d::UnitY();
      else
        c_y.normalize();

      // Pinocchio motion vector ordering is [angular; linear].
      const Eigen::MatrixXd J_linear = J6.block(3, 0, 3, model.nv);

      Eigen::RowVectorXd rowCoeffs = c_y.transpose() * J_linear;
      for (int col = 0; col < model.nv; ++col)
        nh_constraint->matrix()(static_cast<int>(ic), col) = rowCoeffs(col);
      b_nh(static_cast<int>(ic)) = 0.0;
    }

    nh_constraint->setVector(b_nh);
    if (hqp.size() == 0)
      hqp.resize(1);
    hqp[0].push_back(
      tsid::solvers::aligned_pair<double, std::shared_ptr<tsid::math::ConstraintBase> >(1.0, nh_constraint));
  }

  // 2. Solve QP
  last_solution_ = solver_.solve(hqp);
  
  if(last_solution_.status != tsid::solvers::HQPStatus::HQP_STATUS_OPTIMAL) {
      // Fallback or warning
      // std::cerr << "QP Solver failed!" << std::endl;
  }
  tau_res = formulation_.getActuatorForces(last_solution_);

  // ===========================================================================
  // [PART B] Torque Feedforward (Dynamics)
  // Use simple post-processing if complex actuation constraints are not used
  // ===========================================================================
  if (rolling_constraints_.empty())
  {
    const double wheel_radius = settings_.wheel_radius; 
    const double feedforward_scale = settings_.ff_wheel_scale; 

    for (std::size_t foot_nb = 0; foot_nb < model_handler_.getFeetNb(); foot_nb++)
    {
      if (!active_tsid_contacts_[foot_nb]) continue;
      
      const std::string & frame_name = model_handler_.getFootFrameName(foot_nb);
      if (frame_name.find("wheel") == std::string::npos) continue;

      // Get Contact Force in WORLD frame (tsid default)
      const Eigen::VectorXd f = formulation_.getContactForces(frame_name, last_solution_);
      if (f.size() < 3) continue;
      
      const Eigen::Vector3d f_world = f.head<3>();

      // Get Indices
      const pinocchio::FrameIndex frame_id = model_handler_.getFootFrameId(foot_nb);
      const pinocchio::JointIndex joint_id = model.frames[frame_id].parentJoint;
      const int idx_v = model.joints[joint_id].idx_v();
      const int nv_joint = model.joints[joint_id].nv();
      if (nv_joint != 1) continue;

      // Geometry calculation
      const pinocchio::SE3 & foot_pose = data_handler_.getFootPose(foot_nb);
      Eigen::Vector3d axis_world = foot_pose.rotation() * Eigen::Vector3d::UnitY();
      if (axis_world.norm() > 1e-12)
        axis_world.normalize();
      const Eigen::Vector3d normal_world = Eigen::Vector3d::UnitZ(); // 假定地面法向恒为世界 +Z
      const Eigen::Vector3d r_world = -wheel_radius * normal_world;

      // Torque = (r x f) . axis
      // Note on sign: This calculates the torque exerted BY the ground ON the wheel.
      // The motor needs to exert the opposite to maintain equilibrium/drive.
      // However, for "driving forward", f_friction is forward. r is down.
      // r x f = (0,0,-R) x (F,0,0) = (0, -RF, 0).
      // This is a negative torque about Y.
      // To drive forward (positive Y rotation), we likely need positive torque.
      // So we generally Subtract this resistance torque (tau_res -= ...) or Add drive torque.
      // Let's use logic: To push forward, motor torque > 0. Ground force > 0.
      // tau_ff = -RF (negative). So we should SUBTRACT a negative value (Add).
      // Or simply: tau_motor = F_traction * R.
      
      const double tau_load = (r_world.cross(f_world)).dot(axis_world);
      
      const int actuator_index = idx_v - 6;
      if (actuator_index >= 0 && actuator_index < static_cast<int>(tau_res.size()))
      {
          // Apply compensation. 
          // If robot moves backward when it should move forward, flip this sign to +=
          tau_res[actuator_index] -= feedforward_scale * tau_load; 
      }
    }
  }
}
void KinodynamicsID::getAccelerations(Eigen::Ref<Eigen::VectorXd> ddq)
{
  ddq = formulation_.getAccelerations(last_solution_);
}
