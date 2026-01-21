#include <simple-mpc/inverse-dynamics/kinodynamics-id.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <iostream>
#include <tsid/contacts/contact-6d.hpp>
#include <tsid/contacts/contact-point.hpp>

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
  actuationTask_->setBounds(
    -model_handler_.getModel().effortLimit.tail(nu), model_handler_.getModel().effortLimit.tail(nu));
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

void KinodynamicsID::solve(
  const double t,
  const Eigen::Ref<const Eigen::VectorXd> & q_meas,
  const Eigen::Ref<const Eigen::VectorXd> & v_meas,
  Eigen::Ref<Eigen::VectorXd> tau_res)
{
  const pinocchio::Model & model = model_handler_.getModel();
  pinocchio::Data data_target(model);
  pinocchio::forwardKinematics(model, data_target, q_target_);
  pinocchio::updateFramePlacements(model, data_target);

  // Update contact position based on the real robot foot placement
  data_handler_.updateInternalData(q_meas, v_meas, false);
  for (std::size_t foot_nb = 0; foot_nb < model_handler_.getFeetNb(); foot_nb++)
  {
    if (active_tsid_contacts_[foot_nb])
    {
      switch (model_handler_.getFootType(foot_nb))
      {
      case RobotModelHandler::FootType::POINT: {
        static_cast<tsid::contacts::ContactPoint &>(*tsid_contacts[foot_nb])
          .setReference(data_handler_.getFootPose(foot_nb));
        break;
      }
      case RobotModelHandler::FootType::QUAD: {
        static_cast<tsid::contacts::Contact6d &>(*tsid_contacts[foot_nb])
          .setReference(data_handler_.getFootPose(foot_nb));
        break;
      }
      default: {
        assert(false);
      }
      }
    }
  }

  // Convert robot base vel/acc from local to world-aligned frame using actual robot pose
  const pinocchio::SE3 oMb_rotation(data_handler_.getBaseFramePose().rotation(), Eigen::Vector3d::Zero());

  const int nq_actuated = robot_.nq_actuated();
  const int na = robot_.na();
  const Eigen::VectorXd q_ref = q_target_.tail(nq_actuated);
  const Eigen::VectorXd v_ref = v_target_.tail(na);
  const Eigen::VectorXd a_ref = a_target_.tail(na);
  const Eigen::VectorXd q_meas_act = q_meas.tail(nq_actuated);
  const Eigen::VectorXd v_meas_act = v_meas.tail(na);
  const Eigen::VectorXd ddq_des =
    a_ref + postureTask_->Kp().cwiseProduct(q_ref - q_meas_act) +
    postureTask_->Kd().cwiseProduct(v_ref - v_meas_act);
  std::cout << "ID ddq_des: " << ddq_des.transpose() << std::endl;
  const pinocchio::Motion v_world_aligned{oMb_rotation.act(pinocchio::Motion(targetVelBase_))};
  const pinocchio::Motion a_world_aligned{oMb_rotation.act(pinocchio::Motion(targetAccBase_))};
  sampleBase_.setDerivative(v_world_aligned.toVector());
  sampleBase_.setDerivative(a_world_aligned.toVector());
  baseTask_->setReference(sampleBase_);

  // Solve QP
  const tsid::solvers::HQPData & solver_data = formulation_.computeProblemData(t, q_meas, v_meas);

  // Make a mutable copy of the HQP data so we can append our custom equality constraints
  tsid::solvers::HQPData hqp = solver_data;

  // If the user registered rolling constraints, build an equality constraint for them
  if (!rolling_constraints_.empty())
  {
    const unsigned int nvar = formulation_.nVar();

    // Try to find actuation mapping: find the actuation constraint matrix to locate tau columns
    std::vector<int> actuator_col_by_row; // actuator row -> column index in x
    for (const auto &level : hqp)
    {
      for (const auto &p : level)
      {
        const auto &c = p.second;
        if (c && c->name().find("actuation-limits") != std::string::npos)
        {
          const Eigen::MatrixXd A = c->matrix();
          const int rows = static_cast<int>(A.rows());
          const int cols = static_cast<int>(A.cols());
          actuator_col_by_row.assign(rows, -1);
          for (int r = 0; r < rows; ++r)
          {
            for (int col = 0; col < cols; ++col)
            {
              if (std::abs(A(r, col)) > 1e-9)
              {
                actuator_col_by_row[r] = col;
                break;
              }
            }
          }
          break;
        }
      }
      if (!actuator_col_by_row.empty())
        break;
    }

    // For each registered rolling constraint, build a row in the equality matrix
    const std::string eq_name = "actuated-rolling";
    const unsigned int rows = static_cast<unsigned int>(rolling_constraints_.size());
    auto eq = std::make_shared<tsid::math::ConstraintEquality>(eq_name, rows, nvar);
    eq->setVector(Eigen::VectorXd::Zero(rows));

    for (unsigned int ic = 0; ic < rolling_constraints_.size(); ++ic)
    {
      const auto &rc = rolling_constraints_[ic];
      // find foot index
      int foot_nb = -1;
      for (std::size_t f = 0; f < model_handler_.getFeetNb(); ++f)
      {
        if (model_handler_.getFootFrameName(f) == rc.contact_frame_name)
        {
          foot_nb = static_cast<int>(f);
          break;
        }
      }
      if (foot_nb < 0)
        continue;

      // find contact force columns for this contact: search HQP for a constraint referencing the contact frame name
      int contact_force_col0 = -1; // first column index for this contact's force vector
      int contact_force_dim = 0;
      for (const auto &level : hqp)
      {
        for (const auto &p : level)
        {
          const auto &c = p.second;
          if (!c) continue;
          const std::string cname = c->name();
          if (cname.find(rc.contact_frame_name) != std::string::npos && c->cols() == static_cast<unsigned int>(nvar))
          {
            // Inspect matrix to find contiguous non-zero columns for this contact
            const Eigen::MatrixXd A = c->matrix();
            const int cols = static_cast<int>(A.cols());
            // look for columns having any non-zero in rows
            std::vector<int> cols_nonzero;
            for (int col = 0; col < cols; ++col)
            {
              if (A.col(col).cwiseAbs().maxCoeff() > 1e-9)
                cols_nonzero.push_back(col);
            }
            if (!cols_nonzero.empty())
            {
              // assume contiguous and represent contact force components
              contact_force_col0 = cols_nonzero.front();
              contact_force_dim = static_cast<int>(cols_nonzero.size());
              break;
            }
          }
        }
        if (contact_force_col0 >= 0)
          break;
      }

      // If we couldn't find by name, attempt to find contact block by scanning for contact force dimensions
      if (contact_force_col0 < 0)
      {
        // fallback: try to locate the first constraint that looks like a contact-force constraint (rows small, non-zero columns)
        for (const auto &level : hqp)
        {
          for (const auto &p : level)
          {
            const auto &c = p.second;
            if (!c) continue;
            const Eigen::MatrixXd A = c->matrix();
            if (A.rows() > 0 && A.rows() <= 6)
            {
              std::vector<int> cols_nonzero;
              for (int col = 0; col < static_cast<int>(A.cols()); ++col)
              {
                if (A.col(col).cwiseAbs().maxCoeff() > 1e-9)
                  cols_nonzero.push_back(col);
              }
              if (!cols_nonzero.empty())
              {
                contact_force_col0 = cols_nonzero.front();
                contact_force_dim = static_cast<int>(cols_nonzero.size());
                break;
              }
            }
          }
          if (contact_force_col0 >= 0)
            break;
        }
      }

      // determine joint and actuator index for this foot
      const pinocchio::FrameIndex frame_id = model_handler_.getFootFrameId(foot_nb);
      const pinocchio::JointIndex joint_id = model.frames[frame_id].parentJoint;
      const int idx_v = model.joints[joint_id].idx_v();
      const int nv_joint = model.joints[joint_id].nv();
      if (nv_joint != 1)
        continue;

      // actuator row index (row in actuation constraint) typically equals (idx_v - 6)
      const int actuator_row = idx_v - 6;

      // tau column in x
      int tau_col = -1;
      if (actuator_row >= 0 && actuator_row < static_cast<int>(actuator_col_by_row.size()))
        tau_col = actuator_col_by_row[actuator_row];

      // build coefficients for contact force: compute c_f = axis_world.cross(r_world)
      const pinocchio::SE3 & foot_pose = data_handler_.getFootPose(foot_nb);
      const Eigen::Vector3d axis_world = foot_pose.rotation() * Eigen::Vector3d::UnitY();
  // Use per-constraint radius if provided (>0), otherwise fall back to global setting
  const double use_radius = (rc.radius > 0.0) ? rc.radius : settings_.wheel_radius;
  const Eigen::Vector3d r_world = -use_radius * (foot_pose.rotation() * Eigen::Vector3d::UnitZ());
      const Eigen::Vector3d c_f = axis_world.cross(r_world);

      // Fill matrix row: tau coeff
      if (tau_col >= 0 && tau_col < static_cast<int>(nvar))
        eq->matrix()(static_cast<int>(ic), tau_col) = 1.0;

      // Fill contact force coeffs (assume contact_force_dim == 3 when available)
      if (contact_force_col0 >= 0 && contact_force_col0 + contact_force_dim <= static_cast<int>(nvar))
      {
        // we assume force vector order matches world x,y,z or contact components that map directly
        for (int k = 0; k < contact_force_dim && k < 3; ++k)
          eq->matrix()(static_cast<int>(ic), contact_force_col0 + k) = -c_f(static_cast<int>(k));
      }

      // Fill inertia term on ddq: ddq column equals idx_v (acceleration index)
      const int ddq_col = idx_v; // ddq vector includes base (0..5) and joints (6..)
      if (ddq_col >= 0 && ddq_col < static_cast<int>(nvar))
        eq->matrix()(static_cast<int>(ic), ddq_col) = -rc.inertia;
    }

    // Insert constraint at highest priority level (0)
    if (hqp.size() == 0)
      hqp.resize(1);
    hqp[0].push_back(tsid::solvers::aligned_pair<double, std::shared_ptr<tsid::math::ConstraintBase> >(1.0, eq));
  }

  // Non-holonomic rolling constraints: constrain tangential contact acceleration = 0
  // Can be disabled at runtime via settings_.enable_nonholonomic for quick testing.
  if (settings_.enable_nonholonomic && !nonholonomic_constraints_.empty())
  {
    // Ensure jacobians/time variation are computed
    data_handler_.updateInternalData(q_meas, v_meas, true);
    const unsigned int nvar = formulation_.nVar();
    const unsigned int rows = static_cast<unsigned int>(nonholonomic_constraints_.size());
    auto eq_nh = std::make_shared<tsid::math::ConstraintEquality>("nonholonomic-rolling", rows, nvar);
    Eigen::VectorXd b_nh = Eigen::VectorXd::Zero(rows);

    for (unsigned int ic = 0; ic < nonholonomic_constraints_.size(); ++ic)
    {
      const auto &nh = nonholonomic_constraints_[ic];
      // find foot index
      int foot_nb = -1;
      for (std::size_t f = 0; f < model_handler_.getFeetNb(); ++f)
      {
        if (model_handler_.getFootFrameName(f) == nh.contact_frame_name)
        {
          foot_nb = static_cast<int>(f);
          break;
        }
      }
      if (foot_nb < 0)
        continue;

      const pinocchio::FrameIndex frame_id = model_handler_.getFootFrameId(foot_nb);

      // Access internal data (non-const) to call Pinocchio functions
      const pinocchio::Data & data_const = data_handler_.getData();
      pinocchio::Data & data_nc = const_cast<pinocchio::Data &>(data_const);

      // Frame Jacobian (6 x nv): angular top 3, linear bottom 3
      Eigen::MatrixXd J6 = pinocchio::getFrameJacobian(model, data_nc, frame_id, pinocchio::LOCAL_WORLD_ALIGNED);

      // Frame Jacobian time variation (6 x nv)
      Eigen::MatrixXd dJ6 = Eigen::MatrixXd::Zero(6, model.nv);
      pinocchio::getFrameJacobianTimeVariation(model, data_nc, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, dJ6);

      const Eigen::MatrixXd J_lin = J6.block(3, 0, 3, model.nv);
      const Eigen::MatrixXd dJ_lin = dJ6.block(3, 0, 3, model.nv);

      // Tangential direction (wheel rolling direction) in world frame
      const pinocchio::SE3 & foot_pose = data_handler_.getFootPose(foot_nb);
      const Eigen::Vector3d tangent = foot_pose.rotation() * Eigen::Vector3d::UnitY();

      // row coefficients: tangent^T * J_lin  (1 x nv)
      Eigen::RowVectorXd rowCoeffs = tangent.transpose() * J_lin; // size nv

      // bias = tangent^T * dJ_lin * v_meas
      Eigen::VectorXd v_full = v_meas; // nv vector
      double bias = (tangent.transpose() * (dJ_lin * v_full))(0);

      // Fill eq row (only ddq columns, assumed to be first nv columns)
      for (int col = 0; col < model.nv; ++col)
      {
        eq_nh->matrix()(static_cast<int>(ic), col) = rowCoeffs(col);
      }
      b_nh(static_cast<int>(ic)) = -bias;
    }

    eq_nh->setVector(b_nh);
    if (hqp.size() == 0)
      hqp.resize(1);
    hqp[0].push_back(tsid::solvers::aligned_pair<double, std::shared_ptr<tsid::math::ConstraintBase> >(1.0, eq_nh));
  }

  // Solve using the possibly-augmented HQP data
  last_solution_ = solver_.solve(hqp);
  assert(last_solution_.status == tsid::solvers::HQPStatus::HQP_STATUS_OPTIMAL);
  tau_res = formulation_.getActuatorForces(last_solution_);

  // Legacy feed-forward torque from contact forces for wheels:
  // If the user did not register actuated-rolling constraints, add a small
  // feed-forward torque computed from contact forces so wheels slightly
  // over-actuate relative to the contact moment (avoid wheel lag).
  // if (rolling_constraints_.empty())
  // {
    const double wheel_radius = settings_.wheel_radius; // use settings-provided radius
    const double feedforward_scale = settings_.ff_wheel_scale; // use settings-provided scale
    for (std::size_t foot_nb = 0; foot_nb < model_handler_.getFeetNb(); foot_nb++)
    {
      if (!active_tsid_contacts_[foot_nb])
        continue;
      const std::string & frame_name = model_handler_.getFootFrameName(foot_nb);
      if (frame_name.find("wheel") == std::string::npos)
        continue;

      const Eigen::VectorXd f = formulation_.getContactForces(frame_name, last_solution_);
      if (f.size() < 3)
        continue;
      const Eigen::Vector3d f_world = f.head<3>();

      const pinocchio::FrameIndex frame_id = model_handler_.getFootFrameId(foot_nb);
      const pinocchio::JointIndex joint_id = model.frames[frame_id].parentJoint;
      const int idx_v = model.joints[joint_id].idx_v();
      const int nv_joint = model.joints[joint_id].nv();
      if (nv_joint != 1)
        continue;

      const pinocchio::SE3 & foot_pose = data_handler_.getFootPose(foot_nb);
      const Eigen::Vector3d axis_world = foot_pose.rotation() * Eigen::Vector3d::UnitY();
  const Eigen::Vector3d r_world = -wheel_radius * (foot_pose.rotation() * Eigen::Vector3d::UnitZ());
      const double tau_ff = (r_world.cross(f_world)).dot(axis_world);
      // actuator index in tau_res
      const int actuator_index = idx_v - 6;
      if (actuator_index >= 0 && actuator_index < static_cast<int>(tau_res.size()))
        tau_res[actuator_index] += feedforward_scale * tau_ff;

      std::cout << "[KinoID] wheel " << frame_name << " tau_ff=" << tau_ff
          << " scale=" << feedforward_scale << " add=" << (feedforward_scale * tau_ff) << std::endl;
    }
  }
// }

void KinodynamicsID::getAccelerations(Eigen::Ref<Eigen::VectorXd> ddq)
{
  ddq = formulation_.getAccelerations(last_solution_);
}
