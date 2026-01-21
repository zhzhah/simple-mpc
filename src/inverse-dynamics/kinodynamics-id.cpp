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
  postureTask_->Kp(settings_.kp_posture * Eigen::VectorXd::Ones(nu));
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
  last_solution_ = solver_.solve(solver_data);
  assert(last_solution_.status == tsid::solvers::HQPStatus::HQP_STATUS_OPTIMAL);
  tau_res = formulation_.getActuatorForces(last_solution_);

  const double wheel_radius = 0.0762;
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
    tau_res[idx_v - 6] += tau_ff;
  }
}

void KinodynamicsID::getAccelerations(Eigen::Ref<Eigen::VectorXd> ddq)
{
  ddq = formulation_.getAccelerations(last_solution_);
}
