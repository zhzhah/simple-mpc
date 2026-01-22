#pragma once

#include <simple-mpc/robot-handler.hpp>
#include <tsid/contacts/contact-point.hpp>
#include <tsid/formulations/inverse-dynamics-formulation-acc-force.hpp>
#include <tsid/solvers/solver-proxqp.hpp>
#include <tsid/solvers/utils.hpp>
#include <tsid/tasks/task-actuation-bounds.hpp>
#include <tsid/tasks/task-joint-posVelAcc-bounds.hpp>
#include <tsid/tasks/task-joint-posture.hpp>
#include <tsid/tasks/task-se3-equality.hpp>
#include <tsid/trajectories/trajectory-base.hpp>

namespace simple_mpc
{

  class KinodynamicsID
  {
  public:
    struct ActuatedRollingConstraint {
      std::string contact_frame_name; // contact frame (foot) name
      double radius = 0.0;           // wheel radius
      double inertia = 0.0;          // wheel rotational inertia (optional)
      ActuatedRollingConstraint() = default;
      ActuatedRollingConstraint(const std::string &n, double r, double I)
        : contact_frame_name(n), radius(r), inertia(I) {}
    };

    /// Register an actuated rolling constraint for a wheel/contact frame.
    /// The constraint enforces: tau_wheel - c_f^T * f_contact - inertia * ddq_wheel = 0
    void addActuatedRollingConstraint(const std::string &contact_frame_name, double radius, double inertia = 0.0);

    typedef Eigen::VectorXd TargetContactForce;

    struct NonHolonomicRollingConstraint {
      std::string contact_frame_name; // contact frame (foot) name
      double radius = 0.0;           // wheel radius (used to overwrite jacobian entry)
      NonHolonomicRollingConstraint() = default;
      NonHolonomicRollingConstraint(const std::string &n, double r)
        : contact_frame_name(n), radius(r) {}
    };

    /// Register a non-holonomic rolling constraint on a contact frame.
    /// The constraint enforces tangential contact acceleration = 0 at the contact point.
    /// (J_modified * ddq = -dotJ * qdot)
    void addNonHolonomicRollingConstraint(const std::string &contact_frame_name, double radius);

// ...
    struct Settings
    {

      // Physical quantities
      double friction_coefficient = 0.6;

      double contact_weight_ratio_max =
        10.0; // Max force for one foot contact (express as a multiple of the robot weight)

      double contact_weight_ratio_min =
        0.01; // Min force for one foot contact (express as a multiple of the robot weight)

      // Tasks gains
      double kp_base = 0.;
      double kp_posture = 0.;
  double kp_posture_wheel = 0.;
      double kp_contact = 0.;
  // Wheel-specific physical/settings
  double wheel_radius = 0.0762;    // default wheel radius (meters)
  double ff_wheel_scale = 1.1;     // feed-forward scale for wheel torque

  // Enable/disable adding non-holonomic rolling equality constraints
  // when solving the HQP. Set to false to temporarily disable them for testing.
  bool enable_nonholonomic = true;

      // Tasks weights
      double w_base = -1.;           // Disabled by default
      double w_posture = -1.;        // Disabled by default
  double w_posture_wheel = -1.;  // Disabled by default (wheel-specific posture weight)
      double w_contact_motion = -1.; // Disabled by default
      double w_contact_force = -1.;  // Disabled by default

      ///< Are the contact motion = 0 handled as a hard contraint (true) or a cost (if false)
      bool contact_motion_equality = false;
    };

    KinodynamicsID(const simple_mpc::RobotModelHandler & model_handler, double control_dt, const Settings settings);
    KinodynamicsID(const KinodynamicsID &) = delete;

    void setTarget(
      const Eigen::Ref<const Eigen::VectorXd> & q_target,
      const Eigen::Ref<const Eigen::VectorXd> & v_target,
      const Eigen::Ref<const Eigen::VectorXd> & a_target,
      const std::vector<bool> & contact_state_target,
      const std::vector<TargetContactForce> & f_target);

    void solve(
      const double t,
      const Eigen::Ref<const Eigen::VectorXd> & q_meas,
      const Eigen::Ref<const Eigen::VectorXd> & v_meas,
      Eigen::Ref<Eigen::VectorXd> tau_res);

    void getAccelerations(Eigen::Ref<Eigen::VectorXd> a);

  public:
    // Order matters to be instantiated in the right order
    const Settings settings_;
    const simple_mpc::RobotModelHandler & model_handler_;

  protected:
    // Order still matters here
    simple_mpc::RobotDataHandler data_handler_;
    tsid::robots::RobotWrapper robot_;
    tsid::InverseDynamicsFormulationAccForce formulation_;

    std::vector<bool> active_tsid_contacts_;
    std::vector<std::unique_ptr<tsid::contacts::ContactBase>> tsid_contacts;
    std::unique_ptr<tsid::tasks::TaskJointPosture> postureTask_;
    std::unique_ptr<tsid::tasks::TaskSE3Equality> baseTask_;
    std::unique_ptr<tsid::tasks::TaskJointPosVelAccBounds> boundsTask_;
    std::unique_ptr<tsid::tasks::TaskActuationBounds> actuationTask_;
    tsid::solvers::SolverProxQP solver_;
    tsid::solvers::HQPOutput last_solution_;
    tsid::trajectories::TrajectorySample samplePosture_;
    tsid::trajectories::TrajectorySample sampleBase_;
    pinocchio::Motion targetVelBase_;
    pinocchio::Motion targetAccBase_;
    Eigen::VectorXd q_target_;
    Eigen::VectorXd v_target_;
    Eigen::VectorXd a_target_;
    std::vector<ActuatedRollingConstraint> rolling_constraints_;
    std::vector<NonHolonomicRollingConstraint> nonholonomic_constraints_;
  };

} // namespace simple_mpc
