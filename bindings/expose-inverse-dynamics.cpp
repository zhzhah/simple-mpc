#include <eigenpy/eigenpy.hpp>

#include "simple-mpc/inverse-dynamics/centroidal-id.hpp"
#include "simple-mpc/inverse-dynamics/kinodynamics-id.hpp"

namespace simple_mpc
{
  namespace python
  {
    namespace bp = boost::python;

    Eigen::VectorXd solveKinoProxy(
      KinodynamicsID & self,
      const double t,
      const Eigen::Ref<const Eigen::VectorXd> & q_meas,
      const Eigen::Ref<const Eigen::VectorXd> & v_meas)
    {
      Eigen::VectorXd tau_res(self.model_handler_.getModel().nv - 6);
      self.solve(t, q_meas, v_meas, tau_res);
      return tau_res;
    }

    Eigen::VectorXd getAccelerationsKinoProxy(KinodynamicsID & self)
    {
      Eigen::VectorXd a(self.model_handler_.getModel().nv);
      self.getAccelerations(a);
      return a;
    }

    Eigen::VectorXd solveCentroidalProxy(
      CentroidalID & self,
      const double t,
      const Eigen::Ref<const Eigen::VectorXd> & q_meas,
      const Eigen::Ref<const Eigen::VectorXd> & v_meas)
    {
      Eigen::VectorXd tau_res(self.model_handler_.getModel().nv - 6);
      self.solve(t, q_meas, v_meas, tau_res);
      return tau_res;
    }

    Eigen::VectorXd getAccelerationsCentroidalProxy(CentroidalID & self)
    {
      Eigen::VectorXd a(self.model_handler_.getModel().nv);
      self.getAccelerations(a);
      return a;
    }

    void setTarget_CentroidalID(
      CentroidalID & self,
      const Eigen::Ref<const Eigen::Vector<double, 3>> & com_position,
      const Eigen::Ref<const Eigen::Vector<double, 3>> & com_velocity,
      const CentroidalID::FeetPoseVector & feet_pose,
      const CentroidalID::FeetVelocityVector & feet_velocity,
      const std::vector<bool> & contact_state_target,
      const std::vector<CentroidalID::TargetContactForce> & f_target)
    {
      self.setTarget(com_position, com_velocity, feet_pose, feet_velocity, contact_state_target, f_target);
    }

   void exposeInverseDynamics()
    {
      bp::class_<KinodynamicsID::Settings>("KinodynamicsIDSettings", bp::init<>(bp::args("self")))
        .def_readwrite("friction_coefficient", &KinodynamicsID::Settings::friction_coefficient)
        .def_readwrite("contact_weight_ratio_max", &KinodynamicsID::Settings::contact_weight_ratio_max)
        .def_readwrite("contact_weight_ratio_min", &KinodynamicsID::Settings::contact_weight_ratio_min)
        .def_readwrite("kp_base", &KinodynamicsID::Settings::kp_base)
        .def_readwrite("kp_posture", &KinodynamicsID::Settings::kp_posture)
        .def_readwrite("kp_posture_wheel", &KinodynamicsID::Settings::kp_posture_wheel)
        .def_readwrite("kp_contact", &KinodynamicsID::Settings::kp_contact)
        .def_readwrite("wheel_radius", &KinodynamicsID::Settings::wheel_radius)
        .def_readwrite("ff_wheel_scale", &KinodynamicsID::Settings::ff_wheel_scale)
        .def_readwrite("w_base", &KinodynamicsID::Settings::w_base)
        .def_readwrite("w_posture", &KinodynamicsID::Settings::w_posture)
        .def_readwrite("w_posture_wheel", &KinodynamicsID::Settings::w_posture_wheel)
        .def_readwrite("w_contact_motion", &KinodynamicsID::Settings::w_contact_motion)
        .def_readwrite("w_contact_force", &KinodynamicsID::Settings::w_contact_force)
        .def_readwrite("contact_motion_equality", &KinodynamicsID::Settings::contact_motion_equality)
        .def_readwrite("enable_nonholonomic", &KinodynamicsID::Settings::enable_nonholonomic)
        .def_readwrite("enable_lateral_no_slip", &KinodynamicsID::Settings::enable_lateral_no_slip)
        .def_readwrite("lateral_no_slip_min_axis_norm", &KinodynamicsID::Settings::lateral_no_slip_min_axis_norm)
        .def_readwrite("lateral_no_slip_min_cross_norm", &KinodynamicsID::Settings::lateral_no_slip_min_cross_norm)
        .def_readwrite("lateral_no_slip_use_bounds", &KinodynamicsID::Settings::lateral_no_slip_use_bounds)
        .def_readwrite("lateral_no_slip_lower", &KinodynamicsID::Settings::lateral_no_slip_lower)
        .def_readwrite("lateral_no_slip_upper", &KinodynamicsID::Settings::lateral_no_slip_upper)
        .def_readwrite("use_vector_glide_cost", &KinodynamicsID::Settings::use_vector_glide_cost)
        .def_readwrite("vector_glide_weight", &KinodynamicsID::Settings::vector_glide_weight);

      bp::class_<KinodynamicsID, boost::noncopyable>(
        "KinodynamicsID", bp::init<const simple_mpc::RobotModelHandler &, double, const KinodynamicsID::Settings>(
                            bp::args("self", "model_handler", "control_dt", "settings")))
        .def("setTarget", &KinodynamicsID::setTarget) // <--- 注意这里千万不能有分号
        .def("addNonHolonomicRollingConstraint", &KinodynamicsID::addNonHolonomicRollingConstraint,
             bp::args("self", "contact_frame_name", "radius"),
             "Add a non-holonomic rolling constraint for a wheel.")
        .def("addLateralNoSlipConstraint", &KinodynamicsID::addLateralNoSlipConstraint,
             bp::args("self", "contact_frame_name"),
             "Add a lateral no-slip constraint for a wheel.")
        .def("solve", &solveKinoProxy)
        .def("getAccelerations", &getAccelerationsKinoProxy); // <--- 分号只能在这最后一行

      bp::class_<CentroidalID::Settings, bp::bases<KinodynamicsID::Settings>>(
        "CentroidalIDSettings", bp::init<>(bp::args("self")))
        .def_readwrite("kp_com", &CentroidalID::Settings::kp_com)
        .def_readwrite("kp_feet_tracking", &CentroidalID::Settings::kp_feet_tracking)
        .def_readwrite("w_com", &CentroidalID::Settings::w_com)
        .def_readwrite("w_feet_tracking", &CentroidalID::Settings::w_feet_tracking);

      bp::class_<CentroidalID, boost::noncopyable>(
        "CentroidalID", bp::init<const simple_mpc::RobotModelHandler &, double, const CentroidalID::Settings>(
                          bp::args("self", "model_handler", "control_dt", "settings")))
        .def("setTarget", &setTarget_CentroidalID)
        .def("solve", &solveCentroidalProxy)
        .def("getAccelerations", &getAccelerationsCentroidalProxy);
    }
  } // namespace python
} // namespace simple_mpc
