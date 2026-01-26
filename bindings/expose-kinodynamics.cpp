#include "simple-mpc/kinodynamics.hpp"
#include "simple-mpc/python.hpp"

#include <eigenpy/std-map.hpp>

namespace simple_mpc::python
{

  auto * createKinodynamics(const bp::dict & settings, const RobotModelHandler & model_handler)
  {
    KinodynamicsSettings conf;
    conf.timestep = bp::extract<double>(settings["timestep"]);
    conf.w_x = bp::extract<Eigen::MatrixXd>(settings["w_x"]);
    conf.w_u = bp::extract<Eigen::MatrixXd>(settings["w_u"]);
    conf.w_cent = bp::extract<Eigen::MatrixXd>(settings["w_cent"]);
    conf.w_centder = bp::extract<Eigen::MatrixXd>(settings["w_centder"]);
    conf.w_frame = bp::extract<Eigen::VectorXd>(settings["w_frame"]);

    conf.gravity = bp::extract<Eigen::Vector3d>(settings["gravity"]);
    conf.force_size = bp::extract<int>(settings["force_size"]);

    conf.qmin = bp::extract<Eigen::VectorXd>(settings["qmin"]);
    conf.qmax = bp::extract<Eigen::VectorXd>(settings["qmax"]);

    conf.mu = bp::extract<double>(settings["mu"]);
    conf.Lfoot = bp::extract<double>(settings["Lfoot"]);
    conf.Wfoot = bp::extract<double>(settings["Wfoot"]);

    conf.kinematics_limits = bp::extract<bool>(settings["kinematics_limits"]);
    conf.force_cone = bp::extract<bool>(settings["force_cone"]);
    conf.land_cstr = bp::extract<bool>(settings["land_cstr"]);
    conf.nonholonomic_rolling = bp::extract<bool>(settings["nonholonomic_rolling"]);
    conf.enable_lateral_no_slip = bp::extract<bool>(settings["enable_lateral_no_slip"]);
    conf.lateral_no_slip_min_axis_norm = bp::extract<double>(settings["lateral_no_slip_min_axis_norm"]);
    conf.lateral_no_slip_min_cross_norm = bp::extract<double>(settings["lateral_no_slip_min_cross_norm"]);
    conf.use_vector_glide_cost = bp::extract<bool>(settings["use_vector_glide_cost"]);
    conf.vector_glide_weight = bp::extract<double>(settings["vector_glide_weight"]);
    conf.vector_glide_min_omega = bp::extract<double>(settings["vector_glide_min_omega"]);
    conf.min_wheel_distance_cstr = bp::extract<bool>(settings["min_wheel_distance_cstr"]);
    conf.min_wheel_distance = bp::extract<double>(settings["min_wheel_distance"]);
    conf.min_wheel_distance_cost = bp::extract<bool>(settings["min_wheel_distance_cost"]);
    conf.w_min_wheel_distance = bp::extract<double>(settings["w_min_wheel_distance"]);
    conf.min_wheel_distance_cost_eps = bp::extract<double>(settings["min_wheel_distance_cost_eps"]);
    conf.force_z_variance_cost = bp::extract<bool>(settings["force_z_variance_cost"]);
    conf.w_force_z_variance = bp::extract<double>(settings["w_force_z_variance"]);
    conf.joint_limit_soft_cost = bp::extract<bool>(settings["joint_limit_soft_cost"]);
    conf.w_joint_limit_soft = bp::extract<double>(settings["w_joint_limit_soft"]);
    conf.joint_limit_soft_fraction = bp::extract<double>(settings["joint_limit_soft_fraction"]);
    conf.soft_constraints = bp::extract<bool>(settings["soft_constraints"]);
    conf.w_soft_contact_vel = bp::extract<double>(settings["w_soft_contact_vel"]);
    conf.w_soft_friction = bp::extract<double>(settings["w_soft_friction"]);
    conf.w_soft_land = bp::extract<double>(settings["w_soft_land"]);
    conf.track_width_cstr = bp::extract<bool>(settings["track_width_cstr"]);
    conf.w_track_width = bp::extract<double>(settings["w_track_width"]);
    conf.foot_sum_cstr = bp::extract<bool>(settings["foot_sum_cstr"]);
    conf.w_foot_sum = bp::extract<double>(settings["w_foot_sum"]);
    conf.foot_sum_z_offset = bp::extract<double>(settings["foot_sum_z_offset"]);
    if (settings.has_key("foot_height_cstr"))
      conf.foot_height_cstr = bp::extract<bool>(settings["foot_height_cstr"]);
    if (settings.has_key("foot_height"))
      conf.foot_height = bp::extract<double>(settings["foot_height"]);

    return new KinodynamicsOCP(conf, model_handler);
  }

  bp::dict getSettingsKino(KinodynamicsOCP & self)
  {
    KinodynamicsSettings conf = self.getSettings();
    bp::dict settings;
    settings["timestep"] = conf.timestep;
    settings["w_x"] = conf.w_x;
    settings["w_u"] = conf.w_u;
    settings["w_cent"] = conf.w_cent;
    settings["w_centder"] = conf.w_centder;
    settings["w_frame"] = conf.w_frame;
    settings["gravity"] = conf.gravity;
    settings["force_size"] = conf.force_size;
    settings["qmin"] = conf.qmin;
    settings["qmax"] = conf.qmax;
    settings["mu"] = conf.mu;
    settings["Lfoot"] = conf.Lfoot;
    settings["Wfoot"] = conf.Wfoot;
    settings["kinematics_limits"] = conf.kinematics_limits;
    settings["force_cone"] = conf.force_cone;
    settings["land_cstr"] = conf.land_cstr;
    settings["nonholonomic_rolling"] = conf.nonholonomic_rolling;
    settings["enable_lateral_no_slip"] = conf.enable_lateral_no_slip;
    settings["lateral_no_slip_min_axis_norm"] = conf.lateral_no_slip_min_axis_norm;
    settings["lateral_no_slip_min_cross_norm"] = conf.lateral_no_slip_min_cross_norm;
    settings["use_vector_glide_cost"] = conf.use_vector_glide_cost;
    settings["vector_glide_weight"] = conf.vector_glide_weight;
    settings["vector_glide_min_omega"] = conf.vector_glide_min_omega;
    settings["min_wheel_distance_cstr"] = conf.min_wheel_distance_cstr;
    settings["min_wheel_distance"] = conf.min_wheel_distance;
    settings["min_wheel_distance_cost"] = conf.min_wheel_distance_cost;
    settings["w_min_wheel_distance"] = conf.w_min_wheel_distance;
    settings["min_wheel_distance_cost_eps"] = conf.min_wheel_distance_cost_eps;
    settings["force_z_variance_cost"] = conf.force_z_variance_cost;
    settings["w_force_z_variance"] = conf.w_force_z_variance;
    settings["joint_limit_soft_cost"] = conf.joint_limit_soft_cost;
    settings["w_joint_limit_soft"] = conf.w_joint_limit_soft;
    settings["joint_limit_soft_fraction"] = conf.joint_limit_soft_fraction;
    settings["soft_constraints"] = conf.soft_constraints;
    settings["w_soft_contact_vel"] = conf.w_soft_contact_vel;
    settings["w_soft_friction"] = conf.w_soft_friction;
    settings["w_soft_land"] = conf.w_soft_land;
    settings["track_width_cstr"] = conf.track_width_cstr;
    settings["w_track_width"] = conf.w_track_width;
    settings["foot_sum_cstr"] = conf.foot_sum_cstr;
    settings["w_foot_sum"] = conf.w_foot_sum;
    settings["foot_sum_z_offset"] = conf.foot_sum_z_offset;
    settings["foot_height_cstr"] = conf.foot_height_cstr;
    settings["foot_height"] = conf.foot_height;

    return settings;
  }

  StageModel createKinoStage(
    KinodynamicsOCP & self,
    const bp::dict & phase_dict,
    const bp::dict & pose_dict,
    const bp::dict & force_dict,
    const bp::dict & land_dict)
  {
    bp::list phase_keys = bp::list(phase_dict.keys());
    bp::list pose_keys = bp::list(pose_dict.keys());
    bp::list force_keys = bp::list(force_dict.keys());
    bp::list land_keys = bp::list(land_dict.keys());
    std::map<std::string, bool> phase_contact;
    std::map<std::string, pinocchio::SE3> pose_contact;
    std::map<std::string, Eigen::VectorXd> force_contact;
    std::map<std::string, bool> land_constraint;
    for (int i = 0; i < len(phase_keys); ++i)
    {
      bp::extract<std::string> extractor(phase_keys[i]);
      if (extractor.check())
      {
        std::string key = extractor();
        bool ff = bp::extract<bool>(phase_dict[key]);
        phase_contact.insert({key, ff});
      }
    }
    for (int i = 0; i < len(pose_keys); ++i)
    {
      bp::extract<std::string> extractor(pose_keys[i]);
      if (extractor.check())
      {
        std::string key = extractor();
        pinocchio::SE3 ff = bp::extract<pinocchio::SE3>(pose_dict[key]);
        pose_contact.insert({key, ff});
      }
    }
    for (int i = 0; i < len(force_keys); ++i)
    {
      bp::extract<std::string> extractor(force_keys[i]);
      if (extractor.check())
      {
        std::string key = extractor();
        Eigen::VectorXd ff = bp::extract<Eigen::VectorXd>(force_dict[key]);
        force_contact.insert({key, ff});
      }
    }
    for (int i = 0; i < len(land_keys); ++i)
    {
      bp::extract<std::string> extractor(land_keys[i]);
      if (extractor.check())
      {
        std::string key = extractor();
        bool ff = bp::extract<bool>(land_dict[key]);
        land_constraint.insert({key, ff});
      }
    }

    return self.createStage(phase_contact, pose_contact, force_contact, land_constraint);
  }

  void exposeKinodynamicsOcp()
  {
    bp::register_ptr_to_python<shared_ptr<KinodynamicsOCP>>();

    bp::class_<KinodynamicsOCP, bp::bases<OCPHandler>, boost::noncopyable>("KinodynamicsOCP", bp::no_init)
      .def(
        "__init__",
        bp::make_constructor(&createKinodynamics, bp::default_call_policies(), ("settings"_a, "model_handler")))
      .def("getSettings", &getSettingsKino)
      .def("createStage", &createKinoStage);
  }

} // namespace simple_mpc::python
