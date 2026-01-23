///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include "aligator/core/function-abstract.hpp"
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

namespace simple_mpc
{
  template <typename _Scalar>
  struct VectorGlideResidualTpl : aligator::StageFunctionTpl<_Scalar>
  {
    using Scalar = _Scalar;
    ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
    using Base = aligator::StageFunctionTpl<Scalar>;
    using Data = aligator::StageFunctionDataTpl<Scalar>;

    VectorGlideResidualTpl(
      const int ndx,
      const int nu,
      const pinocchio::Model & model,
      const pinocchio::FrameIndex base_frame,
      const pinocchio::FrameIndex foot_frame,
      const Eigen::VectorXd * commanded_base_vel,
      const double min_axis_norm,
      const double min_cross_norm,
      const double min_omega)
    : Base(ndx, nu, 1)
    , model_(model)
    , base_frame_(base_frame)
    , foot_frame_(foot_frame)
    , commanded_base_vel_(commanded_base_vel)
    , min_axis_norm_(min_axis_norm)
    , min_cross_norm_(min_cross_norm)
    , min_omega_(min_omega)
    , data_(model)
    {
    }

    void evaluate(const ConstVectorRef & x, const ConstVectorRef &, Data & data) const override
    {
      data.value_.setZero();
      const int nq = model_.nq;
      const int nv = model_.nv;
      if (x.size() < nq + nv) return;

      const Eigen::VectorXd q = x.head(nq);
      const Eigen::VectorXd v = x.tail(nv);
      data.value_[0] = computeResidual(q, v);
    }

    void computeJacobians(const ConstVectorRef & x, const ConstVectorRef &, Data & data) const override
    {
      data.Jx_.setZero();
      data.Ju_.setZero();
      const int nq = model_.nq;
      const int nv = model_.nv;
      if (x.size() < nq + nv) return;

      const Eigen::VectorXd q = x.head(nq);
      const Eigen::VectorXd v = x.tail(nv);
      const double res0 = computeResidual(q, v);

      constexpr double eps = 1e-6;
      const int ndx = static_cast<int>(data.Jx_.cols());
      for (int k = 0; k < ndx; ++k)
      {
        Eigen::VectorXd q_plus = q;
        Eigen::VectorXd v_plus = v;
        if (k < nv)
        {
          Eigen::VectorXd dq = Eigen::VectorXd::Zero(nv);
          dq[k] = eps;
          q_plus = pinocchio::integrate(model_, q, dq);
        }
        else
        {
          v_plus[k - nv] += eps;
        }
        const double res_plus = computeResidual(q_plus, v_plus);
        data.Jx_(0, k) = (res_plus - res0) / eps;
      }
    }

    double computeResidual(const Eigen::VectorXd & q, const Eigen::VectorXd & v) const
    {
      if (commanded_base_vel_ == nullptr || commanded_base_vel_->size() < 6) return 0.0;
      const double v_des = (*commanded_base_vel_)(0);
      const double omega_des = (*commanded_base_vel_)(5);
      if (std::abs(omega_des) < min_omega_) return 0.0;

      pinocchio::forwardKinematics(model_, data_, q, v);
      pinocchio::updateFramePlacements(model_, data_);

      const pinocchio::SE3 & base_pose = data_.oMf[base_frame_];
      const Eigen::Vector3d r_base = base_pose.translation();
      const Eigen::Matrix3d R_base = base_pose.rotation();
      const Eigen::Vector3d icr_local(0.0, v_des / omega_des, 0.0);
      const Eigen::Vector3d r_O_star = r_base + R_base * icr_local;

      const pinocchio::SE3 & foot_pose = data_.oMf[foot_frame_];
      Eigen::Vector3d a_y = foot_pose.rotation().col(1);
      if (a_y.norm() < min_axis_norm_)
        a_y = Eigen::Vector3d::UnitY();
      else
        a_y.normalize();

      const Eigen::Vector3d g_z = Eigen::Vector3d::UnitZ();
      Eigen::Vector3d c_x = a_y.cross(g_z);
      if (c_x.norm() < min_cross_norm_)
      {
        Eigen::Vector3d a_x = foot_pose.rotation().col(0);
        c_x = a_x - g_z * a_x.dot(g_z);
      }
      if (c_x.norm() < min_cross_norm_)
        c_x = Eigen::Vector3d::UnitX();
      c_x.normalize();

      const Eigen::Vector3d r_contact = foot_pose.translation();
      const Eigen::Vector3d rho = r_O_star - r_contact;
      const double rho_norm = rho.norm();
      if (rho_norm < 1e-9) return 0.0;

      const double e_i = rho.dot(c_x);
      return e_i / rho_norm;
    }

    const pinocchio::Model & model_;
    pinocchio::FrameIndex base_frame_;
    pinocchio::FrameIndex foot_frame_;
    const Eigen::VectorXd * commanded_base_vel_;
    double min_axis_norm_;
    double min_cross_norm_;
    double min_omega_;
    mutable pinocchio::Data data_;
  };

  using VectorGlideResidual = VectorGlideResidualTpl<double>;

} // namespace simple_mpc
