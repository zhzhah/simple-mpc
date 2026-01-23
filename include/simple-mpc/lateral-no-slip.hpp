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
  struct LateralNoSlipResidualTpl : aligator::StageFunctionTpl<_Scalar>
  {
    using Scalar = _Scalar;
    ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
    using Base = aligator::StageFunctionTpl<Scalar>;
    using Data = aligator::StageFunctionDataTpl<Scalar>;

    LateralNoSlipResidualTpl(
      const int ndx,
      const int nu,
      const pinocchio::Model & model,
      const pinocchio::FrameIndex frame_id,
      const double min_axis_norm,
      const double min_cross_norm)
    : Base(ndx, nu, 1)
    , model_(model)
    , frame_id_(frame_id)
    , min_axis_norm_(min_axis_norm)
    , min_cross_norm_(min_cross_norm)
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
      pinocchio::forwardKinematics(model_, data_, q, v);
      pinocchio::updateFramePlacements(model_, data_);

      const pinocchio::SE3 & foot_pose = data_.oMf[frame_id_];
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

      Eigen::Vector3d c_y = g_z.cross(c_x);
      if (c_y.norm() < min_cross_norm_)
        c_y = Eigen::Vector3d::UnitY();
      else
        c_y.normalize();

      const pinocchio::Motion v_frame =
        pinocchio::getFrameVelocity(model_, data_, frame_id_, pinocchio::LOCAL_WORLD_ALIGNED);
      return v_frame.linear().dot(c_y);
    }

    const pinocchio::Model & model_;
    pinocchio::FrameIndex frame_id_;
    double min_axis_norm_;
    double min_cross_norm_;
    mutable pinocchio::Data data_;
  };

  using LateralNoSlipResidual = LateralNoSlipResidualTpl<double>;

} // namespace simple_mpc
