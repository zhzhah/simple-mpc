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
  using namespace aligator;

  template<typename _Scalar>
  struct ICRArcResidualTpl : aligator::StageFunctionTpl<_Scalar>
  {
    using Scalar = _Scalar;
    using Base = aligator::StageFunctionTpl<Scalar>;
    using Data = typename Base::Data;
    using VectorXs = typename Base::VectorXs;
    using ConstVectorRef = typename Base::ConstVectorRef;

    ICRArcResidualTpl(
      const int ndx,
      const int nu,
      const pinocchio::Model & model,
      const pinocchio::FrameIndex base_frame,
      const Eigen::Vector4d * icr_params)
    : Base(ndx, nu, 1)
    , model_(model)
    , base_frame_(base_frame)
    , icr_params_(icr_params)
    , data_(model)
    {
    }

    void evaluate(const ConstVectorRef & x, const ConstVectorRef & u, Data & data) const
    {
      (void)u;
      data.value_.setZero();
      if (icr_params_ == nullptr)
        return;
      if ((*icr_params_)(3) < 0.5)
        return;

      const int nq = model_.nq;
      const int nv = model_.nv;
      Eigen::VectorXd q = x.head(nq);
      Eigen::VectorXd v = x.tail(nv);
      pinocchio::forwardKinematics(model_, data_, q, v);
      pinocchio::updateFramePlacements(model_, data_);
      const pinocchio::SE3 & base_pose = data_.oMf[base_frame_];
      const Eigen::Vector3d r_base = base_pose.translation();

      const double cx = (*icr_params_)(0);
      const double cy = (*icr_params_)(1);
      const double R = (*icr_params_)(2);
      const double dx = r_base.x() - cx;
      const double dy = r_base.y() - cy;
      data.value_[0] = dx * dx + dy * dy - R * R;
    }

    void computeJacobians(const ConstVectorRef & x, const ConstVectorRef & u, Data & data) const
    {
      (void)u;
      const double eps = 1e-7;
      data.Jx_.setZero();
      data.Ju_.setZero();
      const double res0_0 = data.value_[0];
      const int nv = model_.nv;
      VectorXs q = x.head(model_.nq);
      VectorXs v = x.tail(nv);
      VectorXs dq = VectorXs::Zero(nv);
      VectorXs dv = VectorXs::Zero(nv);

      for (int k = 0; k < 2 * nv; ++k)
      {
        VectorXs q_plus = q;
        VectorXs v_plus = v;
        if (k < nv)
        {
          dq.setZero();
          dq[k] = eps;
          q_plus = pinocchio::integrate(model_, q, dq);
        }
        else
        {
          dv.setZero();
          dv[k - nv] = eps;
          v_plus = v + dv;
        }
        pinocchio::forwardKinematics(model_, data_, q_plus, v_plus);
        pinocchio::updateFramePlacements(model_, data_);
        const pinocchio::SE3 & base_pose = data_.oMf[base_frame_];
        const Eigen::Vector3d r_base = base_pose.translation();
        double res_plus_0 = 0.0;
        if (icr_params_ != nullptr && (*icr_params_)(3) >= 0.5)
        {
          const double cx = (*icr_params_)(0);
          const double cy = (*icr_params_)(1);
          const double R = (*icr_params_)(2);
          const double dx = r_base.x() - cx;
          const double dy = r_base.y() - cy;
          res_plus_0 = dx * dx + dy * dy - R * R;
        }
        data.Jx_(0, k) = (res_plus_0 - res0_0) / eps;
      }
    }

    const pinocchio::Model & model_;
    pinocchio::FrameIndex base_frame_;
    const Eigen::Vector4d * icr_params_;
    mutable pinocchio::Data data_;
  };

  using ICRArcResidual = ICRArcResidualTpl<double>;

} // namespace simple_mpc
