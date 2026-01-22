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
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>

namespace simple_mpc
{
  template <typename _Scalar>
  struct TrackWidthResidualTpl : aligator::StageFunctionTpl<_Scalar>
  {
    using Scalar = _Scalar;
    ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
    using Base = aligator::StageFunctionTpl<Scalar>;
    using Data = aligator::StageFunctionDataTpl<Scalar>;

    TrackWidthResidualTpl(
      const int ndx,
      const int nu,
      const pinocchio::Model & model,
      const pinocchio::FrameIndex base_frame,
      const pinocchio::FrameIndex fl_frame,
      const pinocchio::FrameIndex fr_frame,
      const pinocchio::FrameIndex rl_frame,
      const pinocchio::FrameIndex rr_frame,
      Scalar target_front,
      Scalar target_rear)
    : Base(ndx, nu, 2)
    , model_(model)
    , base_frame_(base_frame)
    , fl_frame_(fl_frame)
    , fr_frame_(fr_frame)
    , rl_frame_(rl_frame)
    , rr_frame_(rr_frame)
    , target_front_(target_front)
    , target_rear_(target_rear)
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
      pinocchio::forwardKinematics(model_, data_, q, v);
      pinocchio::updateFramePlacements(model_, data_);

      const pinocchio::SE3 & oMb = data_.oMf[base_frame_];
      const Eigen::Matrix3d Rb = oMb.rotation();
      const Eigen::Vector3d pb = oMb.translation();

      auto y_in_base = [&](pinocchio::FrameIndex fid) -> Scalar {
        const Eigen::Vector3d pw = data_.oMf[fid].translation();
        const Eigen::Vector3d pb_rel = Rb.transpose() * (pw - pb);
        return pb_rel.y();
      };

      const Scalar y_fl = y_in_base(fl_frame_);
      const Scalar y_fr = y_in_base(fr_frame_);
      const Scalar y_rl = y_in_base(rl_frame_);
      const Scalar y_rr = y_in_base(rr_frame_);

      data.value_[0] = (y_fl - y_fr) - target_front_;
      data.value_[1] = (y_rl - y_rr) - target_rear_;
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

      // Finite-difference on configuration only (residual does not depend on v).
      constexpr double eps = 1e-6;
      Eigen::VectorXd res0(2);
      {
        pinocchio::forwardKinematics(model_, data_, q, v);
        pinocchio::updateFramePlacements(model_, data_);
        const pinocchio::SE3 & oMb = data_.oMf[base_frame_];
        const Eigen::Matrix3d Rb = oMb.rotation();
        const Eigen::Vector3d pb = oMb.translation();

        auto y_in_base = [&](pinocchio::FrameIndex fid) -> Scalar {
          const Eigen::Vector3d pw = data_.oMf[fid].translation();
          const Eigen::Vector3d pb_rel = Rb.transpose() * (pw - pb);
          return pb_rel.y();
        };

        const Scalar y_fl = y_in_base(fl_frame_);
        const Scalar y_fr = y_in_base(fr_frame_);
        const Scalar y_rl = y_in_base(rl_frame_);
        const Scalar y_rr = y_in_base(rr_frame_);

        res0[0] = (y_fl - y_fr) - target_front_;
        res0[1] = (y_rl - y_rr) - target_rear_;
      }

      for (int k = 0; k < nv; ++k)
      {
        Eigen::VectorXd dq = Eigen::VectorXd::Zero(nv);
        dq[k] = eps;
        const Eigen::VectorXd q_plus = pinocchio::integrate(model_, q, dq);

        pinocchio::forwardKinematics(model_, data_, q_plus, v);
        pinocchio::updateFramePlacements(model_, data_);
        const pinocchio::SE3 & oMb = data_.oMf[base_frame_];
        const Eigen::Matrix3d Rb = oMb.rotation();
        const Eigen::Vector3d pb = oMb.translation();

        auto y_in_base = [&](pinocchio::FrameIndex fid) -> Scalar {
          const Eigen::Vector3d pw = data_.oMf[fid].translation();
          const Eigen::Vector3d pb_rel = Rb.transpose() * (pw - pb);
          return pb_rel.y();
        };

        const Scalar y_fl = y_in_base(fl_frame_);
        const Scalar y_fr = y_in_base(fr_frame_);
        const Scalar y_rl = y_in_base(rl_frame_);
        const Scalar y_rr = y_in_base(rr_frame_);

        Eigen::VectorXd res_plus(2);
        res_plus[0] = (y_fl - y_fr) - target_front_;
        res_plus[1] = (y_rl - y_rr) - target_rear_;

        data.Jx_.col(k) = (res_plus - res0) / eps;
      }
    }

    const pinocchio::Model & model_;
    pinocchio::FrameIndex base_frame_;
    pinocchio::FrameIndex fl_frame_;
    pinocchio::FrameIndex fr_frame_;
    pinocchio::FrameIndex rl_frame_;
    pinocchio::FrameIndex rr_frame_;
    Scalar target_front_;
    Scalar target_rear_;
    mutable pinocchio::Data data_;
  };

  using TrackWidthResidual = TrackWidthResidualTpl<double>;

} // namespace simple_mpc
