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
  struct FootSumInBaseResidualTpl : aligator::StageFunctionTpl<_Scalar>
  {
    using Scalar = _Scalar;
    ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
    using Base = aligator::StageFunctionTpl<Scalar>;
    using Data = aligator::StageFunctionDataTpl<Scalar>;

    FootSumInBaseResidualTpl(
      const int ndx,
      const int nu,
      const pinocchio::Model & model,
      const pinocchio::FrameIndex base_frame,
      const std::vector<pinocchio::FrameIndex> & foot_frames,
      const Eigen::Vector3d & target_sum)
    : Base(ndx, nu, 3)
    , model_(model)
    , base_frame_(base_frame)
    , foot_frames_(foot_frames)
    , target_sum_(target_sum)
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
      const double yaw = std::atan2(Rb(1, 0), Rb(0, 0));
      const double cy = std::cos(yaw);
      const double sy = std::sin(yaw);
      const Eigen::Matrix3d R_yaw =
        (Eigen::Matrix3d() << cy, -sy, 0.0, sy, cy, 0.0, 0.0, 0.0, 1.0).finished();
      const Eigen::Vector3d pb = oMb.translation();

      Eigen::Vector3d sum = Eigen::Vector3d::Zero();
      for (const auto & fid : foot_frames_)
      {
        const Eigen::Vector3d pw = data_.oMf[fid].translation();
        sum += R_yaw.transpose() * (pw - pb);
      }
      const double nfeet = static_cast<double>(foot_frames_.size());
      Eigen::Vector3d mean = Eigen::Vector3d::Zero();
      if (nfeet > 0.0)
        mean = sum / nfeet;
      data.value_ = -mean - target_sum_;
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

      constexpr double eps = 1e-6;
      Eigen::Vector3d res0 = Eigen::Vector3d::Zero();
      {
        pinocchio::forwardKinematics(model_, data_, q, v);
        pinocchio::updateFramePlacements(model_, data_);
        const pinocchio::SE3 & oMb = data_.oMf[base_frame_];
        const Eigen::Matrix3d Rb = oMb.rotation();
        const double yaw = std::atan2(Rb(1, 0), Rb(0, 0));
        const double cy = std::cos(yaw);
        const double sy = std::sin(yaw);
        const Eigen::Matrix3d R_yaw =
          (Eigen::Matrix3d() << cy, -sy, 0.0, sy, cy, 0.0, 0.0, 0.0, 1.0).finished();
        const Eigen::Vector3d pb = oMb.translation();
        for (const auto & fid : foot_frames_)
        {
          const Eigen::Vector3d pw = data_.oMf[fid].translation();
          res0 += R_yaw.transpose() * (pw - pb);
        }
        const double nfeet = static_cast<double>(foot_frames_.size());
        if (nfeet > 0.0)
          res0 = -res0 / nfeet - target_sum_;
        else
          res0 = -target_sum_;
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
        const double yaw = std::atan2(Rb(1, 0), Rb(0, 0));
        const double cy = std::cos(yaw);
        const double sy = std::sin(yaw);
        const Eigen::Matrix3d R_yaw =
          (Eigen::Matrix3d() << cy, -sy, 0.0, sy, cy, 0.0, 0.0, 0.0, 1.0).finished();
        const Eigen::Vector3d pb = oMb.translation();
        Eigen::Vector3d res_plus = Eigen::Vector3d::Zero();
        for (const auto & fid : foot_frames_)
        {
          const Eigen::Vector3d pw = data_.oMf[fid].translation();
          res_plus += R_yaw.transpose() * (pw - pb);
        }
        const double nfeet = static_cast<double>(foot_frames_.size());
        if (nfeet > 0.0)
          res_plus = -res_plus / nfeet - target_sum_;
        else
          res_plus = -target_sum_;
        data.Jx_.col(k) = (res_plus - res0) / eps;
      }
    }

    const pinocchio::Model & model_;
    pinocchio::FrameIndex base_frame_;
    std::vector<pinocchio::FrameIndex> foot_frames_;
    Eigen::Vector3d target_sum_;
    mutable pinocchio::Data data_;
  };

  using FootSumInBaseResidual = FootSumInBaseResidualTpl<double>;

} // namespace simple_mpc
