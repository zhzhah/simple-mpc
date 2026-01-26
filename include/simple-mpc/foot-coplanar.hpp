///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include "aligator/core/function-abstract.hpp"
#include <array>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>

namespace simple_mpc
{
  template <typename _Scalar>
  struct FootCoplanarResidualTpl : aligator::StageFunctionTpl<_Scalar>
  {
    using Scalar = _Scalar;
    ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
    using Base = aligator::StageFunctionTpl<Scalar>;
    using Data = aligator::StageFunctionDataTpl<Scalar>;

    FootCoplanarResidualTpl(
      const int ndx,
      const int nu,
      const pinocchio::Model & model,
      const std::array<pinocchio::FrameIndex, 4> & frame_ids)
    : Base(ndx, nu, 1)
    , model_(model)
    , data_(model)
    , frame_ids_(frame_ids)
    , nq_(static_cast<int>(model.nq))
    , nv_(static_cast<int>(model.nv))
    {
    }

    void evaluate(const ConstVectorRef & x, const ConstVectorRef &, Data & data) const override
    {
      data.value_.setZero();
      if (x.size() < nq_ + nv_) return;
      const auto q = x.head(nq_);
      const auto v = x.segment(nq_, nv_);
      pinocchio::forwardKinematics(model_, data_, q, v);
      pinocchio::updateFramePlacements(model_, data_);

      const auto & p0 = data_.oMf[frame_ids_[0]].translation();
      const auto & p1 = data_.oMf[frame_ids_[1]].translation();
      const auto & p2 = data_.oMf[frame_ids_[2]].translation();
      const auto & p3 = data_.oMf[frame_ids_[3]].translation();

      const Eigen::Matrix<Scalar,3,1> a = p1 - p0;
      const Eigen::Matrix<Scalar,3,1> b = p2 - p0;
      const Eigen::Matrix<Scalar,3,1> c = p3 - p0;
      const Eigen::Matrix<Scalar,3,1> n = b.cross(c);
      data.value_[0] = a.dot(n);
    }

    void computeJacobians(const ConstVectorRef & x, const ConstVectorRef &, Data & data) const override
    {
      data.Jx_.setZero();
      data.Ju_.setZero();
      if (x.size() < nq_ + nv_) return;
      const auto q = x.head(nq_);
      const auto v = x.segment(nq_, nv_);
      pinocchio::forwardKinematics(model_, data_, q, v);
      pinocchio::updateFramePlacements(model_, data_);

      const auto & p0 = data_.oMf[frame_ids_[0]].translation();
      const auto & p1 = data_.oMf[frame_ids_[1]].translation();
      const auto & p2 = data_.oMf[frame_ids_[2]].translation();
      const auto & p3 = data_.oMf[frame_ids_[3]].translation();

      const Eigen::Matrix<Scalar,3,1> a = p1 - p0;
      const Eigen::Matrix<Scalar,3,1> b = p2 - p0;
      const Eigen::Matrix<Scalar,3,1> c = p3 - p0;
      const Eigen::Matrix<Scalar,3,1> n = b.cross(c);

      const Eigen::Matrix<Scalar,3,1> dr_dp1 = n;
      const Eigen::Matrix<Scalar,3,1> dr_dp2 = a.cross(c);
      const Eigen::Matrix<Scalar,3,1> dr_dp3 = b.cross(a);
      const Eigen::Matrix<Scalar,3,1> dr_dp0 = -dr_dp1 - dr_dp2 - dr_dp3;

      Eigen::Matrix<Scalar, 6, Eigen::Dynamic> J0(6, nv_);
      Eigen::Matrix<Scalar, 6, Eigen::Dynamic> J1(6, nv_);
      Eigen::Matrix<Scalar, 6, Eigen::Dynamic> J2(6, nv_);
      Eigen::Matrix<Scalar, 6, Eigen::Dynamic> J3(6, nv_);
      pinocchio::computeFrameJacobian(model_, data_, q, frame_ids_[0], pinocchio::LOCAL_WORLD_ALIGNED, J0);
      pinocchio::computeFrameJacobian(model_, data_, q, frame_ids_[1], pinocchio::LOCAL_WORLD_ALIGNED, J1);
      pinocchio::computeFrameJacobian(model_, data_, q, frame_ids_[2], pinocchio::LOCAL_WORLD_ALIGNED, J2);
      pinocchio::computeFrameJacobian(model_, data_, q, frame_ids_[3], pinocchio::LOCAL_WORLD_ALIGNED, J3);

      const auto J0p = J0.block(0, 0, 3, nv_);
      const auto J1p = J1.block(0, 0, 3, nv_);
      const auto J2p = J2.block(0, 0, 3, nv_);
      const auto J3p = J3.block(0, 0, 3, nv_);

      data.Jx_.block(0, 0, 1, nv_) =
        dr_dp0.transpose() * J0p +
        dr_dp1.transpose() * J1p +
        dr_dp2.transpose() * J2p +
        dr_dp3.transpose() * J3p;
    }

    const pinocchio::Model & model_;
    mutable pinocchio::Data data_;
    std::array<pinocchio::FrameIndex, 4> frame_ids_;
    int nq_ = 0;
    int nv_ = 0;
  };

  using FootCoplanarResidual = FootCoplanarResidualTpl<double>;
} // namespace simple_mpc
