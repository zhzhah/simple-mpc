///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
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
  struct WheelAxleHeightResidualTpl : aligator::StageFunctionTpl<_Scalar>
  {
    using Scalar = _Scalar;
    ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
    using Base = aligator::StageFunctionTpl<Scalar>;
    using Data = aligator::StageFunctionDataTpl<Scalar>;

    WheelAxleHeightResidualTpl(
      const int ndx,
      const int nu,
      const pinocchio::Model & model,
      const std::array<pinocchio::FrameIndex, 4> & frame_ids)
    : Base(ndx, nu, 4)
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

      for (size_t i = 0; i < 4; ++i)
      {
        data.value_[static_cast<int>(i)] = data_.oMf[frame_ids_[i]].translation().z();
      }
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

      for (size_t i = 0; i < 4; ++i)
      {
        Eigen::Matrix<Scalar, 6, Eigen::Dynamic> J(6, nv_);
        pinocchio::computeFrameJacobian(
          model_, data_, q, frame_ids_[i], pinocchio::LOCAL_WORLD_ALIGNED, J);
        const auto Jp = J.block(0, 0, 3, nv_);
        data.Jx_.block(static_cast<int>(i), 0, 1, nv_) = Jp.row(2);
      }
    }

    const pinocchio::Model & model_;
    mutable pinocchio::Data data_;
    std::array<pinocchio::FrameIndex, 4> frame_ids_;
    int nq_ = 0;
    int nv_ = 0;
  };

  using WheelAxleHeightResidual = WheelAxleHeightResidualTpl<double>;
} // namespace simple_mpc
