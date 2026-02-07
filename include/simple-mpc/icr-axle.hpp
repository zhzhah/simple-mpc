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
  struct ICRAxleResidualTpl : aligator::StageFunctionTpl<_Scalar>
  {
    using Scalar = _Scalar;
    ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
    using Base = aligator::StageFunctionTpl<Scalar>;
    using Data = aligator::StageFunctionDataTpl<Scalar>;

    ICRAxleResidualTpl(
      const int ndx,
      const int nu,
      const pinocchio::Model & model,
      const pinocchio::FrameIndex fl,
      const pinocchio::FrameIndex rl,
      const pinocchio::FrameIndex fr,
      const pinocchio::FrameIndex rr,
      const Eigen::Vector4d * icr_params)
    : Base(ndx, nu, 2)
    , model_(model)
    , fl_(fl)
    , rl_(rl)
    , fr_(fr)
    , rr_(rr)
    , icr_params_(icr_params)
    , data_(model)
    {
    }

    void evaluate(const ConstVectorRef & x, const ConstVectorRef &, Data & data) const override
    {
      data.value_.setZero();
      if (icr_params_ == nullptr || (*icr_params_)(3) < 0.5)
        return;

      const int nq = model_.nq;
      const int nv = model_.nv;
      if (x.size() < nq + nv) return;

      const Eigen::VectorXd q = x.head(nq);
      const Eigen::VectorXd v = x.tail(nv);
      pinocchio::forwardKinematics(model_, data_, q, v);
      pinocchio::updateFramePlacements(model_, data_);

      const Eigen::Vector2d icr((*icr_params_)(0), (*icr_params_)(1));
      const Eigen::Vector2d fl = data_.oMf[fl_].translation().head<2>();
      const Eigen::Vector2d rl = data_.oMf[rl_].translation().head<2>();
      const Eigen::Vector2d fr = data_.oMf[fr_].translation().head<2>();
      const Eigen::Vector2d rr = data_.oMf[rr_].translation().head<2>();

      data.value_[0] = pointLineDistance2D(icr, fl, rl);
      data.value_[1] = pointLineDistance2D(icr, fr, rr);
    }

    void computeJacobians(const ConstVectorRef & x, const ConstVectorRef &, Data & data) const override
    {
      data.Jx_.setZero();
      data.Ju_.setZero();
      const int nq = model_.nq;
      const int nv = model_.nv;
      if (x.size() < nq + nv) return;

      constexpr double eps = 1e-6;
      const Eigen::Vector2d res0 = data.value_;
      for (int k = 0; k < nv; ++k)
      {
        Eigen::VectorXd dq = Eigen::VectorXd::Zero(nv);
        dq[k] = eps;
        const Eigen::VectorXd q = x.head(nq);
        const Eigen::VectorXd v = x.tail(nv);
        const Eigen::VectorXd q_plus = pinocchio::integrate(model_, q, dq);

        pinocchio::forwardKinematics(model_, data_, q_plus, v);
        pinocchio::updateFramePlacements(model_, data_);

        Eigen::Vector2d res_plus = Eigen::Vector2d::Zero();
        if (icr_params_ != nullptr && (*icr_params_)(3) >= 0.5)
        {
          const Eigen::Vector2d icr((*icr_params_)(0), (*icr_params_)(1));
          const Eigen::Vector2d fl = data_.oMf[fl_].translation().head<2>();
          const Eigen::Vector2d rl = data_.oMf[rl_].translation().head<2>();
          const Eigen::Vector2d fr = data_.oMf[fr_].translation().head<2>();
          const Eigen::Vector2d rr = data_.oMf[rr_].translation().head<2>();
          res_plus[0] = pointLineDistance2D(icr, fl, rl);
          res_plus[1] = pointLineDistance2D(icr, fr, rr);
        }
        data.Jx_.col(k) = (res_plus - res0) / eps;
      }
    }

    static double pointLineDistance2D(const Eigen::Vector2d & p, const Eigen::Vector2d & a, const Eigen::Vector2d & b)
    {
      const Eigen::Vector2d ab = b - a;
      const double denom = ab.norm();
      if (denom < 1e-9)
        return 0.0;
      const double num = std::abs((p.x() - a.x()) * ab.y() - (p.y() - a.y()) * ab.x());
      return num / denom;
    }

    const pinocchio::Model & model_;
    pinocchio::FrameIndex fl_;
    pinocchio::FrameIndex rl_;
    pinocchio::FrameIndex fr_;
    pinocchio::FrameIndex rr_;
    const Eigen::Vector4d * icr_params_;
    mutable pinocchio::Data data_;
  };

  using ICRAxleResidual = ICRAxleResidualTpl<double>;

} // namespace simple_mpc

