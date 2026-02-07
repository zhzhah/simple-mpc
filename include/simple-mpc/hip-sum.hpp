///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include "aligator/core/function-abstract.hpp"
#include <pinocchio/multibody/model.hpp>

namespace simple_mpc
{
  template <typename _Scalar>
  struct HipSumResidualTpl : aligator::StageFunctionTpl<_Scalar>
  {
    using Scalar = _Scalar;
    ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
    using Base = aligator::StageFunctionTpl<Scalar>;
    using Data = aligator::StageFunctionDataTpl<Scalar>;

    HipSumResidualTpl(
      const int ndx,
      const int nu,
      const pinocchio::Model & model,
      const std::vector<std::string> & joint_names)
    : Base(ndx, nu, 1)
    , model_(model)
    {
      for (const auto & name : joint_names)
      {
        if (!model_.existJointName(name))
          continue;
        const pinocchio::JointIndex jid = model_.getJointId(name);
        const int idx_q = model_.joints[jid].idx_q();
        const int nq = model_.joints[jid].nq();
        joint_q_slices_.emplace_back(idx_q, nq);
      }
    }

    void evaluate(const ConstVectorRef & x, const ConstVectorRef &, Data & data) const override
    {
      data.value_.setZero();
      const int nq = model_.nq;
      const int nv = model_.nv;
      if (x.size() < nq + nv) return;
      const Eigen::VectorXd q = x.head(nq);
      double sum = 0.0;
      for (const auto & s : joint_q_slices_)
      {
        sum += q.segment(s.first, s.second).sum();
      }
      data.value_[0] = sum;
    }

    void computeJacobians(const ConstVectorRef & x, const ConstVectorRef &, Data & data) const override
    {
      data.Jx_.setZero();
      data.Ju_.setZero();
      const int nq = model_.nq;
      const int nv = model_.nv;
      if (x.size() < nq + nv) return;

      constexpr double eps = 1e-6;
      const double res0 = data.value_[0];
      for (int k = 0; k < nv; ++k)
      {
        Eigen::VectorXd dq = Eigen::VectorXd::Zero(nv);
        dq[k] = eps;
        const Eigen::VectorXd q = x.head(nq);
        const Eigen::VectorXd q_plus = pinocchio::integrate(model_, q, dq);
        double sum = 0.0;
        for (const auto & s : joint_q_slices_)
        {
          sum += q_plus.segment(s.first, s.second).sum();
        }
        const double res_plus = sum;
        data.Jx_(0, k) = (res_plus - res0) / eps;
      }
    }

    const pinocchio::Model & model_;
    std::vector<std::pair<int, int>> joint_q_slices_;
  };

  using HipSumResidual = HipSumResidualTpl<double>;

} // namespace simple_mpc
