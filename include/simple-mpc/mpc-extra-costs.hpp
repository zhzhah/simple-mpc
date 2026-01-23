///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include "aligator/core/function-abstract.hpp"
#include <pinocchio/algorithm/joint-configuration.hpp>

namespace simple_mpc
{
  template <typename _Scalar>
  struct ForceZVarianceResidualTpl : aligator::StageFunctionTpl<_Scalar>
  {
    using Scalar = _Scalar;
    ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
    using Base = aligator::StageFunctionTpl<Scalar>;
    using Data = aligator::StageFunctionDataTpl<Scalar>;

    ForceZVarianceResidualTpl(const int ndx, const int nu, const int force_size, const std::vector<int> & active_contacts)
    : Base(ndx, nu, 1)
    , force_size_(force_size)
    , active_contacts_(active_contacts)
    {
    }

    void evaluate(const ConstVectorRef &, const ConstVectorRef & u, Data & data) const override
    {
      data.value_.setZero();
      if (force_size_ < 3) return;
      if (active_contacts_.empty()) return;

      double mean = 0.0;
      for (int idx : active_contacts_)
      {
        const long offset = static_cast<long>(idx) * force_size_;
        mean += u[offset + 2];
      }
      mean /= static_cast<double>(active_contacts_.size());

      double var = 0.0;
      for (int idx : active_contacts_)
      {
        const long offset = static_cast<long>(idx) * force_size_;
        const double dz = u[offset + 2] - mean;
        var += dz * dz;
      }
      var /= static_cast<double>(active_contacts_.size());
      data.value_[0] = var;
    }

    void computeJacobians(const ConstVectorRef &, const ConstVectorRef & u, Data & data) const override
    {
      data.Jx_.setZero();
      data.Ju_.setZero();
      if (force_size_ < 3) return;
      if (active_contacts_.empty()) return;

      double mean = 0.0;
      for (int idx : active_contacts_)
      {
        const long offset = static_cast<long>(idx) * force_size_;
        mean += u[offset + 2];
      }
      mean /= static_cast<double>(active_contacts_.size());

      const double n = static_cast<double>(active_contacts_.size());
      for (int idx : active_contacts_)
      {
        const long offset = static_cast<long>(idx) * force_size_;
        const double dz = u[offset + 2] - mean;
        data.Ju_(0, offset + 2) = 2.0 * dz / n;
      }
    }

    int force_size_;
    std::vector<int> active_contacts_;
  };

  using ForceZVarianceResidual = ForceZVarianceResidualTpl<double>;

  template <typename _Scalar>
  struct JointLimitSoftResidualTpl : aligator::StageFunctionTpl<_Scalar>
  {
    using Scalar = _Scalar;
    ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
    using Base = aligator::StageFunctionTpl<Scalar>;
    using Data = aligator::StageFunctionDataTpl<Scalar>;

    JointLimitSoftResidualTpl(
      const int ndx,
      const int nu,
      const Eigen::VectorXd & qmin,
      const Eigen::VectorXd & qmax,
      const Scalar soft_fraction)
    : Base(ndx, nu, static_cast<int>(qmin.size()))
    , qmin_(qmin)
    , qmax_(qmax)
    , soft_fraction_(soft_fraction)
    {
    }

    void evaluate(const ConstVectorRef & x, const ConstVectorRef &, Data & data) const override
    {
      data.value_.setZero();
      const int nq = static_cast<int>(qmin_.size()) + 7;
      if (x.size() < nq) return;

      const Eigen::VectorXd q = x.head(nq);
      for (int i = 0; i < qmin_.size(); ++i)
      {
        const double q_i = q[7 + i];
        const double qmin = qmin_[i];
        const double qmax = qmax_[i];
        const double range = qmax - qmin;
        if (!(range > 0.0)) continue;

        const double mid = 0.5 * (qmin + qmax);
        const double half = 0.5 * range;
        const double dead = soft_fraction_ * half;
        const double dist = std::abs(q_i - mid);
        if (dist <= dead) continue;

        const double s = (dist - dead) / std::max(half - dead, 1e-9);
        const double s_clamped = std::min(1.0, std::max(0.0, s));
        // Smoothly increasing penalty towards limit (s in [0,1]).
        data.value_[i] = s_clamped * s_clamped;
      }
    }

    void computeJacobians(const ConstVectorRef & x, const ConstVectorRef &, Data & data) const override
    {
      data.Jx_.setZero();
      data.Ju_.setZero();
      const int nq = static_cast<int>(qmin_.size()) + 7;
      if (x.size() < nq) return;

      const Eigen::VectorXd q = x.head(nq);
      for (int i = 0; i < qmin_.size(); ++i)
      {
        const double q_i = q[7 + i];
        const double qmin = qmin_[i];
        const double qmax = qmax_[i];
        const double range = qmax - qmin;
        if (!(range > 0.0)) continue;

        const double mid = 0.5 * (qmin + qmax);
        const double half = 0.5 * range;
        const double dead = soft_fraction_ * half;
        const double dist = std::abs(q_i - mid);
        if (dist <= dead) continue;

        const double denom = std::max(half - dead, 1e-9);
        const double s = (dist - dead) / denom;
        const double s_clamped = std::min(1.0, std::max(0.0, s));
        const double ds_dq = (q_i >= mid ? 1.0 : -1.0) / denom;
        data.Jx_(i, 7 + i) = 2.0 * s_clamped * ds_dq;
      }
    }

    Eigen::VectorXd qmin_;
    Eigen::VectorXd qmax_;
    Scalar soft_fraction_;
  };

  using JointLimitSoftResidual = JointLimitSoftResidualTpl<double>;

} // namespace simple_mpc
