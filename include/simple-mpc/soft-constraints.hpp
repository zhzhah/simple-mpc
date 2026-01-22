///////////////////////////////////////////////////////////////////////////////
// BSD 2-Clause License
//
// Copyright (C) 2024, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include "aligator/core/function-abstract.hpp"

namespace simple_mpc
{
  template <typename _Scalar>
  struct SoftFrictionConeResidualTpl : aligator::StageFunctionTpl<_Scalar>
  {
    using Scalar = _Scalar;
    ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
    using Base = aligator::StageFunctionTpl<Scalar>;
    using Data = aligator::StageFunctionDataTpl<Scalar>;

    SoftFrictionConeResidualTpl(const int ndx, const int nu, const int contact_index, const int force_size, Scalar mu)
    : Base(ndx, nu, 2)
    , contact_index_(contact_index)
    , force_size_(force_size)
    , mu_(mu)
    {
    }

    void evaluate(const ConstVectorRef &, const ConstVectorRef & u, Data & data) const override
    {
      data.value_.setZero();
      if (force_size_ < 3) return;
      const int start = contact_index_ * force_size_;
      if (start + 2 >= u.size()) return;

      const Scalar fx = u[start + 0];
      const Scalar fy = u[start + 1];
      const Scalar fz = u[start + 2];

      const Scalar r1 = std::sqrt(fx * fx + fy * fy) - mu_ * fz;
      const Scalar r2 = -fz;

      data.value_[0] = std::max<Scalar>(Scalar(0), r1);
      data.value_[1] = std::max<Scalar>(Scalar(0), r2);
    }

    void computeJacobians(const ConstVectorRef &, const ConstVectorRef & u, Data & data) const override
    {
      data.Jx_.setZero();
      data.Ju_.setZero();
      if (force_size_ < 3) return;
      const int start = contact_index_ * force_size_;
      if (start + 2 >= u.size()) return;

      const Scalar fx = u[start + 0];
      const Scalar fy = u[start + 1];
      const Scalar fz = u[start + 2];
      const Scalar norm_xy = std::sqrt(fx * fx + fy * fy);

      const Scalar r1 = norm_xy - mu_ * fz;
      const Scalar r2 = -fz;

      if (r1 > Scalar(0))
      {
        if (norm_xy > Scalar(1e-12))
        {
          data.Ju_(0, start + 0) = fx / norm_xy;
          data.Ju_(0, start + 1) = fy / norm_xy;
        }
        data.Ju_(0, start + 2) = -mu_;
      }

      if (r2 > Scalar(0))
      {
        data.Ju_(1, start + 2) = -Scalar(1);
      }
    }

    int contact_index_ = 0;
    int force_size_ = 3;
    Scalar mu_ = Scalar(0);
  };

  using SoftFrictionConeResidual = SoftFrictionConeResidualTpl<double>;

} // namespace simple_mpc
