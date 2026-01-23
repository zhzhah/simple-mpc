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
  struct WheelDistanceResidualTpl : aligator::StageFunctionTpl<_Scalar>
  {
    using Scalar = _Scalar;
    ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
    using Base = aligator::StageFunctionTpl<Scalar>;
    using Data = aligator::StageFunctionDataTpl<Scalar>;

    WheelDistanceResidualTpl(
      const int ndx,
      const int nu,
      const pinocchio::Model & model,
      const std::vector<std::pair<pinocchio::FrameIndex, pinocchio::FrameIndex>> & frame_pairs,
      const Scalar min_distance)
    : Base(ndx, nu, static_cast<int>(frame_pairs.size()))
    , model_(model)
    , frame_pairs_(frame_pairs)
    , min_distance_(min_distance)
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

      const double min_sq = min_distance_ * min_distance_;
      int k = 0;
      for (const auto & pair : frame_pairs_)
      {
        Eigen::Vector3d pi = data_.oMf[pair.first].translation();
        Eigen::Vector3d pj = data_.oMf[pair.second].translation();
        pi.z() = 0.0;
        pj.z() = 0.0;
        const double dist_sq = (pi - pj).squaredNorm();
        data.value_[k++] = min_sq - dist_sq;
      }
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
      Eigen::VectorXd res0(data.nr);
      {
        pinocchio::forwardKinematics(model_, data_, q, v);
        pinocchio::updateFramePlacements(model_, data_);
        const double min_sq = min_distance_ * min_distance_;
        int k = 0;
        for (const auto & pair : frame_pairs_)
        {
          Eigen::Vector3d pi = data_.oMf[pair.first].translation();
          Eigen::Vector3d pj = data_.oMf[pair.second].translation();
          pi.z() = 0.0;
          pj.z() = 0.0;
          const double dist_sq = (pi - pj).squaredNorm();
          res0[k++] = min_sq - dist_sq;
        }
      }

      for (int k = 0; k < nv; ++k)
      {
        Eigen::VectorXd dq = Eigen::VectorXd::Zero(nv);
        dq[k] = eps;
        const Eigen::VectorXd q_plus = pinocchio::integrate(model_, q, dq);

        pinocchio::forwardKinematics(model_, data_, q_plus, v);
        pinocchio::updateFramePlacements(model_, data_);
        const double min_sq = min_distance_ * min_distance_;
        Eigen::VectorXd res_plus(data.nr);
        int idx = 0;
        for (const auto & pair : frame_pairs_)
        {
          Eigen::Vector3d pi = data_.oMf[pair.first].translation();
          Eigen::Vector3d pj = data_.oMf[pair.second].translation();
          pi.z() = 0.0;
          pj.z() = 0.0;
          const double dist_sq = (pi - pj).squaredNorm();
          res_plus[idx++] = min_sq - dist_sq;
        }
        data.Jx_.col(k) = (res_plus - res0) / eps;
      }
    }

    const pinocchio::Model & model_;
    std::vector<std::pair<pinocchio::FrameIndex, pinocchio::FrameIndex>> frame_pairs_;
    Scalar min_distance_;
    mutable pinocchio::Data data_;
  };

  using WheelDistanceResidual = WheelDistanceResidualTpl<double>;

  template <typename _Scalar>
  struct WheelDistanceSoftResidualTpl : aligator::StageFunctionTpl<_Scalar>
  {
    using Scalar = _Scalar;
    ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
    using Base = aligator::StageFunctionTpl<Scalar>;
    using Data = aligator::StageFunctionDataTpl<Scalar>;

    WheelDistanceSoftResidualTpl(
      const int ndx,
      const int nu,
      const pinocchio::Model & model,
      const std::vector<std::pair<pinocchio::FrameIndex, pinocchio::FrameIndex>> & frame_pairs,
      const std::vector<Scalar> * ref_distances,
      const Scalar min_distance,
      const Scalar eps)
    : Base(ndx, nu, static_cast<int>(frame_pairs.size()))
    , model_(model)
    , frame_pairs_(frame_pairs)
    , ref_distances_(ref_distances)
    , min_distance_(min_distance)
    , eps_(eps)
    , data_(model)
    {
    }

    void evaluate(const ConstVectorRef & x, const ConstVectorRef &, Data & data) const override
    {
      data.value_.setZero();
      const int nq = model_.nq;
      const int nv = model_.nv;
      if (x.size() < nq + nv) return;
      if (ref_distances_ == nullptr) return;

      const Eigen::VectorXd q = x.head(nq);
      const Eigen::VectorXd v = x.tail(nv);
      pinocchio::forwardKinematics(model_, data_, q, v);
      pinocchio::updateFramePlacements(model_, data_);

      int k = 0;
      for (const auto & pair : frame_pairs_)
      {
        Eigen::Vector3d pi = data_.oMf[pair.first].translation();
        Eigen::Vector3d pj = data_.oMf[pair.second].translation();
        pi.z() = 0.0;
        pj.z() = 0.0;
        const double d = (pi - pj).norm();
        const double denom = std::max(d - min_distance_ + eps_, eps_);
        const double denom0 = std::max((*ref_distances_)[k] - min_distance_ + eps_, eps_);
        data.value_[k] = (1.0 / denom) - (1.0 / denom0);
        ++k;
      }
    }

    void computeJacobians(const ConstVectorRef & x, const ConstVectorRef &, Data & data) const override
    {
      data.Jx_.setZero();
      data.Ju_.setZero();
      const int nq = model_.nq;
      const int nv = model_.nv;
      if (x.size() < nq + nv) return;
      if (ref_distances_ == nullptr) return;

      const Eigen::VectorXd q = x.head(nq);
      const Eigen::VectorXd v = x.tail(nv);

      constexpr double eps = 1e-6;
      Eigen::VectorXd res0(data.nr);
      {
        pinocchio::forwardKinematics(model_, data_, q, v);
        pinocchio::updateFramePlacements(model_, data_);
        int k = 0;
        for (const auto & pair : frame_pairs_)
        {
          Eigen::Vector3d pi = data_.oMf[pair.first].translation();
          Eigen::Vector3d pj = data_.oMf[pair.second].translation();
          pi.z() = 0.0;
          pj.z() = 0.0;
          const double d = (pi - pj).norm();
          const double denom = std::max(d - min_distance_ + eps_, eps_);
          const double denom0 = std::max((*ref_distances_)[k] - min_distance_ + eps_, eps_);
          res0[k] = (1.0 / denom) - (1.0 / denom0);
          ++k;
        }
      }

      for (int k = 0; k < nv; ++k)
      {
        Eigen::VectorXd dq = Eigen::VectorXd::Zero(nv);
        dq[k] = eps;
        const Eigen::VectorXd q_plus = pinocchio::integrate(model_, q, dq);

        pinocchio::forwardKinematics(model_, data_, q_plus, v);
        pinocchio::updateFramePlacements(model_, data_);
        Eigen::VectorXd res_plus(data.nr);
        int idx = 0;
        for (const auto & pair : frame_pairs_)
        {
          Eigen::Vector3d pi = data_.oMf[pair.first].translation();
          Eigen::Vector3d pj = data_.oMf[pair.second].translation();
          pi.z() = 0.0;
          pj.z() = 0.0;
          const double d = (pi - pj).norm();
          const double denom = std::max(d - min_distance_ + eps_, eps_);
          const double denom0 = std::max((*ref_distances_)[idx] - min_distance_ + eps_, eps_);
          res_plus[idx] = (1.0 / denom) - (1.0 / denom0);
          ++idx;
        }
        data.Jx_.col(k) = (res_plus - res0) / eps;
      }
    }

    const pinocchio::Model & model_;
    std::vector<std::pair<pinocchio::FrameIndex, pinocchio::FrameIndex>> frame_pairs_;
    const std::vector<Scalar> * ref_distances_;
    Scalar min_distance_;
    Scalar eps_;
    mutable pinocchio::Data data_;
  };

  using WheelDistanceSoftResidual = WheelDistanceSoftResidualTpl<double>;

} // namespace simple_mpc
