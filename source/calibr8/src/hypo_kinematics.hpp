#pragma once

//! \file hypo_kinematics.hpp
//! \brief Shared kinematics helpers for hypoelastic local residuals

#include "defines.hpp"

namespace calibr8 {

template <typename T>
Tensor<T> compute_unrotated_rate_of_deformation(
    Tensor<T> const& F, Tensor<T> const& F_prev, Tensor<T> const& R) {
  Tensor<T> const Finv = minitensor::inverse(F);
  Tensor<T> const L = (F - F_prev) * Finv;
  Tensor<T> const D = 0.5 * (L + minitensor::transpose(L));
  return minitensor::transpose(R) * D * R;
}

}
