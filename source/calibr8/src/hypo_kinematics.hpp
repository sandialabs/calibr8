#pragma once

//! \file hypo_kinematics.hpp
//! \brief Shared kinematics helpers for hypoelastic local residuals

#include <cmath>
#include "defines.hpp"

namespace calibr8 {

template <typename T>
Tensor<T> compute_unrotated_rate_of_deformation_increment(
    Tensor<T> const& F,
    Tensor<T> const& F_prev) {
  int const ndims = F.get_dimension();
  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const F_mid = 0.5 * (F + F_prev);
  Tensor<T> const R_mid = minitensor::polar_rotation(F_mid);
  Tensor<T> const L = (F - F_prev) * minitensor::inverse(F_mid);
  Tensor<T> const midpoint =
      minitensor::transpose(R_mid) *
      (0.5 * (L + minitensor::transpose(L))) *
      R_mid;
  Tensor<T> const deviatoric =
      midpoint - (minitensor::trace(midpoint) / 3.) * I;
  T const volumetric =
      std::log(minitensor::det(F) / minitensor::det(F_prev));
  return deviatoric + (volumetric / 3.) * I;
}

}
