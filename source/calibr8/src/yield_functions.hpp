#pragma once

//! \file yield_functions.hpp
//! \brief Helper methods for yield functions

namespace calibr8 {

template <typename T>
Tensor<T> insert_2D_tensor_into_3D(Tensor<T> const& t_2D) {

  Tensor<T> t_3D = minitensor::zero<T>(3);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      t_3D(i, j) = t_2D(i, j);
    }
  }

  return t_3D;
}

template <typename T>
Tensor<T> extract_2D_tensor_from_3D(Tensor<T> const& t_3D) {

  Tensor<T> t_2D = minitensor::zero<T>(2);
  for (int i = 0; i < 2 ; ++i) {
    for (int j = 0; j < 2; ++j) {
      t_2D(i, j) = t_3D(i, j);
    }
  }

  return t_2D;
}

template <typename T>
Vector<T> compute_hill_params(T const& R00, T const& R11, T const& R22,
    T const& R01, T const& R02, T const& R12) {

  Vector<T> hill_params = minitensor::Vector<T>(6);
  hill_params[0] = 0.5 * (std::pow(R11, -2) + std::pow(R22, -2)
     - std::pow(R00, -2));
  hill_params[1] = 0.5 * (std::pow(R22, -2) + std::pow(R00, -2)
     - std::pow(R11, -2));
  hill_params[2] = 0.5 * (std::pow(R00, -2) + std::pow(R11, -2)
     - std::pow(R22, -2));
  hill_params[3] = 1.5 * std::pow(R12, -2);
  hill_params[4] = 1.5 * std::pow(R02, -2);
  hill_params[5] = 1.5 * std::pow(R01, -2);

  return hill_params;
}

template <typename T>
T compute_hill_value(Tensor<T> const& TC,
    Vector<T> const& hill_params) {

  T const F = hill_params[0];
  T const G = hill_params[1];
  T const H = hill_params[2];
  T const L = hill_params[3];
  T const M = hill_params[4];
  T const N = hill_params[5];

  T const hill = std::sqrt(F * pow(TC(1, 1) - TC(2, 2), 2)
      + G * pow(TC(2, 2) - TC(0, 0), 2)
      + H * pow(TC(0, 0) - TC(1, 1), 2)
      + 2. * (L * pow(TC(1, 2), 2)
      + M * pow(TC(0, 2), 2)
      + N * pow(TC(0, 1), 2)));

  return hill;
}

template <typename T>
Tensor<T> compute_hill_normal(Tensor<T> const& TC,
    Vector<T> const& hill_params,
    T const& hill_value) {

  T const F = hill_params[0];
  T const G = hill_params[1];
  T const H = hill_params[2];
  T const L = hill_params[3];
  T const M = hill_params[4];
  T const N = hill_params[5];

  Tensor<T> n = minitensor::zero<T>(3);
  n(0, 0) = (G + H) * TC(0, 0) - H * TC(1, 1) - G * TC(2, 2);
  n(1, 1) = (F + H) * TC(1, 1) - H * TC(0, 0) - F * TC(2, 2);
  n(2, 2) = (G + F) * TC(2, 2) - G * TC(0, 0) - F * TC(1, 1);
  n(0, 1) = N * TC(0, 1);
  n(0, 2) = M * TC(0, 2);
  n(1, 2) = L * TC(1, 2);
  n(1, 0) = n(0, 1);
  n(2, 0) = n(0, 2);
  n(2, 1) = n(1, 2);

  n /= hill_value;

  return n;
}

}
