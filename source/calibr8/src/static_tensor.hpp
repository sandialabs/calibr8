#pragma once

//! \file static_tensor.hpp
//! \brief Convert between dynamic and fixed-size minitensor tensors

#include "defines.hpp"

namespace calibr8 {

template <minitensor::Index N, typename T>
minitensor::Tensor<T, N> to_static(Tensor<T> const& a) {
  minitensor::Tensor<T, N> result;
  for (minitensor::Index i = 0; i < N; ++i)
    for (minitensor::Index j = 0; j < N; ++j)
      result(i, j) = a(i, j);
  return result;
}

template <typename T, minitensor::Index N>
Tensor<T> to_dynamic(minitensor::Tensor<T, N> const& a) {
  Tensor<T> result(N);
  for (minitensor::Index i = 0; i < N; ++i)
    for (minitensor::Index j = 0; j < N; ++j)
      result(i, j) = a(i, j);
  return result;
}

}
