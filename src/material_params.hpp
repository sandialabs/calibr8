#pragma once

//! \file material_params.hpp
//! \brief Helper methods for material parameters

namespace calibr8 {

//! \brief Compute the shear modulus
//! \param E The elastic modulus
//! \param nu Poisson's ratio
template <typename T>
T compute_mu(T const& E, T const& nu) {
  return E / (2. * (1. + nu));
}

//! \brief Compute the bulk modulus
//! \param E The elastic modulus
//! \param nu Poisson's ratio
template <typename T>
T compute_kappa(T const& E, T const& nu) {
  return E / (3. * (1. - 2. * ( nu)));
}

//! \brief Compute lmabda
//! \param E The elastic modulus
//! \param nu Poisson's ratio
template <typename T>
T compute_lambda(T const& E, T const& nu) {
  return E*nu/((1.+nu)*(1.-2.*nu));
}

}
