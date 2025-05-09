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

template <typename T>
Vector<T> collect_barlat_params(
    T const& p_01,
    T const& p_02,
    T const& p_10,
    T const& p_12,
    T const& p_20,
    T const& p_21,
    T const& p_33,
    T const& p_44,
    T const& p_55)
{
  Vector<T> barlat_params = minitensor::Vector<T>(9);

  barlat_params(0) = p_01;
  barlat_params(1) = p_02;
  barlat_params(2) = p_10;
  barlat_params(3) = p_12;
  barlat_params(4) = p_20;
  barlat_params(5) = p_21;
  barlat_params(6) = p_33;
  barlat_params(7) = p_44;
  barlat_params(8) = p_55;

  return barlat_params;
}

template <typename T>
Vector<T> flatten_stress(Tensor<T> const& stress) {

  Vector<T> flat_stress = minitensor::Vector<T>(6);

  flat_stress(0) = stress(0, 0);
  flat_stress(1) = stress(1, 1);
  flat_stress(2) = stress(2, 2);
  flat_stress(3) = stress(0, 1);
  flat_stress(4) = stress(1, 2);
  flat_stress(5) = stress(2, 0);

  return flat_stress;
}

template <typename T>
Tensor<T> unflatten_stress(Vector<T> const& flat_stress) {

  Tensor<T> stress = minitensor::zero<T>(3);

  stress(0, 0) = flat_stress(0);
  stress(0, 1) = flat_stress(3);
  stress(0, 2) = flat_stress(5);
  stress(1, 0) = flat_stress(3);
  stress(1, 1) = flat_stress(1);
  stress(1, 2) = flat_stress(4);
  stress(2, 0) = flat_stress(5);
  stress(2, 1) = flat_stress(4);
  stress(2, 2) = flat_stress(2);

  return stress;
}

template <typename T>
Tensor<T> unflatten_barlat_params(
    Vector<T> const& flat_barlat_params)
{
  Tensor<T> L = minitensor::zero<T>(6);

  T p_01 = flat_barlat_params(0);
  T p_02 = flat_barlat_params(1);
  T p_10 = flat_barlat_params(2);
  T p_12 = flat_barlat_params(3);
  T p_20 = flat_barlat_params(4);
  T p_21 = flat_barlat_params(5);
  T p_33 = flat_barlat_params(6);
  T p_44 = flat_barlat_params(7);
  T p_55 = flat_barlat_params(8);

  L(0, 0) = (p_01 + p_02) / 3.;
  L(0, 1) = (-2. * p_01 + p_02) / 3.;
  L(0, 2) = (p_01 - 2. * p_02) / 3.;

  L(1, 0) = (-2. * p_10 + p_12) / 3.;
  L(1, 1) = (p_10 + p_12) / 3.;
  L(1, 2) = (p_10 - 2. * p_12) / 3.;

  L(2, 0) = (-2. * p_20 + p_21) / 3.;
  L(2, 1) = (p_20 - 2. * p_21) / 3.;
  L(2, 2) = (p_20 + p_21) / 3.;

  L(3, 3) = p_33;
  L(4, 4) = p_44;
  L(5, 5) = p_55;

  return L;
}

template <typename T>
void compute_barlat_eigen_decomp(
    Tensor<T> const& cauchy,
    Vector<T> const& flat_barlat_params,
    Tensor<T>& eigvecs,
    Tensor<T>& eigvals)
{
  // take cauchy from Cartesian 3x3 to to Voigt 6x1
  Vector<T> const flat_cauchy = flatten_stress(cauchy);
  // L is a 6x6 matrix in Voigt notation
  Tensor<T> const L = unflatten_barlat_params(flat_barlat_params);
  // flat_s is a 6x1 vector in Voigt notation
  Vector<T> const flat_s = L * flat_cauchy;
  // take flat_s from Voigt 6x1 to standard Cartesian 3x3
  Tensor<T> const s = unflatten_stress(flat_s);

  std::pair<Tensor<T>, Tensor<T>> const s_eigen_decomp = eig_spd_cos(s);
  eigvecs = s_eigen_decomp.first;
  eigvals = s_eigen_decomp.second;
}

template <typename T>
Tensor<T> compute_dyad_from_eigvec(
    Tensor<T> const& eigvecs,
    int index)
{
  Vector<T> const eigvec = minitensor::col(eigvecs, index);
  return minitensor::dyad(eigvec, eigvec);
}

template <typename T>
T compute_barlat_sp_normal_multiplier(
    Tensor<T> const& sp_eigvals,
    Tensor<T> const& dp_eigvals,
    T const& a,
    int index)
{
  T const diff_0 = sp_eigvals(index, index) - dp_eigvals(0, 0);
  T const diff_1 = sp_eigvals(index, index) - dp_eigvals(1, 1);
  T const diff_2 = sp_eigvals(index, index) - dp_eigvals(2, 2);

  T const factor_0 = diff_0 * std::pow(std::abs(diff_0), a - 2);
  T const factor_1 = diff_1 * std::pow(std::abs(diff_1), a - 2);
  T const factor_2 = diff_2 * std::pow(std::abs(diff_2), a - 2);

  return 0.25 * (factor_0 + factor_1 + factor_2);
}

template <typename T>
Tensor<T> compute_barlat_sp_normal_component(
    Tensor<T> const& eigvecs,
    Tensor<T> const& sp_eigvals,
    Tensor<T> const& dp_eigvals,
    T const& a,
    int index)
{
  Tensor<T> const sp_normal_component =
      compute_barlat_sp_normal_multiplier(sp_eigvals, dp_eigvals, a, index)
      * compute_dyad_from_eigvec(eigvecs, index);
  return sp_normal_component;
}

template <typename T>
T compute_barlat_dp_normal_multiplier(
    Tensor<T> const& sp_eigvals,
    Tensor<T> const& dp_eigvals,
    T const& a,
    int index)
{
  T const diff_0 = sp_eigvals(0, 0) - dp_eigvals(index, index);
  T const diff_1 = sp_eigvals(1, 1) - dp_eigvals(index, index);
  T const diff_2 = sp_eigvals(2, 2) - dp_eigvals(index, index);

  T const factor_0 = -diff_0 * std::pow(std::abs(diff_0), a - 2);
  T const factor_1 = -diff_1 * std::pow(std::abs(diff_1), a - 2);
  T const factor_2 = -diff_2 * std::pow(std::abs(diff_2), a - 2);

  return 0.25 * (factor_0 + factor_1 + factor_2);
}

template <typename T>
Tensor<T> compute_barlat_dp_normal_component(
    Tensor<T> const& eigvecs,
    Tensor<T> const& sp_eigvals,
    Tensor<T> const& dp_eigvals,
    T const& a,
    int index)
{
  Tensor<T> const dp_normal_component =
      compute_barlat_dp_normal_multiplier(sp_eigvals, dp_eigvals, a, index)
      * compute_dyad_from_eigvec(eigvecs, index);
  return dp_normal_component;
}

template <typename T>
Tensor<T> compute_barlat_normal(
    Tensor<T> const& sp_eigvecs,
    Tensor<T> const& dp_eigvecs,
    Tensor<T> const& sp_eigvals,
    Tensor<T> const& dp_eigvals,
    Vector<T> const& flat_sp_barlat_params,
    Vector<T> const& flat_dp_barlat_params,
    T const& a)
{
  Tensor<T> const L_sp = unflatten_barlat_params(flat_sp_barlat_params);
  Tensor<T> const dphi_d_sp = unflatten_stress(L_sp * flatten_stress(
      compute_barlat_sp_normal_component(sp_eigvecs, sp_eigvals, dp_eigvals, a, 0)
      + compute_barlat_sp_normal_component(sp_eigvecs, sp_eigvals, dp_eigvals, a, 1)
      + compute_barlat_sp_normal_component(sp_eigvecs, sp_eigvals, dp_eigvals, a, 2)
  ));

  Tensor<T> const L_dp = unflatten_barlat_params(flat_dp_barlat_params);
  Tensor<T> const dphi_d_dp = unflatten_stress(L_dp * flatten_stress(
      compute_barlat_dp_normal_component(dp_eigvecs, sp_eigvals, dp_eigvals, a, 0)
      + compute_barlat_dp_normal_component(dp_eigvecs, sp_eigvals, dp_eigvals, a, 1)
      + compute_barlat_dp_normal_component(dp_eigvecs, sp_eigvals, dp_eigvals, a, 2)
  ));

  return dphi_d_sp + dphi_d_dp;
}

template <typename T>
void evaluate_barlat_phi_and_normal(
    Tensor<T> const& cauchy,
    Vector<T> const& flat_sp_barlat_params,
    Vector<T> const& flat_dp_barlat_params,
    T const& a,
    T& phi,
    Tensor<T>& normal)
{
  double const sqrt_32 = std::sqrt(3. / 2.);
  Tensor<T> const dev_cauchy = minitensor::dev(cauchy);
  T const norm_dev_cauchy = minitensor::norm(dev_cauchy);
  T const vm_phi = sqrt_32 * norm_dev_cauchy;

  Tensor<T> sp_eigvecs(3);
  Tensor<T> sp_eigvals(3);
  compute_barlat_eigen_decomp(cauchy, flat_sp_barlat_params,
      sp_eigvecs, sp_eigvals);

  Tensor<T> dp_eigvecs(3);
  Tensor<T> dp_eigvals(3);
  compute_barlat_eigen_decomp(cauchy, flat_dp_barlat_params,
      dp_eigvecs, dp_eigvals);

  // vms -> von-mises scaled
  Tensor<T> const vms_sp_eigvals = sp_eigvals / vm_phi;
  Tensor<T> const vms_dp_eigvals = dp_eigvals / vm_phi;

  phi = vm_phi * std::pow(0.25
      * (std::pow(std::abs(vms_sp_eigvals(0, 0) - vms_dp_eigvals(0, 0)), a)
      + std::pow(std::abs(vms_sp_eigvals(0, 0) - vms_dp_eigvals(1, 1)), a)
      + std::pow(std::abs(vms_sp_eigvals(0, 0) - vms_dp_eigvals(2, 2)), a)
      + std::pow(std::abs(vms_sp_eigvals(1, 1) - vms_dp_eigvals(0, 0)), a)
      + std::pow(std::abs(vms_sp_eigvals(1, 1) - vms_dp_eigvals(1, 1)), a)
      + std::pow(std::abs(vms_sp_eigvals(1, 1) - vms_dp_eigvals(2, 2)), a)
      + std::pow(std::abs(vms_sp_eigvals(2, 2) - vms_dp_eigvals(0, 0)), a)
      + std::pow(std::abs(vms_sp_eigvals(2, 2) - vms_dp_eigvals(1, 1)), a)
      + std::pow(std::abs(vms_sp_eigvals(2, 2) - vms_dp_eigvals(2, 2)), a)
      ) , 1. / a);

  // bs -> barlat scaled
  Tensor<T> const bs_sp_eigvals = sp_eigvals / phi;
  Tensor<T> const bs_dp_eigvals = dp_eigvals / phi;

  normal = compute_barlat_normal(
      sp_eigvecs, dp_eigvecs,
      bs_sp_eigvals, bs_dp_eigvals,
      flat_sp_barlat_params, flat_dp_barlat_params, a
  );
}

}
