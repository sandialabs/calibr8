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
minitensor::Vector<T, 9> collect_barlat_params(
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
  minitensor::Vector<T, 9> barlat_params;

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
minitensor::Vector<T, 6> flatten_stress(minitensor::Tensor<T, 3> const& stress) {

  minitensor::Vector<T, 6> flat_stress;

  flat_stress(0) = stress(0, 0);
  flat_stress(1) = stress(1, 1);
  flat_stress(2) = stress(2, 2);
  flat_stress(3) = stress(0, 1);
  flat_stress(4) = stress(1, 2);
  flat_stress(5) = stress(2, 0);

  return flat_stress;
}

template <typename T>
minitensor::Tensor<T, 3> unflatten_stress(minitensor::Vector<T, 6> const& flat_stress) {

  minitensor::Tensor<T, 3> stress;

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
minitensor::Tensor<T, 6> unflatten_barlat_params(
    minitensor::Vector<T, 9> const& flat_barlat_params)
{
  minitensor::Tensor<T, 6> L(minitensor::Filler::ZEROS);

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
    minitensor::Tensor<T, 3> const& cauchy,
    minitensor::Vector<T, 9> const& flat_barlat_params,
    minitensor::Tensor<T, 3>& eigvecs,
    minitensor::Tensor<T, 3>& eigvals)
{
  // take cauchy from Cartesian 3x3 to to Voigt 6x1
  minitensor::Vector<T, 6> const flat_cauchy = flatten_stress(cauchy);
  // L is a 6x6 matrix in Voigt notation
  minitensor::Tensor<T, 6> const L = unflatten_barlat_params(flat_barlat_params);
  // flat_s is a 6x1 vector in Voigt notation
  minitensor::Vector<T, 6> const flat_s = L * flat_cauchy;
  // take flat_s from Voigt 6x1 to standard Cartesian 3x3
  minitensor::Tensor<T, 3> const s = unflatten_stress(flat_s);

  std::pair<minitensor::Tensor<T, 3>, minitensor::Tensor<T, 3>> const s_eigen_decomp = eig_spd_cos(s);
  eigvecs = s_eigen_decomp.first;
  eigvals = s_eigen_decomp.second;
}

template <typename T>
minitensor::Tensor<T, 3> compute_dyad_from_eigvec(
    minitensor::Tensor<T, 3> const& eigvecs,
    int index)
{
  minitensor::Vector<T, 3> const eigvec = minitensor::col(eigvecs, index);
  return minitensor::dyad(eigvec, eigvec);
}

template <typename T>
T compute_barlat_sp_normal_multiplier(
    minitensor::Tensor<T, 3> const& sp_eigvals,
    minitensor::Tensor<T, 3> const& dp_eigvals,
    T const& a,
    int index)
{
  T const diff_0 = sp_eigvals(index, index) - dp_eigvals(0, 0);
  T const diff_1 = sp_eigvals(index, index) - dp_eigvals(1, 1);
  T const diff_2 = sp_eigvals(index, index) - dp_eigvals(2, 2);

  T const factor_0 = diff_0 * std::pow(std::abs(diff_0), a - 2.);
  T const factor_1 = diff_1 * std::pow(std::abs(diff_1), a - 2.);
  T const factor_2 = diff_2 * std::pow(std::abs(diff_2), a - 2.);

  return 0.25 * (factor_0 + factor_1 + factor_2);
}

template <typename T>
minitensor::Tensor<T, 3> compute_barlat_sp_normal_component(
    minitensor::Tensor<T, 3> const& eigvecs,
    minitensor::Tensor<T, 3> const& sp_eigvals,
    minitensor::Tensor<T, 3> const& dp_eigvals,
    T const& a,
    int index)
{
  minitensor::Tensor<T, 3> const sp_normal_component =
      compute_barlat_sp_normal_multiplier(sp_eigvals, dp_eigvals, a, index)
      * compute_dyad_from_eigvec(eigvecs, index);
  return sp_normal_component;
}

template <typename T>
T compute_barlat_dp_normal_multiplier(
    minitensor::Tensor<T, 3> const& sp_eigvals,
    minitensor::Tensor<T, 3> const& dp_eigvals,
    T const& a,
    int index)
{
  T const diff_0 = sp_eigvals(0, 0) - dp_eigvals(index, index);
  T const diff_1 = sp_eigvals(1, 1) - dp_eigvals(index, index);
  T const diff_2 = sp_eigvals(2, 2) - dp_eigvals(index, index);

  T const factor_0 = -diff_0 * std::pow(std::abs(diff_0), a - 2.);
  T const factor_1 = -diff_1 * std::pow(std::abs(diff_1), a - 2.);
  T const factor_2 = -diff_2 * std::pow(std::abs(diff_2), a - 2.);

  return 0.25 * (factor_0 + factor_1 + factor_2);
}

template <typename T>
minitensor::Tensor<T, 3> compute_barlat_dp_normal_component(
    minitensor::Tensor<T, 3> const& eigvecs,
    minitensor::Tensor<T, 3> const& sp_eigvals,
    minitensor::Tensor<T, 3> const& dp_eigvals,
    T const& a,
    int index)
{
  minitensor::Tensor<T, 3> const dp_normal_component =
      compute_barlat_dp_normal_multiplier(sp_eigvals, dp_eigvals, a, index)
      * compute_dyad_from_eigvec(eigvecs, index);
  return dp_normal_component;
}

template <typename T>
minitensor::Tensor<T, 3> compute_barlat_normal(
    minitensor::Tensor<T, 3> const& sp_eigvecs,
    minitensor::Tensor<T, 3> const& dp_eigvecs,
    minitensor::Tensor<T, 3> const& sp_eigvals,
    minitensor::Tensor<T, 3> const& dp_eigvals,
    minitensor::Vector<T, 9> const& flat_sp_barlat_params,
    minitensor::Vector<T, 9> const& flat_dp_barlat_params,
    T const& a)
{
  minitensor::Tensor<T, 3> sp_normal = minitensor::zero<T, 3>();
  minitensor::Tensor<T, 3> dp_normal = minitensor::zero<T, 3>();
  for (int i = 0; i < 3; ++i) {
    sp_normal += compute_barlat_sp_normal_component(sp_eigvecs, sp_eigvals, dp_eigvals, a, i);
    dp_normal += compute_barlat_dp_normal_component(dp_eigvecs, sp_eigvals, dp_eigvals, a, i);
  }

  minitensor::Tensor<T, 6> const L_sp = unflatten_barlat_params(flat_sp_barlat_params);
  minitensor::Tensor<T, 6> const L_dp = unflatten_barlat_params(flat_dp_barlat_params);

  minitensor::Vector<T, 6> const flat_dphi =
      L_sp * flatten_stress(sp_normal) + L_dp * flatten_stress(dp_normal);

  return unflatten_stress(flat_dphi);
}

template <typename T>
struct BarlatEigenDecomp {
  minitensor::Tensor<T, 3> sp_eigvecs;
  minitensor::Tensor<T, 3> sp_eigvals;
  minitensor::Tensor<T, 3> dp_eigvecs;
  minitensor::Tensor<T, 3> dp_eigvals;
};

template <typename T>
void evaluate_barlat_phi(
    minitensor::Tensor<T, 3> const& cauchy,
    minitensor::Vector<T, 9> const& flat_sp_barlat_params,
    minitensor::Vector<T, 9> const& flat_dp_barlat_params,
    T const& a,
    T& phi,
    BarlatEigenDecomp<T>& decomp)
{
  double const sqrt_32 = std::sqrt(3. / 2.);
  minitensor::Tensor<T, 3> const dev_cauchy = minitensor::dev(cauchy);
  double const norm_dev_cauchy = val(minitensor::norm(dev_cauchy));
  double const vm_phi = sqrt_32 * norm_dev_cauchy;

  compute_barlat_eigen_decomp(cauchy, flat_sp_barlat_params,
      decomp.sp_eigvecs, decomp.sp_eigvals);
  compute_barlat_eigen_decomp(cauchy, flat_dp_barlat_params,
      decomp.dp_eigvecs, decomp.dp_eigvals);

  // vms -> von-mises scaled
  minitensor::Tensor<T, 3> const vms_sp_eigvals = decomp.sp_eigvals / vm_phi;
  minitensor::Tensor<T, 3> const vms_dp_eigvals = decomp.dp_eigvals / vm_phi;

  T const s0 = vms_sp_eigvals(0, 0);
  T const s1 = vms_sp_eigvals(1, 1);
  T const s2 = vms_sp_eigvals(2, 2);

  T const d0 = vms_dp_eigvals(0, 0);
  T const d1 = vms_dp_eigvals(1, 1);
  T const d2 = vms_dp_eigvals(2, 2);

  T const t00 = std::pow(std::abs(s0 - d0), a);
  T const t01 = std::pow(std::abs(s0 - d1), a);
  T const t02 = std::pow(std::abs(s0 - d2), a);
  T const t10 = std::pow(std::abs(s1 - d0), a);
  T const t11 = std::pow(std::abs(s1 - d1), a);
  T const t12 = std::pow(std::abs(s1 - d2), a);
  T const t20 = std::pow(std::abs(s2 - d0), a);
  T const t21 = std::pow(std::abs(s2 - d1), a);
  T const t22 = std::pow(std::abs(s2 - d2), a);

  T const sum = 0.25 * (t00 + t01 + t02 + t10 + t11 + t12 + t20 + t21 + t22);

  phi = vm_phi * std::exp((1.0 / a) * std::log(sum));
}

template <typename T>
minitensor::Tensor<T, 3> evaluate_barlat_normal(
    BarlatEigenDecomp<T> const& decomp,
    T const& phi,
    minitensor::Vector<T, 9> const& flat_sp_barlat_params,
    minitensor::Vector<T, 9> const& flat_dp_barlat_params,
    T const& a)
{
  // bs -> barlat scaled
  minitensor::Tensor<T, 3> const bs_sp_eigvals = decomp.sp_eigvals / phi;
  minitensor::Tensor<T, 3> const bs_dp_eigvals = decomp.dp_eigvals / phi;

  return compute_barlat_normal(
      decomp.sp_eigvecs, decomp.dp_eigvecs,
      bs_sp_eigvals, bs_dp_eigvals,
      flat_sp_barlat_params, flat_dp_barlat_params, a
  );
}

}
