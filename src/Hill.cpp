#include <Eigen/Dense>
#include "control.hpp"
#include "defines.hpp"
#include "fad.hpp"
#include "global_residual.hpp"
#include "Hill.hpp"
#include "material_params.hpp"

namespace calibr8 {

using minitensor::dev;
using minitensor::inverse;
using minitensor::trace;
using minitensor::transpose;

static ParameterList get_valid_local_residual_params() {
  ParameterList p;
  p.set<std::string>("type", "Hill");
  p.set<int>("nonlinear max iters", 0);
  p.set<double>("nonlinear absolute tol", 0.);
  p.set<double>("nonlinear relative tol", 0.);
  p.sublist("materials");
  return p;
}

static ParameterList get_valid_material_params() {
  ParameterList p;
  p.set<double>("E", 0.);
  p.set<double>("nu", 0.);
  p.set<double>("Y", 0.);
  p.set<double>("R00", 0.);
  p.set<double>("R11", 0.);
  p.set<double>("R22", 0.);
  p.set<double>("R01", 0.);
  p.set<double>("R02", 0.);
  p.set<double>("R12", 0.);
  p.set<double>("S", 0.);
  p.set<double>("D", 0.);
  return p;
}

template <typename T>
Hill<T>::Hill(ParameterList const& inputs, int ndims) {

  this->m_params_list = inputs;
  this->m_params_list.validateParameters(get_valid_local_residual_params(), 0);

  int const num_residuals = 2;

  this->m_num_residuals = num_residuals;
  this->m_num_eqs.resize(num_residuals);
  this->m_var_types.resize(num_residuals);
  this->m_resid_names.resize(num_residuals);

  // unrotated Cauchy stress
  this->m_resid_names[0] = "TC";
  this->m_var_types[0] = SYM_TENSOR;
  this->m_num_eqs[0] = get_num_eqs(SYM_TENSOR, ndims);

  // isotropic hardening variable
  this->m_resid_names[1] = "alpha";
  this->m_var_types[1] = SCALAR;
  this->m_num_eqs[1] = get_num_eqs(SCALAR, ndims);

  m_max_iters = inputs.get<int>("nonlinear max iters");
  m_abs_tol = inputs.get<double>("nonlinear absolute tol");
  m_rel_tol = inputs.get<double>("nonlinear relative tol");

}

template <typename T>
Hill<T>::~Hill() {
}

template <typename T>
void Hill<T>::init_params() {

  // 2 elastic + Y + 6 Hill + S + D
  int const num_params = 11;
  this->m_params.resize(num_params);
  this->m_param_names.resize(num_params);

  this->m_param_names.resize(num_params);
  this->m_param_names[0] = "E";
  this->m_param_names[1] = "nu";
  this->m_param_names[2] = "Y";
  this->m_param_names[3] = "R00";
  this->m_param_names[4] = "R11";
  this->m_param_names[5] = "R22";
  this->m_param_names[6] = "R01";
  this->m_param_names[7] = "R02";
  this->m_param_names[8] = "R12";
  this->m_param_names[9] = "S";
  this->m_param_names[10] = "D";

  int const num_elem_sets = this->m_elem_set_names.size();
  resize(this->m_param_values, num_elem_sets, num_params);

  ParameterList& all_material_params =
      this->m_params_list.sublist("materials", true);

  for (int es = 0; es < num_elem_sets; ++es) {
    std::string const& elem_set_name = this->m_elem_set_names[es];
    ParameterList& material_params =
        all_material_params.sublist(elem_set_name, true);
    material_params.validateParameters(get_valid_material_params(), 0);
    this->m_param_values[es][0] = material_params.get<double>("E");
    this->m_param_values[es][1] = material_params.get<double>("nu");
    this->m_param_values[es][2] = material_params.get<double>("Y");
    this->m_param_values[es][3] = material_params.get<double>("R00");
    this->m_param_values[es][4] = material_params.get<double>("R11");
    this->m_param_values[es][5] = material_params.get<double>("R22");
    this->m_param_values[es][6] = material_params.get<double>("R01");
    this->m_param_values[es][7] = material_params.get<double>("R02");
    this->m_param_values[es][8] = material_params.get<double>("R12");
    this->m_param_values[es][9] = material_params.get<double>("S");
    this->m_param_values[es][10] = material_params.get<double>("D");
  }

  this->m_active_indices.resize(1);
  this->m_active_indices[0].resize(1);
  this->m_active_indices[0][0] = 0;
}

template <typename T>
void Hill<T>::init_variables_impl() {

  int const ndims = this->m_num_dims;
  int const TC_idx = 0;
  int const alpha_idx = 1;

  Tensor<T> const TC = minitensor::zero<T>(ndims);
  T const alpha = 0.0;

  this->set_sym_tensor_xi(TC_idx, TC);
  this->set_scalar_xi(alpha_idx, alpha);

}

template <typename T>
Tensor<T> eval_d(RCP<GlobalResidual<T>> global) {
  int const ndims = global->num_dims();
  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const grad_u = global->grad_vector_x(0);
  Tensor<T> const grad_u_prev = global->grad_vector_x_prev(0);
  Tensor<T> const F = grad_u + I;
  Tensor<T> const F_prev = grad_u_prev + I;
  Tensor<T> const Finv = inverse(F);
  Tensor<T> const R = minitensor::polar_rotation(F);
  Tensor<T> const L = (F - F_prev) * Finv;
  Tensor<T> const D = 0.5 * (L + transpose(L));
  Tensor<T> const d = transpose(R) * D * R;
  return d;
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

template <>
int Hill<double>::solve_nonlinear(RCP<GlobalResidual<double>>) {
  return 0;
}

template <>
int Hill<FADT>::solve_nonlinear(RCP<GlobalResidual<FADT>> global) {

  int path;

  // pick an initial guess for the local variables
  {
    double const E = val(this->m_params[0]);
    double const nu = val(this->m_params[1]);
    double const lambda = compute_lambda(E, nu);
    double const mu = compute_mu(E, nu);
    int const ndims = this->m_num_dims;
    Tensor<FADT> const I = minitensor::eye<FADT>(ndims);
    Tensor<FADT> const TC_old = this->sym_tensor_xi_prev(0);
    FADT const alpha_old = this->scalar_xi_prev(1);
    Tensor<FADT> const d = eval_d(global);
    Tensor<FADT> const TC = TC_old + lambda * trace(d) * I + 2. * mu * d;
    FADT const alpha = alpha_old;
    this->set_sym_tensor_xi(0, TC);
    this->set_scalar_xi(1, alpha);
    path = ELASTIC;
  }

  // newton iteration until convergence

  int iter = 1;
  double R_norm_0 = 1.;
  bool converged = false;

  while ((iter <= m_max_iters) && (!converged)) {

    path = this->evaluate(global);

    double const R_norm = this->norm_residual();
    if (iter == 1) R_norm_0 = R_norm;
    double const R_norm_rel = R_norm / R_norm_0;
    if ((R_norm_rel < m_rel_tol) || (R_norm < m_abs_tol)) {
      converged = true;
      break;
    }

    EMatrix const J = this->eigen_jacobian(this->m_num_dofs);
    EVector const R = this->eigen_residual();
    EVector const dxi = J.fullPivLu().solve(-R);

    this->add_to_sym_tensor_xi(0, dxi);
    this->add_to_scalar_xi(1, dxi);

    iter++;

  }

  // fail if convergence was not achieved
  if ((iter > m_max_iters) && (!converged)) {
    fail("Hill:solve_nonlinear failed in %d iterations", m_max_iters);
  }

  return path;

}

template <typename T>
int Hill<T>::evaluate(
    RCP<GlobalResidual<T>> global,
    bool force_path,
    int path_in) {

  int path = ELASTIC;
  int const ndims = this->m_num_dims;

  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const Y = this->m_params[2];
  T const R00 = this->m_params[3];
  T const R11 = this->m_params[4];
  T const R22 = this->m_params[5];
  T const R01 = this->m_params[6];
  T const R02 = this->m_params[7];
  T const R12 = this->m_params[8];
  T const S = this->m_params[9];
  T const D = this->m_params[10];
  T const lambda = compute_lambda(E, nu);
  T const mu = compute_mu(E, nu);

  Vector<T> const hill_params = compute_hill_params(R00, R11, R22,
      R01, R02, R12);

  Tensor<T> const TC_old = this->sym_tensor_xi_prev(0);
  T const alpha_old = this->scalar_xi_prev(1);

  Tensor<T> const TC = this->sym_tensor_xi(0);
  T const alpha = this->scalar_xi(1);

  T const hill = compute_hill_value(TC, hill_params);
  T const sigma_yield = Y + S * (1. - std::exp(-D * alpha));
  T const f = hill - sigma_yield;

  Tensor<T> R_TC;
  T R_alpha;

  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const d = eval_d(global);
  R_TC = TC - TC_old - lambda * trace(d) * I - 2. * mu * d;

  if (!force_path) {
    // plastic step
    if (f > m_abs_tol || std::abs(f) < m_abs_tol) {
      T const dgam = alpha - alpha_old;
      Tensor<T> n = compute_hill_normal(TC, hill_params, hill);
      R_TC += 2. * mu * dgam * n;
      R_alpha = f;
      path = PLASTIC;
    }
    // elastic step
    else {
      R_alpha = alpha - alpha_old;
      path = ELASTIC;
    }
  }

  // force the path
  else {
    path = path_in;
    // plastic step
    if (path == PLASTIC) {
      T const dgam = alpha - alpha_old;
      Tensor<T> n = compute_hill_normal(TC, hill_params, hill);
      R_TC += 2. * mu * dgam * n;
      R_alpha = f;
    }
    // elastic step
    else {
      R_alpha = alpha - alpha_old;
    }
  }

  this->set_sym_tensor_R(0, R_TC);
  this->set_scalar_R(1, R_alpha);

  return path;

}

// returns rotated Cauchy instead of dev
template <typename T>
Tensor<T> Hill<T>::dev_cauchy(RCP<GlobalResidual<T>> global) {
  int const ndims = this->m_num_dims;
  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const grad_u = global->grad_vector_x(0);
  Tensor<T> const F = grad_u + I;
  Tensor<T> const TC = this->sym_tensor_xi(0);
  Tensor<T> const R = minitensor::polar_rotation(F);
  Tensor<T> const RC = R * TC * transpose(R);
  return RC;
}

template <typename T>
Tensor<T> Hill<T>::cauchy(RCP<GlobalResidual<T>> global, T p) {
  int const ndims = this->m_num_dims;
  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const RC = this->dev_cauchy(global);
  Tensor<T> const sigma = dev(RC) - p * I;
  return sigma;
}

template class Hill<double>;
template class Hill<FADT>;

}
