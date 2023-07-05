#include <Eigen/Dense>
#include "control.hpp"
#include "defines.hpp"
#include "fad.hpp"
#include "global_residual.hpp"
#include "hosford.hpp"
#include "material_params.hpp"

namespace calibr8 {

using minitensor::eig_spd_cos;
using minitensor::eye;
using minitensor::col;
using minitensor::dyad;
using minitensor::trace;
using minitensor::transpose;
using minitensor::zero;

static ParameterList get_valid_local_residual_params() {
  ParameterList p;
  p.set<std::string>("type", "hosford");
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
  p.set<double>("a", 0.);
  p.set<double>("K", 0.);
  p.set<double>("S", 0.);
  p.set<double>("D", 0.);
  return p;
}

template <typename T>
Hosford<T>::Hosford(ParameterList const& inputs, int ndims) {

  this->m_params_list = inputs;
  this->m_params_list.validateParameters(get_valid_local_residual_params(), 0);

  int const num_residuals = 2;
  int const num_params = 7;

  this->m_num_residuals = num_residuals;
  this->m_num_eqs.resize(num_residuals);
  this->m_var_types.resize(num_residuals);
  this->m_resid_names.resize(num_residuals);

  this->m_resid_names[0] = "pstrain";
  this->m_var_types[0] = SYM_TENSOR;
  this->m_num_eqs[0] = get_num_eqs(SYM_TENSOR, ndims);

  this->m_resid_names[1] = "alpha";
  this->m_var_types[1] = SCALAR;
  this->m_num_eqs[1] = get_num_eqs(SCALAR, ndims);

  m_max_iters = inputs.get<int>("nonlinear max iters");
  m_abs_tol = inputs.get<double>("nonlinear absolute tol");
  m_rel_tol = inputs.get<double>("nonlinear relative tol");

}

template <typename T>
Hosford<T>::~Hosford() {
}

template <typename T>
void Hosford<T>::init_params() {

  int const num_params = 7;
  this->m_params.resize(num_params);
  this->m_param_names.resize(num_params);

  this->m_param_names[0] = "E";
  this->m_param_names[1] = "nu";
  this->m_param_names[2] = "Y";
  this->m_param_names[3] = "Y";
  this->m_param_names[4] = "K";
  this->m_param_names[5] = "S";
  this->m_param_names[6] = "D";

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
    this->m_param_values[es][3] = material_params.get<double>("a");
    this->m_param_values[es][4] = material_params.get<double>("K");
    this->m_param_values[es][5] = material_params.get<double>("S");
    this->m_param_values[es][6] = material_params.get<double>("D");
  }

  this->m_active_indices.resize(1);
  this->m_active_indices[0].resize(1);
  this->m_active_indices[0][0] = 0;
}

template <typename T>
void Hosford<T>::init_variables_impl() {

  int const ndims = this->m_num_dims;
  int const pstrain_idx = 0;
  int const alpha_idx = 1;

  T const alpha = 0.0;
  Tensor<T> const pstrain = zero<T>(ndims);

  this->set_scalar_xi(alpha_idx, alpha);
  this->set_sym_tensor_xi(pstrain_idx, pstrain);

}

template <>
int Hosford<double>::solve_nonlinear(RCP<GlobalResidual<double>>) {
  return 0;
}

template <>
int Hosford<FADT>::solve_nonlinear(RCP<GlobalResidual<FADT>> global) {

  int path;

  // pick an initial guess for the local variables
  {
    Tensor<FADT> const pstrain_old = this->sym_tensor_xi_prev(0);
    Tensor<FADT> const pstrain = pstrain_old;
    FADT const alpha_old = this->scalar_xi_prev(1);
    FADT const alpha = alpha_old;
    this->set_sym_tensor_xi(0, pstrain);
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
    fail("Hosford:solve_nonlinear failed in %d iterations", m_max_iters);
  }

  return path;

}

template <typename T>
void Hosford<T>::evaluate_phi_and_normal(
    RCP<GlobalResidual<T>> global,
    T const& a,
    T& phi,
    Tensor<T>& n) {

  // compute the von-mises stress
  double const vm_val = std::sqrt(3. / 2.) * val(norm(this->dev_cauchy(global)));

  std::pair<Tensor<T>, Tensor<T>> const eigen_decomp = eig_spd_cos(this->cauchy(global));

  Tensor<T> const cauchy_eigvecs = eigen_decomp.first;
  Tensor<T> const cauchy_eigvals = eigen_decomp.second;

  Vector<T> const eigvec_0 = col(cauchy_eigvecs, 0);
  Vector<T> const eigvec_1 = col(cauchy_eigvecs, 1);
  Vector<T> const eigvec_2 = col(cauchy_eigvecs, 2);

  T const vm_scaled_eigval_0 = cauchy_eigvals(0, 0) / vm_val;
  T const vm_scaled_eigval_1 = cauchy_eigvals(1, 1) / vm_val;
  T const vm_scaled_eigval_2 = cauchy_eigvals(2, 2) / vm_val;

  phi = vm_val * std::pow(0.5 * (std::pow(std::abs(vm_scaled_eigval_0 - vm_scaled_eigval_1), a)
      + std::pow(std::abs(vm_scaled_eigval_1 - vm_scaled_eigval_2), a)
      + std::pow(std::abs(vm_scaled_eigval_2 - vm_scaled_eigval_0), a)), 1. / a);
  double const phi_val = val(phi);

  T const phi_scaled_eigval_0 = cauchy_eigvals(0, 0) / phi_val;
  T const phi_scaled_eigval_1 = cauchy_eigvals(1, 1) / phi_val;
  T const phi_scaled_eigval_2 = cauchy_eigvals(2, 2) / phi_val;

  T const sig_diff_0_1 = phi_scaled_eigval_0 - phi_scaled_eigval_1;
  T const sig_diff_1_2 = phi_scaled_eigval_1 - phi_scaled_eigval_2;
  T const sig_diff_2_0 = phi_scaled_eigval_2 - phi_scaled_eigval_0;

  T const sig_factor_0_1 = sig_diff_0_1 * std::pow(std::abs(sig_diff_0_1), a - 2.);
  T const sig_factor_1_2 = sig_diff_1_2 * std::pow(std::abs(sig_diff_1_2), a - 2.);
  T const sig_factor_2_0 = sig_diff_2_0 * std::pow(std::abs(sig_diff_2_0), a - 2.);

  n = 0.5 * ((sig_factor_0_1 - sig_factor_2_0) * dyad(eigvec_0, eigvec_0)
  + (sig_factor_1_2 - sig_factor_0_1) * dyad(eigvec_1, eigvec_1)
  + (sig_factor_2_0 - sig_factor_1_2) * dyad(eigvec_2, eigvec_2));
}

template <typename T>
int Hosford<T>::evaluate(
    RCP<GlobalResidual<T>> global,
    bool force_path,
    int path_in) {

  int path = ELASTIC;

  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const Y = this->m_params[2];
  T const a = this->m_params[3];
  T const K = this->m_params[4];
  T const S = this->m_params[5];
  T const D = this->m_params[6];
  T const mu = compute_mu(E, nu);

  Tensor<T> const pstrain_old = this->sym_tensor_xi_prev(0);
  T const alpha_old = this->scalar_xi_prev(1);

  Tensor<T> const pstrain = this->sym_tensor_xi(0);
  T const alpha = this->scalar_xi(1);

  T phi = 0.;
  Tensor<T> n = zero<T>(3);
  evaluate_phi_and_normal(global, a, phi, n);

  T const flow_stress = Y + K * alpha + S * (1. - std::exp(-D * alpha));
  T const f = (phi - flow_stress) / (2. * val(mu));

  Tensor<T> R_pstrain;
  T R_alpha;

  if (!force_path) {
    // plastic step
    if (f > m_abs_tol || std::abs(f) < m_abs_tol) {
      T const dgam = alpha - alpha_old;
      R_pstrain = pstrain - pstrain_old - dgam * n;
      R_alpha = f;
      path = PLASTIC;
    }
    // elastic step
    else {
      R_pstrain = pstrain - pstrain_old;
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
      R_pstrain = pstrain - pstrain_old - dgam * n;
      R_alpha = f;
    }
    // elastic step
    else {
      R_pstrain = pstrain - pstrain_old;
      R_alpha = alpha - alpha_old;
    }
  }

  this->set_sym_tensor_R(0, R_pstrain);
  this->set_scalar_R(1, R_alpha);

  return path;

}

template <typename T>
Tensor<T> Hosford<T>::cauchy(RCP<GlobalResidual<T>> global) {
  int const pressure_idx = 1;
  T const p = global->scalar_x(pressure_idx);
  int const ndims = global->num_dims();
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  Tensor<T> const I = eye<T>(ndims);
  Tensor<T> const dev_sigma = this->dev_cauchy(global);
  Tensor<T> const sigma = dev_sigma - p * I;
  return sigma;
}

template <typename T>
Tensor<T> Hosford<T>::dev_cauchy(RCP<GlobalResidual<T>> global) {
  int const ndims = global->num_dims();
  Tensor<T> const I = eye<T>(ndims);
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const mu = compute_mu(E, nu);
  Tensor<T> const pstrain = this->sym_tensor_xi(0);
  Tensor<T> const grad_u = global->grad_vector_x(0);
  Tensor<T> const eps = 0.5 * (grad_u + transpose(grad_u));
  Tensor<T> const dev_eps = eps - (trace(eps) / 3.) * I;
  return 2. * mu * (dev_eps - pstrain);
}

template <typename T>
T Hosford<T>::hydro_cauchy(RCP<GlobalResidual<T>> global) {
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const kappa = compute_kappa(E, nu);
  Tensor<T> const grad_u = global->grad_vector_x(0);
  Tensor<T> const eps = 0.5 * (grad_u + transpose(grad_u));
  return kappa * trace(eps);
}

template <typename T>
T Hosford<T>::pressure_scale_factor() {
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const kappa = compute_kappa(E, nu);
  return kappa;
}

template class Hosford<double>;
template class Hosford<FADT>;

}
