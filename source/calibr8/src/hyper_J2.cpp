#include <Eigen/Dense>
#include "control.hpp"
#include "defines.hpp"
#include "fad.hpp"
#include "global_residual.hpp"
#include "hyper_J2.hpp"
#include "material_params.hpp"

namespace calibr8 {

using minitensor::det;
using minitensor::dev;
using minitensor::inverse;
using minitensor::trace;
using minitensor::transpose;

static ParameterList get_valid_local_residual_params() {
  ParameterList p;
  p.set<std::string>("type", "hyper_J2");
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
  p.set<double>("K", 0.);
  p.set<double>("Y", 0.);
  return p;
}

template <typename T>
HyperJ2<T>::HyperJ2(ParameterList const& inputs, int ndims) {

  this->m_params_list = inputs;
  this->m_params_list.validateParameters(get_valid_local_residual_params(), 0);

  int const num_residuals = 3;

  this->m_num_residuals = num_residuals;
  this->m_num_eqs.resize(num_residuals);
  this->m_var_types.resize(num_residuals);
  this->m_resid_names.resize(num_residuals);

  this->m_resid_names[0] = "zeta";
  this->m_var_types[0] = SYM_TENSOR;
  this->m_num_eqs[0] = get_num_eqs(SYM_TENSOR, ndims);

  this->m_resid_names[1] = "Ie";
  this->m_var_types[1] = SCALAR;
  this->m_num_eqs[1] = get_num_eqs(SCALAR, ndims);

  this->m_resid_names[2] = "alpha";
  this->m_var_types[2] = SCALAR;
  this->m_num_eqs[2] = get_num_eqs(SCALAR, ndims);

  m_max_iters = inputs.get<int>("nonlinear max iters");
  m_abs_tol = inputs.get<double>("nonlinear absolute tol");
  m_rel_tol = inputs.get<double>("nonlinear relative tol");

}

template <typename T>
HyperJ2<T>::~HyperJ2() {
}

template <typename T>
void HyperJ2<T>::init_params() {

  int const num_params = 4;
  this->m_params.resize(num_params);
  this->m_param_names.resize(num_params);

  this->m_param_names.resize(num_params);
  this->m_param_names[0] = "E";
  this->m_param_names[1] = "nu";
  this->m_param_names[2] = "K";
  this->m_param_names[3] = "Y";

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
    this->m_param_values[es][2] = material_params.get<double>("K");
    this->m_param_values[es][3] = material_params.get<double>("Y");
  }

  this->m_active_indices.resize(1);
  this->m_active_indices[0].resize(1);
  this->m_active_indices[0][0] = 0;
}

template <typename T>
void HyperJ2<T>::init_variables_impl() {

  int const ndims = this->m_num_dims;
  int const zeta_idx = 0;
  int const Ie_idx = 1;
  int const alpha_idx = 2;

  T const Ie = 1.0;
  T const alpha = 0.0;
  Tensor<T> const zeta = minitensor::zero<T>(ndims);

  this->set_scalar_xi(Ie_idx, Ie);
  this->set_scalar_xi(alpha_idx, alpha);
  this->set_sym_tensor_xi(zeta_idx, zeta);

}

template <typename T>
Tensor<T> eval_be_bar(
    RCP<GlobalResidual<T>> global,
    Tensor<T> const& zeta,
    T const& Ie) {
  int const ndims = global->num_dims();
  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const grad_u = global->grad_vector_x(0);
  Tensor<T> const grad_u_prev = global->grad_vector_x_prev(0);
  Tensor<T> const F = grad_u + I;
  Tensor<T> const F_prev = grad_u_prev + I;
  Tensor<T> const rF = F * inverse(F_prev);
  T const det_rF = det(rF);
  T const det_rF_13 = cbrt(det_rF);
  Tensor<T> const rF_bar = rF / det_rF_13;
  Tensor<T> const rF_barT = transpose(rF_bar);
  Tensor<T> const be_bar = rF_bar * (zeta + Ie * I) * rF_barT;
  return be_bar;
}

template <>
int HyperJ2<double>::solve_nonlinear(RCP<GlobalResidual<double>>) {
  return 0;
}

template <>
int HyperJ2<FADT>::solve_nonlinear(RCP<GlobalResidual<FADT>> global) {

  int path;

  // pick an initial guess for the local variables
  {
    Tensor<FADT> const zeta_old = this->sym_tensor_xi_prev(0);
    FADT const Ie_old = this->scalar_xi_prev(1);
    FADT const alpha_old = this->scalar_xi_prev(2);
    Tensor<FADT> const be_bar_trial = eval_be_bar(global, zeta_old, Ie_old);
    Tensor<FADT> const zeta = dev(be_bar_trial);
    FADT const Ie = trace(be_bar_trial) / 3.;
    FADT const alpha = alpha_old;
    this->set_sym_tensor_xi(0, zeta);
    this->set_scalar_xi(1, Ie);
    this->set_scalar_xi(2, alpha);
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
    this->add_to_scalar_xi(2, dxi);

    iter++;

  }

  if ((iter > m_max_iters) && (!converged)) {
    std::cout << "HyperJ2:solve_nonlinear failed in "  << iter << " iterations\n";
    return -1;
  }

  return path;

}

template <>
int HyperJ2<DFADT>::solve_nonlinear(RCP<GlobalResidual<DFADT>>) {
  return 0;
}

template <typename T>
int HyperJ2<T>::evaluate(
    RCP<GlobalResidual<T>> global,
    bool force_path,
    int path_in) {

  int path = ELASTIC;
  int const ndims = this->m_num_dims;
  double const sqrt_23 = std::sqrt(2./3.);
  double const sqrt_32 = std::sqrt(3./2.);

  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const K = this->m_params[2];
  T const Y = this->m_params[3];
  T const mu = compute_mu(E, nu);

  Tensor<T> const zeta_old = this->sym_tensor_xi_prev(0);
  T const Ie_old = this->scalar_xi_prev(1);
  T const alpha_old = this->scalar_xi_prev(2);

  Tensor<T> const zeta = this->sym_tensor_xi(0);
  T const Ie = this->scalar_xi(1);
  T const alpha = this->scalar_xi(2);

  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const be_bar_trial = eval_be_bar(global, zeta_old, Ie_old);
  Tensor<T> const s = mu * zeta;
  T const s_mag = minitensor::norm(s);
  T const sigma_yield = Y + K * alpha;
  T const f = (s_mag - sqrt_23 * sigma_yield) / val(mu);

  Tensor<T> R_zeta;
  T R_Ie;
  T R_alpha;

  if (!force_path) {
    // plastic step
    if (f > m_abs_tol || std::abs(f) < m_abs_tol) {
      Tensor<T> const n = s / s_mag;
      T const dgam = sqrt_32 * (alpha - alpha_old);
      R_zeta = zeta - dev(be_bar_trial) + 2. * dgam * Ie * n;
      R_zeta(2, 2) = trace(zeta);
      R_Ie = det(zeta + Ie * I) - 1.;
      R_alpha = f;
      path = PLASTIC;
    }
    // elastic step
    else {
      R_zeta = zeta - dev(be_bar_trial);
      R_Ie = Ie - trace(be_bar_trial) / 3.;
      R_alpha = alpha - alpha_old;
      path = ELASTIC;
    }
  }

  // force the path
  else {
    path = path_in;
    // plastic step
    if (path == PLASTIC) {
      Tensor<T> const n = s / s_mag;
      T const dgam = sqrt_32 * (alpha - alpha_old);
      R_zeta = zeta - dev(be_bar_trial) + 2. * dgam * Ie * n;
      R_zeta(2, 2) = trace(zeta);
      R_Ie = det(zeta + Ie * I) - 1.;
      R_alpha = f;
    }
    // elastic step
    else {
      R_zeta = zeta - dev(be_bar_trial);
      R_Ie = Ie - trace(be_bar_trial) / 3.;
      R_alpha = alpha - alpha_old;
    }
  }

  this->set_sym_tensor_R(0, R_zeta);
  this->set_scalar_R(1, R_Ie);
  this->set_scalar_R(2, R_alpha);

  return path;

}
template <typename T>
Tensor<T> HyperJ2<T>::cauchy(RCP<GlobalResidual<T>> global) {
  int const pressure_idx = 1;
  T const p = global->scalar_x(pressure_idx);
  int const ndims = global->num_dims();
  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const dev_sigma = this->dev_cauchy(global);
  Tensor<T> const sigma = dev_sigma - p * I;
  return sigma;
}

template <typename T>
Tensor<T> HyperJ2<T>::dev_cauchy(RCP<GlobalResidual<T>> global) {
  int const ndims = global->num_dims();
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const mu = compute_mu(E, nu);
  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const grad_u = global->grad_vector_x(0);
  Tensor<T> const F = grad_u + I;
  Tensor<T> const zeta = this->sym_tensor_xi(0);
  T const J = det(F);
  return mu * zeta / J;
}

template <typename T>
T HyperJ2<T>::hydro_cauchy(RCP<GlobalResidual<T>> global) {
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const kappa = compute_kappa(E, nu);
  int const ndims = global->num_dims();
  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const grad_u = global->grad_vector_x(0);
  Tensor<T> const F = grad_u + I;
  T const J = det(F);
  T const hydro_cauchy = kappa / 2. * (J - 1. / J);
  return hydro_cauchy;
}

template <typename T>
T HyperJ2<T>::pressure_scale_factor() {
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const kappa = compute_kappa(E, nu);
  return kappa;
}

template class HyperJ2<double>;
template class HyperJ2<FADT>;
template class HyperJ2<DFADT>;

}
