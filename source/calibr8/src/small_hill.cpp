#include <Eigen/Dense>
#include "control.hpp"
#include "defines.hpp"
#include "fad.hpp"
#include "global_residual.hpp"
#include "material_params.hpp"
#include "small_hill.hpp"
#include "yield_functions.hpp"

namespace calibr8 {

static ParameterList get_valid_local_residual_params() {
  ParameterList p;
  p.set<std::string>("type", "small_hill");
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
SmallHill<T>::SmallHill(ParameterList const& inputs, int ndims) {

  this->m_params_list = inputs;
  this->m_params_list.validateParameters(get_valid_local_residual_params(), 0);

  int const num_residuals = 2;
  int const num_params = 11;

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
SmallHill<T>::~SmallHill() {
}

template <typename T>
void SmallHill<T>::init_params() {

  int const num_params = 11;
  this->m_params.resize(num_params);
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
void SmallHill<T>::init_variables_impl() {

  int const ndims = this->m_num_dims;
  int const pstrain_idx = 0;
  int const alpha_idx = 1;

  T const alpha = 0.0;
  Tensor<T> const pstrain = minitensor::zero<T>(ndims);

  this->set_scalar_xi(alpha_idx, alpha);
  this->set_sym_tensor_xi(pstrain_idx, pstrain);

}

template <>
int SmallHill<double>::solve_nonlinear(RCP<GlobalResidual<double>>) {
  return 0;
}

template <>
int SmallHill<FADT>::solve_nonlinear(RCP<GlobalResidual<FADT>> global) {

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

  if ((iter > m_max_iters) && (!converged)) {
    // std::cout << "SmallHill:solve_nonlinear failed in "  << iter << " iterations\n";
    return -1;
  }

  return path;

}

template <>
int SmallHill<DFADT>::solve_nonlinear(RCP<GlobalResidual<DFADT>>) {
  return 0;
}

template <typename T>
int SmallHill<T>::evaluate(
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
  T const mu = compute_mu(E, nu);

  Vector<T> const hill_params = compute_hill_params(R00, R11, R22,
      R01, R02, R12);

  Tensor<T> const pstrain_old = this->sym_tensor_xi_prev(0);
  T const alpha_old = this->scalar_xi_prev(1);

  Tensor<T> const pstrain = this->sym_tensor_xi(0);
  T const alpha = this->scalar_xi(1);

  Tensor<T> const s = this->dev_cauchy(global);
  T const hill = compute_hill_value(s, hill_params);
  T const sigma_yield = Y + S * (1. - std::exp(-D * alpha));
  T const f = (hill - sigma_yield) / val(mu);

  Tensor<T> R_pstrain;
  T R_alpha;

  if (!force_path) {
    // plastic step
    if (f > m_abs_tol || std::abs(f) < m_abs_tol) {
      Tensor<T> const n = compute_hill_normal(s, hill_params, hill);
      T const dgam = alpha - alpha_old;
      R_pstrain = pstrain - pstrain_old - dgam * n;
      R_pstrain(2, 2) = trace(pstrain);
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
      Tensor<T> const n = compute_hill_normal(s, hill_params, hill);
      T const dgam = alpha - alpha_old;
      R_pstrain = pstrain - pstrain_old - dgam * n;
      R_pstrain(2, 2) = trace(pstrain);
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
Tensor<T> SmallHill<T>::cauchy(RCP<GlobalResidual<T>> global) {
  int const pressure_idx = 1;
  T const p = global->scalar_x(pressure_idx);
  int const ndims = global->num_dims();
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const dev_sigma = this->dev_cauchy(global);
  Tensor<T> const sigma = dev_sigma - p * I;
  return sigma;
}

template <typename T>
Tensor<T> SmallHill<T>::dev_cauchy(RCP<GlobalResidual<T>> global) {
  int const ndims = global->num_dims();
  Tensor<T> const I = minitensor::eye<T>(ndims);
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const mu = compute_mu(E, nu);
  Tensor<T> const pstrain = this->sym_tensor_xi(0);
  Tensor<T> const grad_u = global->grad_vector_x(0);
  Tensor<T> const eps = 0.5 * (grad_u + minitensor::transpose(grad_u));
  Tensor<T> const dev_eps = eps - (minitensor::trace(eps) / 3.) * I;
  return 2. * mu * (dev_eps - pstrain);
}

template <typename T>
T SmallHill<T>::hydro_cauchy(RCP<GlobalResidual<T>> global) {
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const kappa = compute_kappa(E, nu);
  Tensor<T> const grad_u = global->grad_vector_x(0);
  Tensor<T> const eps = 0.5 * (grad_u + minitensor::transpose(grad_u));
  return kappa * trace(eps);
}

template <typename T>
T SmallHill<T>::pressure_scale_factor() {
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const kappa = compute_kappa(E, nu);
  return kappa;
}

template class SmallHill<double>;
template class SmallHill<FADT>;
template class SmallHill<DFADT>;

}
