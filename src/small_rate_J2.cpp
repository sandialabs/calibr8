#include <Eigen/Dense>
#include "control.hpp"
#include "defines.hpp"
#include "fad.hpp"
#include "global_residual.hpp"
#include "macros.hpp"
#include "material_params.hpp"
#include "small_rate_J2.hpp"

namespace calibr8 {

static ParameterList get_valid_local_residual_params() {
  ParameterList p;
  p.set<std::string>("type", "small_J2_mechanics");
  p.set<int>("nonlinear max iters", 0);
  p.set<double>("nonlinear absolute tol", 0.);
  p.set<double>("nonlinear relative tol", 0.);
  p.set<double>("temperature increment", 0.);
  p.sublist("materials");
  return p;
}
static ParameterList get_valid_material_params() {
  ParameterList p;
  p.set<double>("E", 0.);
  p.set<double>("nu", 0.);
  p.set<double>("Y", 0.);
  p.set<double>("S", 0.);
  p.set<double>("D", 0.);
  p.set<double>("cte", 0.);
  return p;
}

template <typename T>
SmallRateJ2<T>::SmallRateJ2(ParameterList const& inputs, int ndims) {

  this->m_params_list = inputs;
  this->m_params_list.validateParameters(get_valid_local_residual_params(), 0);

  int const num_residuals = 2;
  int const num_params = 6;

  this->m_num_residuals = num_residuals;
  this->m_num_eqs.resize(num_residuals);
  this->m_var_types.resize(num_residuals);
  this->m_resid_names.resize(num_residuals);

  this->m_resid_names[0] = "cauchy";
  this->m_var_types[0] = SYM_TENSOR;
  this->m_num_eqs[0] = get_num_eqs(SYM_TENSOR, ndims);

  this->m_resid_names[1] = "alpha";
  this->m_var_types[1] = SCALAR;
  this->m_num_eqs[1] = get_num_eqs(SCALAR, ndims);

  this->m_num_aux_vars = 0;

  m_max_iters = inputs.get<int>("nonlinear max iters");
  m_abs_tol = inputs.get<double>("nonlinear absolute tol");
  m_rel_tol = inputs.get<double>("nonlinear relative tol");
  m_delta_temp = inputs.get<double>("temperature increment");
}

template <typename T>
SmallRateJ2<T>::~SmallRateJ2() {
}

template <typename T>
void SmallRateJ2<T>::init_params() {

  int const num_params = 6;
  this->m_params.resize(num_params);
  this->m_param_names.resize(num_params);

  this->m_param_names[0] = "E";
  this->m_param_names[1] = "nu";
  this->m_param_names[2] = "Y";
  this->m_param_names[3] = "S";
  this->m_param_names[4] = "D";
  this->m_param_names[5] = "cte";

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
    this->m_param_values[es][3] = material_params.get<double>("S");
    this->m_param_values[es][4] = material_params.get<double>("D");
    this->m_param_values[es][5] = material_params.get<double>("cte");
  }

  this->m_active_indices.resize(1);
  this->m_active_indices[0].resize(1);
  this->m_active_indices[0][0] = 0;
}

template <typename T>
void SmallRateJ2<T>::init_variables_impl() {

  int const ndims = this->m_num_dims;
  ALWAYS_ASSERT(ndims == 3); // this is a 3D model
  int const cauchy_idx = 0;
  int const alpha_idx = 1;

  T const alpha = 0.0;
  Tensor<T> const cauchy = minitensor::zero<T>(ndims);

  this->set_scalar_xi(alpha_idx, alpha);
  this->set_sym_tensor_xi(cauchy_idx, cauchy);
}

template <typename T>
void SmallRateJ2<T>::compute_delta_strains(
    RCP<GlobalResidual<T>> global,
    T& delta_vol_eps,
    Tensor<T>& delta_dev_eps) {

  Tensor<T> const grad_u = global->grad_vector_x(0);
  Tensor<T> const grad_u_T = transpose(grad_u);
  Tensor<T> const eps = 0.5 * (grad_u + grad_u_T);
  T const vol_eps = trace(eps);
  Tensor<T> const dev_eps = dev(eps);

  Tensor<T> const grad_u_prev = global->grad_vector_x_prev(0);
  Tensor<T> const grad_u_prev_T = transpose(grad_u_prev);
  Tensor<T> const eps_prev = 0.5 * (grad_u_prev + grad_u_prev_T);
  T const vol_eps_prev = trace(eps_prev);
  Tensor<T> const dev_eps_prev = dev(eps_prev);

  delta_vol_eps = vol_eps - vol_eps_prev;
  delta_dev_eps = dev_eps - dev_eps_prev;
}


template <>
int SmallRateJ2<double>::solve_nonlinear(RCP<GlobalResidual<double>>, int step) {
  return 0;
}

template <>
int SmallRateJ2<FADT>::solve_nonlinear(RCP<GlobalResidual<FADT>> global, int step) {

  (void)step;
  int path;

  // pick an initial guess for the local variables
  {
    int const ndims = global->num_dims();

    FADT const E  = this->m_params[0];
    FADT const nu = this->m_params[1];
    FADT const cte = this->m_params[5];
    FADT const mu = compute_mu(E, nu);
    FADT const kappa = compute_kappa(E, nu);
    Tensor<FADT> cauchy_prev = this->sym_tensor_xi_prev(0);
    Tensor<FADT> const I = minitensor::eye<FADT>(ndims);

    FADT delta_vol_eps;
    Tensor<FADT> delta_dev_eps;
    compute_delta_strains(global, delta_vol_eps, delta_dev_eps);

    Tensor<FADT> const cauchy = cauchy_prev
      + kappa * (delta_vol_eps - 3. * cte * m_delta_temp) * I
      + 2. * mu * delta_dev_eps;
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
    fail("SmallRateJ2:solve_nonlinear failed in %d iterations", m_max_iters);
  }

  return path;

}

template <typename T>
int SmallRateJ2<T>::evaluate(
    RCP<GlobalResidual<T>> global,
    bool force_path,
    int path_in,
    int step) {

  (void)step;
  int path = ELASTIC;
  int const ndims = this->m_num_dims;

  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const Y = this->m_params[2];
  T const S = this->m_params[3];
  T const D = this->m_params[4];
  T const cte  = this->m_params[5];
  T const kappa = compute_kappa(E, nu);
  T const mu = compute_mu(E, nu);

  Tensor<T> const cauchy_old = this->sym_tensor_xi_prev(0);
  T const alpha_old = this->scalar_xi_prev(1);

  Tensor<T> const cauchy = this->sym_tensor_xi(0);
  T const alpha = this->scalar_xi(1);

  T delta_vol_eps;
  Tensor<T> delta_dev_eps;
  compute_delta_strains(global, delta_vol_eps, delta_dev_eps);

  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const s = minitensor::dev(cauchy);
  T const s_mag = minitensor::norm(s);
  Tensor<T> const n = s / s_mag;
  T const sigma_yield = Y + S * (1. - std::exp(-D * alpha));
  T const f = (s_mag - sigma_yield) / val(mu);

  Tensor<T> R_cauchy;
  T R_alpha;

  R_cauchy = cauchy - cauchy_old
    - kappa * (delta_vol_eps - 3. * cte * m_delta_temp) * I
    - 2. * mu * delta_dev_eps;

  if (!force_path) {
    // plastic step
    if (f > m_abs_tol || std::abs(f) < m_abs_tol) {
      T const dgam = alpha - alpha_old;
      R_cauchy += 2. * mu * dgam * n;
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
      R_cauchy += 2. * mu * dgam * n;
      R_alpha = f;
    }
    // elastic step
    else {
      R_alpha = alpha - alpha_old;
    }
  }

  this->set_sym_tensor_R(0, R_cauchy);
  this->set_scalar_R(1, R_alpha);

  return path;

}

template <typename T>
Tensor<T> SmallRateJ2<T>::cauchy(RCP<GlobalResidual<T>> global) {
  return this->sym_tensor_xi(0);
}

template <typename T>
Tensor<T> SmallRateJ2<T>::dev_cauchy(RCP<GlobalResidual<T>> global) {
  Tensor<T> const cauchy = this->sym_tensor_xi(0);
  return minitensor::dev(cauchy);
}

template <typename T>
T SmallRateJ2<T>::hydro_cauchy(RCP<GlobalResidual<T>> global) {
  Tensor<T> const cauchy = this->sym_tensor_xi(0);
  return minitensor::trace(cauchy) / 3.;
}

template <typename T>
T SmallRateJ2<T>::pressure_scale_factor() {
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const kappa = compute_kappa(E, nu);
  return kappa;
}

template class SmallRateJ2<double>;
template class SmallRateJ2<FADT>;

}
