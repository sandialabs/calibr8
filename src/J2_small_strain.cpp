#include <Eigen/Dense>
#include "control.hpp"
#include "defines.hpp"
#include "fad.hpp"
#include "global_residual.hpp"
#include "J2_small_strain.hpp"
#include "material_params.hpp"

namespace calibr8 {

static ParameterList get_valid_params() {
  ParameterList p;
  p.set<std::string>("type", "J2_small_strain");
  p.set<double>("E", 0.);
  p.set<double>("nu", 0.);
  p.set<double>("K", 0.);
  p.set<double>("Y", 0.);
  p.set<double>("cte", 0.);
  p.set<double>("delta_T", 0.);
  p.set<int>("nonlinear max iters", 0);
  p.set<double>("nonlinear absolute tol", 0.);
  p.set<double>("nonlinear relative tol", 0.);
  return p;
}

template <typename T>
J2_small_strain<T>::J2_small_strain(ParameterList const& inputs, int ndims) {

  inputs.validateParameters(get_valid_params(), 0);

  int const num_residuals = 2;
  int const num_params = 6;

  this->m_num_residuals = num_residuals;
  this->m_num_eqs.resize(num_residuals);
  this->m_var_types.resize(num_residuals);
  this->m_resid_names.resize(num_residuals);

  this->m_params.resize(num_params);
  this->m_param_names.resize(num_params);

  this->m_resid_names[0] = "pstrain";
  this->m_var_types[0] = SYM_TENSOR;
  this->m_num_eqs[0] = get_num_eqs(SYM_TENSOR, ndims);

  this->m_resid_names[1] = "alpha";
  this->m_var_types[1] = SCALAR;
  this->m_num_eqs[1] = get_num_eqs(SCALAR, ndims);

  this->m_param_names[0] = "E";
  this->m_param_names[1] = "nu";
  this->m_param_names[2] = "K";
  this->m_param_names[3] = "Y";
  this->m_param_names[4] = "cte";
  this->m_param_names[5] = "delta_T";

  this->m_params[0] = inputs.get<double>("E");
  this->m_params[1] = inputs.get<double>("nu");
  this->m_params[2] = inputs.get<double>("K");
  this->m_params[3] = inputs.get<double>("Y");
  this->m_params[4] = inputs.get<double>("cte");
  this->m_params[5] = inputs.get<double>("delta_T");

  m_max_iters = inputs.get<int>("nonlinear max iters");
  m_abs_tol = inputs.get<double>("nonlinear absolute tol");
  m_rel_tol = inputs.get<double>("nonlinear relative tol");

}

template <typename T>
J2_small_strain<T>::~J2_small_strain() {
}

template <typename T>
void J2_small_strain<T>::init_variables_impl() {

  int const ndims = this->m_num_dims;
  int const pstrain_idx = 0;
  int const alpha_idx = 1;

  T const alpha = 0.0;
  Tensor<T> const pstrain = minitensor::zero<T>(ndims);

  this->set_scalar_xi(alpha_idx, alpha);
  this->set_sym_tensor_xi(pstrain_idx, pstrain);

}

template <>
int J2_small_strain<double>::solve_nonlinear(RCP<GlobalResidual<double>>) {
  return 0;
}

template <>
int J2_small_strain<FADT>::solve_nonlinear(RCP<GlobalResidual<FADT>> global) {

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

    EMatrix const J = this->eigen_jacobian();
    EVector const R = this->eigen_residual();
    EVector const dxi = J.fullPivLu().solve(-R);

    this->add_to_sym_tensor_xi(0, dxi);
    this->add_to_scalar_xi(1, dxi);

    iter++;

  }

  // fail if convergence was not achieved
  if ((iter > m_max_iters) && (!converged)) {
    fail("J2_small_strain:solve_nonlinear failed in %d iterations", m_max_iters);
  }

  return path;

}

template <typename T>
int J2_small_strain<T>::evaluate(
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

  Tensor<T> const pstrain_old = this->sym_tensor_xi_prev(0);
  T const alpha_old = this->scalar_xi_prev(1);

  Tensor<T> const pstrain = this->sym_tensor_xi(0);
  T const alpha = this->scalar_xi(1);

  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const s = this->dev_cauchy(global);
  T const s_mag = minitensor::norm(s);
  Tensor<T> const n = s / s_mag;
  T const sigma_yield = Y + K * alpha;
  T const f = s_mag - sqrt_23 * sigma_yield;

  Tensor<T> R_pstrain;
  T R_alpha;

  if (!force_path) {
    // plastic step
    if (f > m_abs_tol || std::abs(f) < m_abs_tol) {
      T const dgam = sqrt_32 * (alpha - alpha_old);
      R_pstrain = pstrain - pstrain_old - dgam * n;
      R_alpha = (s_mag - sqrt_23 * sigma_yield) / val(mu);
      path = PLASTIC;
    }
    // elastic step
    else {
      R_pstrain = (0. * mu + 1.) * pstrain - pstrain_old;
      R_alpha = alpha - alpha_old + 0. * mu;
      path = ELASTIC;
    }
  }

  // force the path
  else {
    path = path_in;
    // plastic step
    if (path == PLASTIC) {
      T const dgam = sqrt_32 * (alpha - alpha_old);
      R_pstrain = pstrain - pstrain_old - dgam * n;
      R_alpha = (s_mag - sqrt_23 * sigma_yield) / val(mu);
    }
    // elastic step
    else {
      R_pstrain = (0. * mu + 1.) * pstrain - pstrain_old;
      R_alpha = alpha - alpha_old + 0. * mu;
    }
  }

  this->set_sym_tensor_R(0, R_pstrain);
  this->set_scalar_R(1, R_alpha);

  return path;

}

template <typename T>
Tensor<T> J2_small_strain<T>::dev_cauchy(RCP<GlobalResidual<T>> global) {
  int const ndims = global->num_dims();
  Tensor<T> const I = minitensor::eye<T>(ndims);
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const mu = E / (2. * (1. + nu));
  Tensor<T> const pstrain = this->sym_tensor_xi(0);
  Tensor<T> const grad_u = global->grad_vector_x(0);
  Tensor<T> const eps = 0.5 * (grad_u + minitensor::transpose(grad_u));
  Tensor<T> const dev_eps = eps - (minitensor::trace(eps) / 3.) * I;
  return 2. * mu * (dev_eps - pstrain);
}

template <typename T>
Tensor<T> J2_small_strain<T>::cauchy(RCP<GlobalResidual<T>> global, T p) {
  int const ndims = global->num_dims();
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const cte = this->m_params[4];
  T const kappa = compute_kappa(E, nu);
  T const delta_T = this->m_params[5];
  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const dev_sigma = this->dev_cauchy(global);
  Tensor<T> const sigma = dev_sigma - p * I - 3.*kappa*cte*delta_T*I;
  return sigma;
}

template class J2_small_strain<double>;
template class J2_small_strain<FADT>;

}
