#include <Eigen/Dense>
#include "control.hpp"
#include "defines.hpp"
#include "fad.hpp"
#include "global_residual.hpp"
#include "hypo_hosford.hpp"
#include "material_params.hpp"
#include "yield_functions.hpp"

namespace calibr8 {

using minitensor::col;
using minitensor::dev;
using minitensor::dyad;
using minitensor::eig_spd_cos;
using minitensor::eye;
using minitensor::inverse;
using minitensor::norm;
using minitensor::trace;
using minitensor::transpose;
using minitensor::zero;

static ParameterList get_valid_local_residual_params() {
  ParameterList p;
  p.set<std::string>("type", "hypo_hosford");
  p.set<int>("nonlinear max iters", 0);
  p.set<double>("nonlinear absolute tol", 0.);
  p.set<double>("nonlinear relative tol", 0.);
  p.sublist("materials");
  p.set<double>("line search beta", 1.0e-4);
  p.set<double>("line search eta", 0.5);
  p.set<int>("line search max evals", 10);
  p.set<bool>("line search print", false);
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
HypoHosford<T>::HypoHosford(ParameterList const& inputs, int ndims) {

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
  m_ls_beta = inputs.get<double>("line search beta");
  m_ls_eta = inputs.get<double>("line search eta");
  m_ls_max_evals = inputs.get<int>("line search max evals");
  m_ls_print = inputs.get<bool>("line search print");

}

template <typename T>
HypoHosford<T>::~HypoHosford() {
}

template <typename T>
void HypoHosford<T>::init_params() {

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
void HypoHosford<T>::init_variables_impl() {

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


template <>
int HypoHosford<double>::solve_nonlinear(RCP<GlobalResidual<double>>) {
  return 0;
}

template <>
int HypoHosford<FADT>::solve_nonlinear(RCP<GlobalResidual<FADT>> global) {

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

    if (iter == 1) {
      path = this->evaluate(global);
    } else {
      this->evaluate(global, true, path);
    }

    double const R_norm = this->norm_residual();
    double R_norm_prev;
    if (iter == 1) {
      R_norm_0 = R_norm;
      R_norm_prev = 10. * R_norm;
    } else {
      R_norm_prev = R_norm;
    }
    if (R_norm_prev < R_norm) {
      print("(local) Newton iter %d: RESIDUAL INCREASE!!!", iter);
      print("R_norm_prev = %e, R_norm = %e", R_norm_prev, R_norm);
    }

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

    // Use the RMA in Section 3 of http://dx.doi.org/10.1016/j.cma.2016.11.026
    // This approach doesn't work for the global line search or small strain hosford plasticity
    {
      this->evaluate(global, true, path);

      double const R_0 = R_norm;
      double const psi_0 = 0.5 * R_0 * R_0;
      double const psi_0_deriv = -2. * psi_0;

      int j = 1;
      double alpha_prev = 1.;
      double alpha_j = 1.;
      double alpha_diff = alpha_j - alpha_prev;
      double R_j = this->norm_residual();
      double psi_j = 0.5 * R_j * R_j;

      while (psi_j >= ((1. - 2. * m_ls_beta * alpha_j) * psi_0)) {

        alpha_prev = alpha_j;
        alpha_j  = std::max(m_ls_eta * alpha_j,
            -(std::pow(alpha_j, 2) * psi_0_deriv) /
             (2. * (psi_j - psi_0 - alpha_j * psi_0_deriv)));

        if (m_ls_print) {
          print(" > (local) residual increase -- line search alpha_%d = %.2e",
              j, alpha_j);
        }

        if (j == m_ls_max_evals) {
          print(" > (local) Reached max line search evals");
          break;
        }

        ++j;

        alpha_diff = alpha_j - alpha_prev;
        this->add_to_sym_tensor_xi(0, alpha_diff * dxi);
        this->add_to_scalar_xi(1, alpha_diff * dxi);

        path = this->evaluate(global, true, path);

        R_j = this->norm_residual();
        psi_j = 0.5 * R_j * R_j;

      }
    }

    iter++;

  }

  if ((iter > m_max_iters) && (!converged)) {
    std::cout << "HypoHosford:solve_nonlinear failed in "  << iter << " iterations\n";
    return -1;
  }

  return path;
}

template <>
int HypoHosford<DFADT>::solve_nonlinear(RCP<GlobalResidual<DFADT>>) {
  return 0;
}

template <typename T>
void HypoHosford<T>::evaluate_phi_and_normal(
    T const& a,
    T& phi,
    Tensor<T>& n) {

  Tensor<T> const TC = this->sym_tensor_xi(0);
  T const vm_stress = std::sqrt(3. / 2.) * norm(dev(TC));

  std::pair<Tensor<T>, Tensor<T>> const eigen_decomp = eig_spd_cos(TC);
  Tensor<T> const cauchy_eigvecs = eigen_decomp.first;
  Tensor<T> const cauchy_eigvals = eigen_decomp.second;

  Vector<T> const eigvec_0 = col(cauchy_eigvecs, 0);
  Vector<T> const eigvec_1 = col(cauchy_eigvecs, 1);
  Vector<T> const eigvec_2 = col(cauchy_eigvecs, 2);

  T const vm_scaled_eigval_0 = cauchy_eigvals(0, 0) / vm_stress;
  T const vm_scaled_eigval_1 = cauchy_eigvals(1, 1) / vm_stress;
  T const vm_scaled_eigval_2 = cauchy_eigvals(2, 2) / vm_stress;

  phi = vm_stress * std::pow(0.5 * (std::pow(std::abs(vm_scaled_eigval_0 - vm_scaled_eigval_1), a)
    + std::pow(std::abs(vm_scaled_eigval_1 - vm_scaled_eigval_2), a)
    + std::pow(std::abs(vm_scaled_eigval_2 - vm_scaled_eigval_0), a)), 1. / a);

  T const phi_scaled_eigval_0 = cauchy_eigvals(0, 0) / phi;
  T const phi_scaled_eigval_1 = cauchy_eigvals(1, 1) / phi;
  T const phi_scaled_eigval_2 = cauchy_eigvals(2, 2) / phi;

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
int HypoHosford<T>::evaluate(
    RCP<GlobalResidual<T>> global,
    bool force_path,
    int path_in) {

  int path = ELASTIC;
  int const ndims = this->m_num_dims;

  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const Y = this->m_params[2];
  T const a = this->m_params[3];
  T const K = this->m_params[4];
  T const S = this->m_params[5];
  T const D = this->m_params[6];
  T const mu = compute_mu(E, nu);
  T const lambda = compute_lambda(E, nu);

  Tensor<T> const TC_old = this->sym_tensor_xi_prev(0);
  T const alpha_old = this->scalar_xi_prev(1);

  Tensor<T> const TC = this->sym_tensor_xi(0);
  T const alpha = this->scalar_xi(1);

  T phi = 0.;
  Tensor<T> n = zero<T>(3);
  evaluate_phi_and_normal(a, phi, n);

  T const scale_factor = 2. * mu;

  T const flow_stress = Y + S * (1. - std::exp(-D * alpha));
  T const f = (phi - flow_stress) / scale_factor;

  Tensor<T> R_TC;
  T R_alpha;

  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const d = eval_d(global);
  R_TC = (TC - TC_old - lambda * trace(d) * I - 2. * mu * d) / scale_factor;


  if (!force_path) {
    // plastic step
    if (f > m_abs_tol || std::abs(f) < m_abs_tol) {
      T const dgam = alpha - alpha_old;
      // scale_factor in R_TC removes the (2. * mu) multiplier
      R_TC += dgam * n;
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
      // scale_factor in R_TC removes the (2. * mu) multiplier
      R_TC += dgam * n;
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

template <typename T>
Tensor<T> HypoHosford<T>::rotated_cauchy(RCP<GlobalResidual<T>> global) {
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
Tensor<T> HypoHosford<T>::cauchy(RCP<GlobalResidual<T>> global) {
  int const pressure_idx = 1;
  T const p = global->scalar_x(pressure_idx);
  int const ndims = this->m_num_dims;
  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const dev_RC = this->dev_cauchy(global);
  Tensor<T> const sigma = dev_RC - p * I;
  return sigma;
}

template <typename T>
Tensor<T> HypoHosford<T>::dev_cauchy(RCP<GlobalResidual<T>> global) {
  Tensor<T> const RC = this->rotated_cauchy(global);
  return dev(RC);
}

template <typename T>
T HypoHosford<T>::hydro_cauchy(RCP<GlobalResidual<T>> global) {
  Tensor<T> const RC = this->rotated_cauchy(global);
  return trace(RC) / 3.;
}


template <typename T>
T HypoHosford<T>::pressure_scale_factor() {
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const kappa = compute_kappa(E, nu);
  return kappa;
}


template class HypoHosford<double>;
template class HypoHosford<FADT>;
template class HypoHosford<DFADT>;

}
