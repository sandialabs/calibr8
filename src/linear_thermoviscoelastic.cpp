#include <fstream>
#include <iomanip>
#include <Eigen/Dense>
#include "control.hpp"
#include "defines.hpp"
#include "fad.hpp"
#include "global_residual.hpp"
#include "linear_thermoviscoelastic.hpp"
#include "macros.hpp"
#include "material_params.hpp"

namespace calibr8 {

using minitensor::det;
using minitensor::dev;
using minitensor::inverse;
using minitensor::trace;
using minitensor::transpose;

static ParameterList get_valid_local_residual_params() {
  ParameterList p;
  p.set<std::string>("type", "linear_thermoviscoelastic");
  p.set<bool>("mixed formulation", true);
  p.sublist("materials");
  p.set<int>("num steps", 1);
  p.set<double>("temperature increment", 1.);
  p.set<double>("time increment", 1.);
  p.set<double>("initial temperature", 1.);
  p.set<double>("reference temperature", 1.);
  p.set<double>("WLF C_1", 1.);
  p.set<double>("WLF C_2", 1.);
  p.sublist("prony files");
  return p;
}

static ParameterList get_valid_material_params() {
  ParameterList p;
  p.set<double>("K_g", 0.);
  p.set<double>("mu_g", 0.);
  p.set<double>("alpha_g", 0.);
  p.set<double>("K_inf", 0.);
  p.set<double>("mu_inf", 0.);
  p.set<double>("alpha_inf", 0.);
  return p;
}

template <typename T>
void LTVE<T>::read_prony_series(ParameterList const& prony_files) {
  std::string const& vol_prony_file = prony_files.get<std::string>("volume prony file");
  std::string const& shear_prony_file = prony_files.get<std::string>("shear prony file");
  std::string line;
  Array1D<double> prony_data(2);

  std::ifstream vol_in_file(vol_prony_file);
  while (getline(vol_in_file, line)) {
    std::stringstream ss(line);
    if (ss >> prony_data[0] >> prony_data[1]) {
      m_vol_prony.push_back(prony_data);
    }
  }

  std::ifstream shear_in_file(shear_prony_file);
  while (getline(shear_in_file, line)) {
    std::stringstream ss(line);
    if (ss >> prony_data[0] >> prony_data[1]) {
      m_shear_prony.push_back(prony_data);
    }
  }
}

template <typename T>
void LTVE<T>::compute_temperature(ParameterList const& inputs) {
  int const num_steps = inputs.get<int>("num steps");
  double const initial_temp = inputs.get<double>("initial temperature");
  m_delta_temp = inputs.get<double>("temperature increment");
  m_delta_t = inputs.get<double>("time increment");
  m_temp_ref = inputs.get<double>("reference temperature");
  m_C_1 = inputs.get<double>("WLF C_1");
  m_C_2 = inputs.get<double>("WLF C_2");
  m_temperature.resize(num_steps + 1);
  m_temperature[0] = initial_temp;
  for (size_t t = 1; t <= num_steps; ++t) {
    m_temperature[t] = initial_temp + t * m_delta_temp;
  }
}

template <typename T>
Array1D<double> LTVE<T>::compute_J3_k(
    double const psi,
    Array1D<double> const& J3_k_prev,
    bool compute_func) {

  int const num_terms = m_vol_prony.size();
  Array1D<double> J3_k(num_terms);

  if (compute_func) {
    for (size_t n = 0; n < num_terms; ++n) {
      double const a_tau = std::pow(10., psi) * m_vol_prony[n][0];
      J3_k[n] = a_tau / (a_tau + m_delta_t) * (J3_k_prev[n] + m_delta_temp);
    }
  } else {
    for (size_t n = 0; n < num_terms; ++n) {
      double const a_tau = std::pow(10., psi) * m_vol_prony[n][0];
      J3_k[n] = std::log(10.) * a_tau * m_delta_t / std::pow(a_tau + m_delta_t, 2)
          * (J3_k_prev[n] + m_delta_temp);
    }
  }

  return J3_k;
}

template <typename T>
double LTVE<T>::compute_J3(Array1D<double> const& J3_k) {
  int const num_terms = m_vol_prony.size();
  double J3 = 0.;
  for (int n = 0; n < num_terms; ++n) {
    J3 += m_vol_prony[n][1] * J3_k[n];
  }
  return J3;
}

template <typename T>
void LTVE<T>::residual_and_deriv(
    double const psi,
    Array1D<double> const& J3_k_prev,
    double const temp,
    double& r_val,
    double& r_deriv) {

  Array1D<double> J3_k = this->compute_J3_k(psi, J3_k_prev);
  Array1D<double> J3_k_deriv = this->compute_J3_k(psi, J3_k_prev, false);
  double const J3 = compute_J3(J3_k);
  double const J3_deriv = compute_J3(J3_k_deriv);

  double const N = temp - m_temp_ref - J3;
  r_val = psi + (m_C_1 * N) / (m_C_2 + N);
  r_deriv = 1. - m_C_1 * m_C_2 * J3_deriv / std::pow(m_C_2 + N, 2);
}

template <typename T>
double LTVE<T>::lag_nonlinear_solve(
    double const psi,
    Array1D<double> const& J3_k_prev,
    double const temp,
    double const tol,
    int const max_iters) {

  int iter = 0;
  bool converged = false;

  double R = 10. * tol;
  double dR = 0.;
  double psi_newton = psi;

  while ((!converged) && (iter < max_iters)) {
    this->residual_and_deriv(psi_newton, J3_k_prev, temp, R, dR);
    if (std::abs(R) < tol) {
      converged = true;
      break;
    }
    psi_newton -= R / dR;
    iter += 1;
  }
  return psi_newton;
}


template <typename T>
void LTVE<T>::compute_shift_factors() {
  int const total_num_steps = m_temperature.size(); // includes initial condition
  double psi = -8.;
  m_log10_shift_factor.resize(total_num_steps);
  m_log10_shift_factor[0] = psi;
  m_J3.resize(total_num_steps);
  m_J3[0] = 0.;
  Array1D<double> J3_k(m_vol_prony.size(), 0.);

  for (size_t i = 1; i < total_num_steps; ++i) {
    psi = lag_nonlinear_solve(m_log10_shift_factor[i-1], J3_k,
        m_temperature[i]);
    J3_k = compute_J3_k(psi, J3_k);
    m_log10_shift_factor[i] = psi;
    m_J3[i] = compute_J3(J3_k);
  }
}

template <>
EMatrix LTVE<double>::daux_dxT(RCP<GlobalResidual<double>> global, int step) {
  int const num_global_dofs = global->num_dofs();
  EMatrix daux_dxT = EMatrix::Zero(num_global_dofs, this->m_num_aux_dofs);
  return daux_dxT;
}

template <>
EMatrix LTVE<FADT>::daux_dxT(RCP<GlobalResidual<FADT>> global, int step) {
  int const num_global_dofs = global->num_dofs();
  EMatrix daux_dxT = EMatrix::Zero(num_global_dofs, this->m_num_aux_dofs);

  EVector const scale_factor = this->daux_dchi_prev_diag(step);

  int const num_vol_prony_terms = m_vol_prony.size();
  int const num_shear_prony_terms = m_shear_prony.size();

  int const ndims = this->m_num_dims;
  int const num_sym_tensor_eqs = get_num_eqs(SYM_TENSOR, ndims);

  Tensor<FADT> const grad_u = global->grad_vector_x(0);
  Tensor<FADT> const grad_u_T = transpose(grad_u);
  Tensor<FADT> const eps = 0.5 * (grad_u + grad_u_T);
  FADT const vol_eps = trace(eps);
  Tensor<FADT> const dev_eps = dev(eps);

  Tensor<FADT> const grad_u_prev = global->grad_vector_x_prev(0);
  Tensor<FADT> const grad_u_prev_T = transpose(grad_u_prev);
  Tensor<FADT> const eps_prev = 0.5 * (grad_u_prev + grad_u_prev_T);
  FADT const vol_eps_prev = trace(eps_prev);
  Tensor<FADT> const dev_eps_prev = dev(eps_prev);

  FADT const delta_vol_eps = vol_eps - vol_eps_prev;
  Tensor<FADT> const delta_dev_eps = dev_eps - dev_eps_prev;

  EVector delta_vol_eps_derivs = EVector::Zero(num_global_dofs);
  for (int d = 0; d < num_global_dofs; ++d) {
    delta_vol_eps_derivs(d) = delta_vol_eps.fastAccessDx(d);
  }

  EMatrix delta_dev_eps_derivs = EMatrix::Zero(num_global_dofs, num_sym_tensor_eqs);
  int sym_comp = 0;
  for (int i = 0; i < ndims; ++i) {
    for (int j = i; j < ndims; ++j) {
      for (int d = 0; d < num_global_dofs; ++d) {
        delta_dev_eps_derivs(d, sym_comp) = delta_dev_eps(i, j).fastAccessDx(d);
      }
      ++sym_comp;
    }
  }

  double const a = std::pow(10., m_log10_shift_factor[step]);
  int m = 0;

  for (int k = 0; k < num_vol_prony_terms; ++k) {
    daux_dxT.col(m) = scale_factor(m) * delta_vol_eps_derivs;
    ++m;
  }

  for (int k = 0; k < num_shear_prony_terms; ++k) {
    for (int i = 0; i < num_sym_tensor_eqs; ++i) {
      daux_dxT.col(m) = scale_factor(m) * delta_dev_eps_derivs.col(i);
      ++m;
    }
  }

  return daux_dxT;
}

template <typename T>
EVector LTVE<T>::daux_dchi_prev_diag(int step) {

  EVector daux_dchi_prev_diag(this->m_num_aux_dofs);

  int const num_vol_prony_terms = m_vol_prony.size();
  int const num_shear_prony_terms = m_shear_prony.size();

  int const ndims = this->m_num_dims;
  int const num_sym_tensor_eqs = get_num_eqs(SYM_TENSOR, ndims);

  double const a = std::pow(10., m_log10_shift_factor[step]);
  int m = 0;

  for (int k = 0; k < num_vol_prony_terms; ++k) {
    double const a_tau = a * m_vol_prony[k][0];
    daux_dchi_prev_diag(m) = -a_tau / (a_tau + m_delta_t);
    ++m;
  }

  for (int k = 0; k < num_shear_prony_terms; ++k) {
    double const a_tau = a * m_shear_prony[k][0];
    for (int i = 0; i < num_sym_tensor_eqs; ++i) {
      daux_dchi_prev_diag(m) = -a_tau / (a_tau + m_delta_t);
      ++m;
    }
  }

  return daux_dchi_prev_diag;
}

template <typename T>
EVector LTVE<T>::dlocal_dchi_prev_diag(int step) {

  EVector dlocal_dchi_prev_diag(this->m_num_aux_dofs);

  double const K_g = val(this->m_params[0]);
  double const mu_g = val(this->m_params[1]);
  double const K_inf = val(this->m_params[3]);
  double const mu_inf = val(this->m_params[4]);

  double const delta_K = K_g - K_inf;
  double const delta_mu = mu_g - mu_inf;

  int const num_vol_prony_terms = m_vol_prony.size();
  int const num_shear_prony_terms = m_shear_prony.size();

  int const ndims = this->m_num_dims;
  int const num_sym_tensor_eqs = get_num_eqs(SYM_TENSOR, ndims);

  double const a = std::pow(10., m_log10_shift_factor[step]);
  int m = 0;

  for (int k = 0; k < num_vol_prony_terms; ++k) {
    double const a_tau = a * m_vol_prony[k][0];
    dlocal_dchi_prev_diag(m) = -delta_K * m_delta_t * m_vol_prony[k][1]
        / (a_tau + m_delta_t);
    ++m;
  }

  for (int k = 0; k < num_shear_prony_terms; ++k) {
    double const a_tau = a * m_shear_prony[k][0];
    for (int i = 0; i < num_sym_tensor_eqs; ++i) {
      dlocal_dchi_prev_diag(m) = -2. * val(delta_mu) * m_delta_t * m_shear_prony[k][1]
          / (a_tau + m_delta_t);
      ++m;
    }
  }

  return dlocal_dchi_prev_diag;
}

template <typename T>
EVector LTVE<T>::eigen_aux_residual(RCP<GlobalResidual<T>> global, int step) {
  EVector aux_residual = EVector::Zero(this->m_num_aux_dofs);

  double const K_g = val(this->m_params[0]);
  double const mu_g = val(this->m_params[1]);
  double const K_inf = val(this->m_params[3]);
  double const mu_inf = val(this->m_params[4]);

  double const delta_K = K_g - K_inf;
  double const delta_mu = mu_g - mu_inf;

  int const ndims = this->m_num_dims;
  int const num_sym_tensor_eqs = get_num_eqs(SYM_TENSOR, ndims);

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

  T const delta_vol_eps = vol_eps - vol_eps_prev;
  Tensor<T> const delta_dev_eps = dev_eps - dev_eps_prev;

  double const a = std::pow(10., m_log10_shift_factor[step]);

  int v = 0;
  int m = 0;

  int const num_vol_prony_terms = m_vol_prony.size();
  for (int k = 0; k < num_vol_prony_terms; ++k) {
    double const a_tau = a * m_vol_prony[k][0];
    double const J_vol_k = val(this->scalar_chi(v));
    double const J_vol_k_prev = val(this->scalar_chi_prev(v));
    aux_residual(m) = J_vol_k - a_tau / (a_tau + m_delta_t)
        * (J_vol_k_prev + val(delta_vol_eps));
    ++v;
    ++m;
  }

  int const num_shear_prony_terms = m_shear_prony.size();
  for (int k = 0; k < num_shear_prony_terms; ++k) {
    double const a_tau = a * m_shear_prony[k][0];
    Tensor<T> const J_shear_k = this->sym_tensor_chi(v);
    Tensor<T> const J_shear_k_prev = this->sym_tensor_chi_prev(v);
    for (int i = 0; i < num_sym_tensor_eqs; ++i) {
      for (int j = i; j < num_sym_tensor_eqs; ++j) {
        aux_residual(m) = val(J_shear_k(i, j)) - a_tau / (a_tau + m_delta_t)
            * (val(J_shear_k_prev(i, j)) + val(delta_dev_eps(i, j)));
        ++m;
      }
    }
    ++v;
  }

  return aux_residual;
}

template <typename T>
LTVE<T>::LTVE(ParameterList const& inputs, int ndims) {

  this->m_params_list = inputs;
  this->m_params_list.validateParameters(get_valid_local_residual_params(), 0);

  bool is_mixed = inputs.get<bool>("mixed formulation");
  if (is_mixed) {
    m_mode = MIXED;
  } else {
    m_mode = DISPLACEMENT;
  }

  ParameterList const& prony_params = inputs.sublist("prony files");
  this->read_prony_series(prony_params);
  this->compute_temperature(inputs);

  int const num_residuals = 1;

  this->m_num_residuals = num_residuals;
  this->m_num_eqs.resize(num_residuals);
  this->m_var_types.resize(num_residuals);
  this->m_resid_names.resize(num_residuals);

  int const num_sym_tensor_eqs = get_num_eqs(SYM_TENSOR, ndims);
  int const num_scalar_eqs = get_num_eqs(SCALAR, ndims);

  this->m_resid_names[0] = "cauchy";
  this->m_var_types[0] = SYM_TENSOR;
  this->m_num_eqs[0] = num_sym_tensor_eqs;

  int const num_vol_prony_terms = m_vol_prony.size();
  int const num_shear_prony_terms = m_shear_prony.size();

  int const num_aux_variables = num_vol_prony_terms + num_shear_prony_terms;
  this->m_num_aux_vars = num_aux_variables;
  this->m_aux_var_num_eqs.resize(num_aux_variables);
  this->m_aux_var_types.resize(num_aux_variables);
  this->m_aux_var_names.resize(num_aux_variables);

  int v = 0;

  for (int k = 0; k < num_vol_prony_terms; ++k) {
    this->m_aux_var_names[v] = "J_vol_" + std::to_string(k);
    this->m_aux_var_types[v] = SCALAR;
    this->m_aux_var_num_eqs[v] = num_scalar_eqs;
    ++v;
  }

  for (int k = 0; k < num_shear_prony_terms; ++k) {
    this->m_aux_var_names[v] = "J_shear_" + std::to_string(k);
    this->m_aux_var_types[v] = SYM_TENSOR;
    this->m_aux_var_num_eqs[v] = num_sym_tensor_eqs;
    ++v;
  }
}

template <typename T>
LTVE<T>::~LTVE() {
}

template <typename T>
void LTVE<T>::init_params() {
  int const num_params = 6;
  this->m_params.resize(num_params);
  this->m_param_names.resize(num_params);
  this->m_param_names[0] = "K_g";
  this->m_param_names[1] = "mu_g";
  this->m_param_names[2] = "alpha_g";
  this->m_param_names[3] = "K_inf";
  this->m_param_names[4] = "mu_inf";
  this->m_param_names[5] = "alpha_inf";
  int const num_elem_sets = this->m_elem_set_names.size();
  resize(this->m_param_values, num_elem_sets, num_params);
  ParameterList& all_material_params = this->m_params_list.sublist("materials", true);
  for (int es = 0; es < num_elem_sets; ++es) {
    std::string const& elem_set_name = this->m_elem_set_names[es];
    ParameterList& material_params = all_material_params.sublist(elem_set_name, true);
    material_params.validateParameters(get_valid_material_params(), 0);
    this->m_param_values[es][0] = material_params.get<double>("K_g");
    this->m_param_values[es][1] = material_params.get<double>("mu_g");
    this->m_param_values[es][2] = material_params.get<double>("alpha_g");
    this->m_param_values[es][3] = material_params.get<double>("K_inf");
    this->m_param_values[es][4] = material_params.get<double>("mu_inf");
    this->m_param_values[es][5] = material_params.get<double>("alpha_inf");
  }
  this->m_active_indices.resize(1);
  this->m_active_indices[0].resize(1);
  this->m_active_indices[0][0] = 0;

  this->compute_shift_factors();
}

template <typename T>
void LTVE<T>::init_variables_impl() {
  int const ndims = this->m_num_dims;
  ALWAYS_ASSERT(ndims == 3); // not implemented for 2D yet
  int const cauchy_idx = 0;
  Tensor<T> const cauchy = minitensor::zero<T>(ndims);
  this->set_sym_tensor_xi(cauchy_idx, cauchy);
}

template<typename T>
void LTVE<T>::compute_present_aux_variables(
    RCP<GlobalResidual<T>> global,
    int step) {

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

  T const delta_vol_eps = vol_eps - vol_eps_prev;
  Tensor<T> const delta_dev_eps = dev_eps - dev_eps_prev;

  double const a = std::pow(10., m_log10_shift_factor[step]);

  int v = 0;

  int const num_vol_prony_terms = m_vol_prony.size();
  for (int k = 0; k < num_vol_prony_terms; ++k) {
    double const a_tau = a * m_vol_prony[k][0];
    T const J_vol_k_prev = this->scalar_chi_prev(v);
    T const J_vol_k = a_tau / (a_tau + m_delta_t)
        * (J_vol_k_prev + delta_vol_eps);
    this->set_scalar_chi(v, J_vol_k);
    ++v;
  }

  int const num_shear_prony_terms = m_shear_prony.size();
  for (int k = 0; k < num_shear_prony_terms; ++k) {
    double const a_tau = a * m_shear_prony[k][0];
    Tensor<T> const J_shear_k_prev = this->sym_tensor_chi_prev(v);
    Tensor<T> const J_shear_k = a_tau / (a_tau + m_delta_t)
        * (J_shear_k_prev + delta_dev_eps);
    this->set_sym_tensor_chi(v, J_shear_k);
    ++v;
  }
}

template<>
void LTVE<double>::compute_past_aux_variables(
    RCP<GlobalResidual<double>> global,
    int step) {

  Tensor<double> const grad_u = global->grad_vector_x(0);
  Tensor<double> const grad_u_T = transpose(grad_u);
  Tensor<double> const eps = 0.5 * (grad_u + grad_u_T);
  double const vol_eps = trace(eps);
  Tensor<double> const dev_eps = dev(eps);

  Tensor<double> const grad_u_prev = global->grad_vector_x_prev(0);
  Tensor<double> const grad_u_prev_T = transpose(grad_u_prev);
  Tensor<double> const eps_prev = 0.5 * (grad_u_prev + grad_u_prev_T);
  double const vol_eps_prev = trace(eps_prev);
  Tensor<double> const dev_eps_prev = dev(eps_prev);

  double const delta_vol_eps = vol_eps - vol_eps_prev;
  Tensor<double> const delta_dev_eps = dev_eps - dev_eps_prev;

  double const a = std::pow(10., m_log10_shift_factor[step]);

  int v = 0;

  int const num_vol_prony_terms = m_vol_prony.size();
  for (int k = 0; k < num_vol_prony_terms; ++k) {
    double const a_tau = a * m_vol_prony[k][0];
    double const J_vol_k = this->scalar_chi(v);
    double const J_vol_k_prev = (a_tau + m_delta_t) / a_tau
        * J_vol_k - delta_vol_eps;
    this->set_scalar_chi_prev(v, J_vol_k_prev);
    ++v;
  }

  int const num_shear_prony_terms = m_shear_prony.size();
  for (int k = 0; k < num_shear_prony_terms; ++k) {
    double const a_tau = a * m_shear_prony[k][0];
    Tensor<double> const J_shear_k = this->sym_tensor_chi(v);
    Tensor<double> const J_shear_k_prev = (a_tau + m_delta_t) / a_tau
        * J_shear_k - delta_dev_eps;
    this->set_sym_tensor_chi_prev(v, J_shear_k_prev);
    ++v;
  }

}

template<>
void LTVE<FADT>::compute_past_aux_variables(
    RCP<GlobalResidual<FADT>> global,
    int step) {}

template <>
int LTVE<double>::solve_nonlinear(RCP<GlobalResidual<double>>, int step) {
  return 0;
}

template <>
int LTVE<FADT>::solve_nonlinear(RCP<GlobalResidual<FADT>> global, int step) {

  int path;

  // pick an initial guess for the local variables
  {
    int const ndims = global->num_dims();

    FADT const K_g = this->m_params[0];
    FADT const mu_g = this->m_params[1];
    FADT const alpha_g = this->m_params[2];
    FADT const K_inf = this->m_params[3];
    FADT const mu_inf = this->m_params[4];
    FADT const alpha_inf = this->m_params[5];

    FADT const delta_K = K_g - K_inf;
    FADT const delta_mu = mu_g - mu_inf;
    FADT const delta_alpha_K = alpha_g * K_g - alpha_inf * K_inf;

    Tensor<FADT> const I = minitensor::eye<FADT>(ndims);

    Tensor<FADT> const grad_u = global->grad_vector_x(0);
    Tensor<FADT> const grad_u_T = transpose(grad_u);
    Tensor<FADT> const eps = 0.5 * (grad_u + grad_u_T);
    FADT const vol_eps = trace(eps);
    Tensor<FADT> const dev_eps = dev(eps);

    Tensor<FADT> const grad_u_prev = global->grad_vector_x_prev(0);
    Tensor<FADT> const grad_u_prev_T = transpose(grad_u_prev);
    Tensor<FADT> const eps_prev = 0.5 * (grad_u_prev + grad_u_prev_T);
    FADT const vol_eps_prev = trace(eps_prev);
    Tensor<FADT> const dev_eps_prev = dev(eps_prev);

    Tensor<FADT> cauchy_prev = this->sym_tensor_xi_prev(0);

    FADT const delta_vol_eps = vol_eps - vol_eps_prev;
    Tensor<FADT> const delta_dev_eps = dev_eps - dev_eps_prev;

    double const a = std::pow(10., m_log10_shift_factor[step]);

    FADT bar_K = K_inf;
    FADT bar_mu = mu_inf;
    FADT bar_alpha_K = alpha_inf * K_inf;
    FADT visco_vol = 0.;
    Tensor<FADT> visco_shear = minitensor::zero<FADT>(ndims);

    int v = 0;

    int const num_vol_prony_terms = m_vol_prony.size();
    for (int k = 0; k < num_vol_prony_terms; ++k) {
      double const vol_weight_k = m_vol_prony[k][1];
      bar_K += vol_weight_k * delta_K;
      bar_alpha_K += vol_weight_k * delta_alpha_K;
      double const a_tau = a * m_vol_prony[k][0];
      FADT const J_vol_k_prev = this->scalar_chi_prev(v);
      visco_vol += vol_weight_k / (a_tau + m_delta_t)
          * (J_vol_k_prev + delta_vol_eps);
      ++v;
    }

    int const num_shear_prony_terms = m_shear_prony.size();
    for (int k = 0; k < num_shear_prony_terms; ++k) {
      double const shear_weight_k = m_shear_prony[k][1];
      bar_mu += shear_weight_k * delta_mu;
      double const a_tau = a * m_shear_prony[k][0];
      Tensor<FADT> const J_shear_k_prev = this->sym_tensor_chi_prev(v);
      visco_shear += shear_weight_k / (a_tau + m_delta_t)
          * (J_shear_k_prev + delta_dev_eps);
      ++v;
    }

    Tensor<FADT> const cauchy = cauchy_prev
        + bar_K * delta_vol_eps * I
        + 2. * bar_mu * delta_dev_eps
        - 3. * K_inf * alpha_inf * m_delta_temp * I
        - 3. * delta_alpha_K * (m_J3[step] - m_J3[step - 1]) * I
        - m_delta_t * delta_K * visco_vol * I
        - 2. * m_delta_t * delta_mu * visco_shear;

    this->set_sym_tensor_xi(0, cauchy);

  }

  path = this->evaluate(global, false, 0, step);

  EMatrix const J = this->eigen_jacobian(this->m_num_dofs);
  EVector const R = this->eigen_residual();
  EVector const dxi = J.fullPivLu().solve(-R);

  this->add_to_sym_tensor_xi(0, dxi);

  this->compute_present_aux_variables(global, step);

  return path;

}

template <typename T>
int LTVE<T>::evaluate(
    RCP<GlobalResidual<T>> global,
    bool force_path,
    int path_in,
    int step) {

  // always elastic
  int const path = 0;

  Tensor<T> const cauchy = this->sym_tensor_xi(0);
  Tensor<T> const cauchy_prev = this->sym_tensor_xi_prev(0);

  int const ndims = this->m_num_dims;
  Tensor<T> const I = minitensor::eye<T>(ndims);

  T const K_g = this->m_params[0];
  T const mu_g = this->m_params[1];
  T const alpha_g = this->m_params[2];
  T const K_inf = this->m_params[3];
  T const mu_inf = this->m_params[4];
  T const alpha_inf = this->m_params[5];

  T const delta_K = K_g - K_inf;
  T const delta_mu = mu_g - mu_inf;
  T const delta_alpha_K = alpha_g * K_g - alpha_inf * K_inf;

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

  T const delta_vol_eps = vol_eps - vol_eps_prev;
  Tensor<T> const delta_dev_eps = dev_eps - dev_eps_prev;

  double const a = std::pow(10., m_log10_shift_factor[step]);

  T bar_K = K_inf;
  T bar_mu = mu_inf;
  T bar_alpha_K = alpha_inf * K_inf;
  T visco_vol = 0.;
  Tensor<T> visco_shear = minitensor::zero<T>(ndims);

  int v = 0;

  int const num_vol_prony_terms = m_vol_prony.size();
  for (int k = 0; k < num_vol_prony_terms; ++k) {
    double const vol_weight_k = m_vol_prony[k][1];
    bar_K += vol_weight_k * delta_K;
    bar_alpha_K += vol_weight_k * delta_alpha_K;
    double const a_tau = a * m_vol_prony[k][0];
    T const J_vol_k_prev = this->scalar_chi_prev(v);
    visco_vol += vol_weight_k / (a_tau + m_delta_t)
        * (J_vol_k_prev + delta_vol_eps);
    ++v;
  }

  int const num_shear_prony_terms = m_shear_prony.size();
  for (int k = 0; k < num_shear_prony_terms; ++k) {
    double const shear_weight_k = m_shear_prony[k][1];
    bar_mu += shear_weight_k * delta_mu;
    double const a_tau = a * m_shear_prony[k][0];
    Tensor<T> const J_shear_k_prev = this->sym_tensor_chi_prev(v);
    visco_shear += shear_weight_k / (a_tau + m_delta_t)
        * (J_shear_k_prev + delta_dev_eps);
    ++v;
  }

  Tensor<T> const R_cauchy = cauchy - cauchy_prev
      - bar_K * delta_vol_eps * I
      - 2. * bar_mu * delta_dev_eps
      + 3. * K_inf * alpha_inf * m_delta_temp * I
      + 3. * delta_alpha_K * (m_J3[step] - m_J3[step - 1]) * I
      + m_delta_t * delta_K * visco_vol * I
      + 2. * m_delta_t * delta_mu * visco_shear;

  this->set_sym_tensor_R(0, R_cauchy);

  return path;
}

template <typename T>
Tensor<T> LTVE<T>::dev_cauchy(RCP<GlobalResidual<T>> global) {
  Tensor<T> const local_cauchy = this->sym_tensor_xi(0);
  int const ndims = this->m_num_dims;
  Tensor<T> const I = minitensor::eye<T>(ndims);
  return local_cauchy - this->hydro_cauchy(global) * I;
}

template <typename T>
T LTVE<T>::hydro_cauchy(RCP<GlobalResidual<T>> global) {
  Tensor<T> const local_cauchy = this->sym_tensor_xi(0);
  int const ndims = this->m_num_dims;
  Tensor<T> const I = minitensor::eye<T>(ndims);
  if (ndims == 3) {
    return trace(local_cauchy) / 3.;
  } else {
    T const nu = this->m_params[1];
    return (1. + nu) * trace(local_cauchy) / 3.;
  }
}

template <typename T>
T LTVE<T>::pressure_scale_factor() {
  return this->m_params[0];
}

template <typename T>
Tensor<T> LTVE<T>::cauchy(RCP<GlobalResidual<T>> global) {
  if (m_mode == MIXED) {
    return cauchy_mixed(global);
  } else {
    return this->sym_tensor_xi(0);
  }
}

template <typename T>
Tensor<T> LTVE<T>::cauchy_mixed(RCP<GlobalResidual<T>> global) {
  int const pressure_idx = 1;
  T const p = global->scalar_x(pressure_idx);
  Tensor<T> const I = minitensor::eye<T>(this->m_num_dims);
  return this->dev_cauchy(global) - p * I;
}


template class LTVE<double>;
template class LTVE<FADT>;

}
