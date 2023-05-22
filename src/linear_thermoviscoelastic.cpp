#include <fstream>
#include <iomanip>
#include <Eigen/Dense>
#include "control.hpp"
#include "defines.hpp"
#include "fad.hpp"
#include "global_residual.hpp"
#include "linear_thermoviscoelastic.hpp"
#include "material_params.hpp"

// read in prony files to populate tau/weight for volume and shear
// compute J_3 and shift factor
// compute rate form of linear elastic problem
// add in viscoelastic parts
// compare with sierra simple seal

namespace calibr8 {

using minitensor::det;
using minitensor::dev;
using minitensor::inverse;
using minitensor::trace;
using minitensor::transpose;

static ParameterList get_valid_local_residual_params() {
  ParameterList p;
  p.set<std::string>("type", "linear_thermoviscoelastic");
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
  p.set<double>("E", 0.);
  p.set<double>("nu", 0.);
  p.set<double>("cte", 0.);
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

template <typename T>
LTVE<T>::LTVE(ParameterList const& inputs, int ndims) {

  this->m_params_list = inputs;
  this->m_params_list.validateParameters(get_valid_local_residual_params(), 0);

  ParameterList const& prony_params = inputs.sublist("prony files");
  this->read_prony_series(prony_params);
  this->compute_temperature(inputs);

  int const num_residuals = 1;

  this->m_num_residuals = num_residuals;
  this->m_num_eqs.resize(num_residuals);
  this->m_var_types.resize(num_residuals);
  this->m_resid_names.resize(num_residuals);

  this->m_resid_names[0] = "cauchy";
  this->m_var_types[0] = SYM_TENSOR;
  this->m_num_eqs[0] = get_num_eqs(SYM_TENSOR, ndims);
}

template <typename T>
LTVE<T>::~LTVE() {
}

template <typename T>
void LTVE<T>::init_params() {
  int const num_params = 3;
  this->m_params.resize(num_params);
  this->m_param_names.resize(num_params);
  this->m_param_names[0] = "E";
  this->m_param_names[1] = "nu";
  this->m_param_names[2] = "cte";
  int const num_elem_sets = this->m_elem_set_names.size();
  resize(this->m_param_values, num_elem_sets, num_params);
  ParameterList& all_material_params = this->m_params_list.sublist("materials", true);
  for (int es = 0; es < num_elem_sets; ++es) {
    std::string const& elem_set_name = this->m_elem_set_names[es];
    ParameterList& material_params = all_material_params.sublist(elem_set_name, true);
    material_params.validateParameters(get_valid_material_params(), 0);
    this->m_param_values[es][0] = material_params.get<double>("E");
    this->m_param_values[es][1] = material_params.get<double>("nu");
    this->m_param_values[es][2] = material_params.get<double>("cte");
  }
  this->m_active_indices.resize(1);
  this->m_active_indices[0].resize(1);
  this->m_active_indices[0][0] = 0;

  this->compute_shift_factors();
}

template <typename T>
void LTVE<T>::init_variables_impl() {
  int const ndims = this->m_num_dims;
  int const cauchy_idx = 0;
  Tensor<T> const cauchy = minitensor::zero<T>(ndims);
  this->set_sym_tensor_xi(cauchy_idx, cauchy);
}

template <>
int LTVE<double>::solve_nonlinear(RCP<GlobalResidual<double>>) {
  return 0;
}

template <>
int LTVE<FADT>::solve_nonlinear(RCP<GlobalResidual<FADT>> global) {

  int path;

  // pick an initial guess for the local variables
  {
    int const ndims = global->num_dims();
    FADT const E = this->m_params[0];
    FADT const nu = this->m_params[1];
    FADT const cte = this->m_params[2];
    FADT const mu = compute_mu(E, nu);
    FADT const kappa = compute_kappa(E, nu);
    Tensor<FADT> const I = minitensor::eye<FADT>(ndims);
    Tensor<FADT> const grad_u = global->grad_vector_x(0);
    Tensor<FADT> const grad_u_T = transpose(grad_u);
    Tensor<FADT> const eps = 0.5 * (grad_u + grad_u_T);
    Tensor<FADT> const cauchy = kappa * (trace(eps) - 3. * cte * m_delta_temp)
      * I + 2. * mu * dev(eps);
    this->set_sym_tensor_xi(0, cauchy);
  }

  path = this->evaluate(global);

  EMatrix const J = this->eigen_jacobian(this->m_num_dofs);
  EVector const R = this->eigen_residual();
  EVector const dxi = J.fullPivLu().solve(-R);

  this->add_to_sym_tensor_xi(0, dxi);

  return path;

}

template <typename T>
int LTVE<T>::evaluate(
    RCP<GlobalResidual<T>> global,
    bool force_path,
    int path_in) {

  // always elastic
  int const path = 0;

  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const cte = this->m_params[2];
  T const mu = compute_mu(E, nu);
  T const kappa = compute_kappa(E, nu);

  Tensor<T> const cauchy = this->sym_tensor_xi(0);

  int const ndims = this->m_num_dims;
  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const grad_u = global->grad_vector_x(0);
  Tensor<T> const grad_u_T = transpose(grad_u);
  Tensor<T> const eps = 0.5 * (grad_u + grad_u_T);
  Tensor<T> R_cauchy = cauchy - kappa * (trace(eps) - 3. * cte * m_delta_temp)
    * I - 2. * mu * dev(eps);

  this->set_sym_tensor_R(0, R_cauchy);

  return path;
}

template <typename T>
Tensor<T> LTVE<T>::dev_cauchy(RCP<GlobalResidual<T>> global) {
  Tensor<T> const cauchy = this->sym_tensor_xi(0);
  int const ndims = this->m_num_dims;
  Tensor<T> const I = minitensor::eye<T>(ndims);
  return cauchy - this->hydro_cauchy(global) * I;
}


template <typename T>
T LTVE<T>::hydro_cauchy(RCP<GlobalResidual<T>> global) {
  Tensor<T> const cauchy = this->sym_tensor_xi(0);
  int const ndims = this->m_num_dims;
  Tensor<T> const I = minitensor::eye<T>(ndims);
  if (ndims == 3) {
    return trace(cauchy) / 3.;
  } else {
    T const nu = this->m_params[1];
    return (1. + nu) * trace(cauchy) / 3.;
  }
}

template <typename T>
T LTVE<T>::pressure_scale_factor() {
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const kappa = compute_kappa(E, nu);
  return kappa;
}

template <typename T>
Tensor<T> LTVE<T>::cauchy(RCP<GlobalResidual<T>> global) {
  int const pressure_idx = 1;
  T const p = global->scalar_x(pressure_idx);
  Tensor<T> const I = minitensor::eye<T>(this->m_num_dims);
  return this->dev_cauchy(global) - p * I;
}

template class LTVE<double>;
template class LTVE<FADT>;

}
