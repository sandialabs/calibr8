#include <Eigen/Dense>
#include "control.hpp"
#include "defines.hpp"
#include "elastic.hpp"
#include "fad.hpp"
#include "global_residual.hpp"
#include "material_params.hpp"

namespace calibr8 {

static ParameterList get_valid_local_residual_params() {
  ParameterList p;
  p.set<std::string>("type", "elastic");
  p.sublist("materials");
  return p;
}

static ParameterList get_valid_material_params() {
  ParameterList p;
  p.set<double>("E", 0.);
  p.set<double>("nu", 0.);
  p.set<double>("cte", 0.);
  p.set<double>("delta_T", 0.);
  return p;
}

template <typename T>
Elastic<T>::Elastic(ParameterList const& inputs, int ndims) {
  this->m_params_list = inputs;
  this->m_params_list.validateParameters(get_valid_local_residual_params(), 0);
  int const num_residuals = 1;
  this->m_num_residuals = num_residuals;
  this->m_num_eqs.resize(num_residuals);
  this->m_var_types.resize(num_residuals);
  this->m_resid_names.resize(num_residuals);
  this->m_resid_names[0] = "dummy";
  this->m_var_types[0] = SCALAR;
  this->m_num_eqs[0] = 1;
}

template <typename T>
Elastic<T>::~Elastic() {
}

template <typename T>
void Elastic<T>::init_params() {
  int const num_params = 4;
  this->m_params.resize(num_params);
  this->m_param_names.resize(num_params);
  this->m_param_names[0] = "E";
  this->m_param_names[1] = "nu";
  this->m_param_names[2] = "cte";
  this->m_param_names[3] = "delta_T";
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
    this->m_param_values[es][3] = material_params.get<double>("delta_T");
  }
  this->m_active_indices.resize(1);
  this->m_active_indices[0].resize(1);
  this->m_active_indices[0][0] = 0;
}

template <typename T>
void Elastic<T>::init_variables_impl() {
  this->set_scalar_xi(0, 0.);
}

template <typename T>
int Elastic<T>::solve_nonlinear(RCP<GlobalResidual<T>>) {
  this->set_scalar_xi(0, 0.);
  return 0;
}

template <typename T>
int Elastic<T>::evaluate(
    RCP<GlobalResidual<T>> global,
    bool force_path,
    int path_in) {
  (void)force_path;
  (void)path_in;
  return 0;
}

template <typename T>
Tensor<T> Elastic<T>::cauchy(RCP<GlobalResidual<T>> global) {
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
Tensor<T> Elastic<T>::dev_cauchy(RCP<GlobalResidual<T>> global) {
  int const ndims = global->num_dims();
  Tensor<T> const I = minitensor::eye<T>(ndims);
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const mu = compute_mu(E, nu);
  Tensor<T> const grad_u = global->grad_vector_x(0);
  Tensor<T> const eps = 0.5 * (grad_u + minitensor::transpose(grad_u));
  Tensor<T> const dev_eps = eps - (minitensor::trace(eps) / 3.) * I;
  return 2. * mu * dev_eps;
}

template <typename T>
T Elastic<T>::hydro_cauchy(RCP<GlobalResidual<T>> global) {
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const kappa = compute_kappa(E, nu);
  T const cte = this->m_params[2];
  T const delta_T = this->m_params[3];
  Tensor<T> const grad_u = global->grad_vector_x(0);
  Tensor<T> const eps = 0.5 * (grad_u + minitensor::transpose(grad_u));
  return kappa * trace(eps) - cte * delta_T * E / (1. - 2. * nu);
}

template <typename T>
T Elastic<T>::pressure_scale_factor() {
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const kappa = compute_kappa(E, nu);
  return kappa;
}

template class Elastic<double>;
template class Elastic<FADT>;
template class Elastic<DFADT>;

}
