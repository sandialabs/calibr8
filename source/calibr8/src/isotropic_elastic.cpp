#include <Eigen/Dense>
#include "control.hpp"
#include "defines.hpp"
#include "fad.hpp"
#include "global_residual.hpp"
#include "isotropic_elastic.hpp"
#include "material_params.hpp"


namespace calibr8 {

using minitensor::det;
using minitensor::dev;
using minitensor::inverse;
using minitensor::trace;
using minitensor::transpose;

static ParameterList get_valid_local_residual_params() {
  ParameterList p;
  p.set<std::string>("type", "elastic");
  p.set<bool>("mixed formulation", true);
  p.sublist("materials");
  return p;
}

static ParameterList get_valid_material_params() {
  ParameterList p;
  p.set<double>("E", 0.);
  p.set<double>("nu", 0.);
  return p;
}

template <typename T>
IsotropicElastic<T>::IsotropicElastic(ParameterList const& inputs, int ndims) {

  this->m_params_list = inputs;
  this->m_params_list.validateParameters(get_valid_local_residual_params(), 0);

  bool is_mixed = (this->m_params_list).template get<bool>("mixed formulation", true);
  if (is_mixed) {
    m_mode = MIXED;
  } else {
    m_mode = DISPLACEMENT;
  }

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
IsotropicElastic<T>::~IsotropicElastic() {
}

template <typename T>
void IsotropicElastic<T>::init_params() {
  int const num_params = 2;
  this->m_params.resize(num_params);
  this->m_param_names.resize(num_params);
  this->m_param_names[0] = "E";
  this->m_param_names[1] = "nu";
  int const num_elem_sets = this->m_elem_set_names.size();
  resize(this->m_param_values, num_elem_sets, num_params);
  ParameterList& all_material_params = this->m_params_list.sublist("materials", true);
  for (int es = 0; es < num_elem_sets; ++es) {
    std::string const& elem_set_name = this->m_elem_set_names[es];
    ParameterList& material_params = all_material_params.sublist(elem_set_name, true);
    material_params.validateParameters(get_valid_material_params(), 0);
    this->m_param_values[es][0] = material_params.get<double>("E");
    this->m_param_values[es][1] = material_params.get<double>("nu");
  }
  this->m_active_indices.resize(1);
  this->m_active_indices[0].resize(1);
  this->m_active_indices[0][0] = 0;
}

template <typename T>
void IsotropicElastic<T>::init_variables_impl() {
  int const ndims = this->m_num_dims;
  int const cauchy_idx = 0;
  Tensor<T> const cauchy = minitensor::zero<T>(ndims);
  this->set_sym_tensor_xi(cauchy_idx, cauchy);
}

template <>
int IsotropicElastic<double>::solve_nonlinear(RCP<GlobalResidual<double>>) {
  return 0;
}

template <>
int IsotropicElastic<FADT>::solve_nonlinear(RCP<GlobalResidual<FADT>> global) {

  int path;

  // pick an initial guess for the local variables
  {
    int const ndims = global->num_dims();
    FADT const E = this->m_params[0];
    FADT const nu = this->m_params[1];
    FADT const mu = compute_mu(E, nu);
    FADT const lambda = compute_lambda(E, nu);
    Tensor<FADT> const I = minitensor::eye<FADT>(ndims);
    Tensor<FADT> const grad_u = global->grad_vector_x(0);
    Tensor<FADT> const grad_u_T = transpose(grad_u);
    Tensor<FADT> const eps = 0.5 * (grad_u + grad_u_T);
    Tensor<FADT> const cauchy = lambda * trace(eps) * I + 2. * mu * eps;
    this->set_sym_tensor_xi(0, cauchy);
  }

  path = this->evaluate(global);

  EMatrix const J = this->eigen_jacobian(this->m_num_dofs);
  EVector const R = this->eigen_residual();
  EVector const dxi = J.fullPivLu().solve(-R);

  this->add_to_sym_tensor_xi(0, dxi);

  return path;

}

template <>
int IsotropicElastic<DFADT>::solve_nonlinear(RCP<GlobalResidual<DFADT>>) {
  return 0;
}

template <typename T>
int IsotropicElastic<T>::evaluate(
    RCP<GlobalResidual<T>> global,
    bool force_path,
    int path_in) {

  // always elastic
  int const path = 0;

  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const mu = compute_mu(E, nu);
  T const lambda = compute_lambda(E, nu);

  Tensor<T> const cauchy = this->sym_tensor_xi(0);

  int const ndims = this->m_num_dims;
  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const grad_u = global->grad_vector_x(0);
  Tensor<T> const grad_u_T = transpose(grad_u);
  Tensor<T> const eps = 0.5 * (grad_u + grad_u_T);
  Tensor<T> R_cauchy = cauchy - lambda * trace(eps) * I - 2. * mu * eps;

  this->set_sym_tensor_R(0, R_cauchy);

  return path;
}

template <typename T>
Tensor<T> IsotropicElastic<T>::cauchy(RCP<GlobalResidual<T>> global) {
  if (m_mode == MIXED) {
    return cauchy_mixed(global);
  } else {
    return this->sym_tensor_xi(0);
  }
}

template <typename T>
Tensor<T> IsotropicElastic<T>::dev_cauchy(RCP<GlobalResidual<T>> global) {
  Tensor<T> const cauchy = this->sym_tensor_xi(0);
  int const ndims = this->m_num_dims;
  Tensor<T> const I = minitensor::eye<T>(ndims);
  return cauchy - this->hydro_cauchy(global) * I;
}


template <typename T>
T IsotropicElastic<T>::hydro_cauchy(RCP<GlobalResidual<T>> global) {
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
T IsotropicElastic<T>::pressure_scale_factor() {
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const kappa = compute_kappa(E, nu);
  return kappa;
}

template <typename T>
Tensor<T> IsotropicElastic<T>::cauchy_mixed(RCP<GlobalResidual<T>> global) {
  int const pressure_idx = 1;
  T const p = global->scalar_x(pressure_idx);
  Tensor<T> const I = minitensor::eye<T>(this->m_num_dims);
  return this->dev_cauchy(global) - p * I;
}

template class IsotropicElastic<double>;
template class IsotropicElastic<FADT>;
template class IsotropicElastic<DFADT>;

}
