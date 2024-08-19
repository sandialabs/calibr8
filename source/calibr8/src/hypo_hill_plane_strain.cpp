#include <Eigen/Dense>
#include "control.hpp"
#include "defines.hpp"
#include "fad.hpp"
#include "global_residual.hpp"
#include "hypo_hill_plane_strain.hpp"
#include "material_params.hpp"
#include "yield_functions.hpp"

namespace calibr8 {

using minitensor::dev;
using minitensor::inverse;
using minitensor::trace;
using minitensor::transpose;

static ParameterList get_valid_local_residual_params() {
  ParameterList p;
  p.set<std::string>("type", "hypo_hill_plane_strain");
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
  p.set<double>("S", 0.);
  p.set<double>("D", 0.);
  p.set<double>("R00", 0.);
  p.set<double>("R11", 0.);
  p.set<double>("R22", 0.);
  p.set<double>("R01", 0.);
  return p;
}

template <typename T>
HypoHillPlaneStrain<T>::HypoHillPlaneStrain(ParameterList const& inputs, int ndims) {

  this->m_params_list = inputs;
  this->m_params_list.validateParameters(get_valid_local_residual_params(), 0);

  int const num_residuals = 3;

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

  // out-of-plane stress
  this->m_resid_names[2] = "TC_zz";
  this->m_var_types[2] = SCALAR;
  this->m_num_eqs[2] = get_num_eqs(SCALAR, ndims);
  this->m_z_stress_idx = 2;

  m_max_iters = inputs.get<int>("nonlinear max iters");
  m_abs_tol = inputs.get<double>("nonlinear absolute tol");
  m_rel_tol = inputs.get<double>("nonlinear relative tol");

}

template <typename T>
HypoHillPlaneStrain<T>::~HypoHillPlaneStrain() {
}

template <typename T>
void HypoHillPlaneStrain<T>::init_params() {

  int const num_params = 9;
  this->m_params.resize(num_params);
  this->m_param_names.resize(num_params);

  this->m_param_names.resize(num_params);
  this->m_param_names[0] = "E";
  this->m_param_names[1] = "nu";
  this->m_param_names[2] = "Y";
  this->m_param_names[3] = "S";
  this->m_param_names[4] = "D";
  this->m_param_names[5] = "R00";
  this->m_param_names[6] = "R11";
  this->m_param_names[7] = "R22";
  this->m_param_names[8] = "R01";

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
    this->m_param_values[es][5] = material_params.get<double>("R00");
    this->m_param_values[es][6] = material_params.get<double>("R11");
    this->m_param_values[es][7] = material_params.get<double>("R22");
    this->m_param_values[es][8] = material_params.get<double>("R01");
  }

  this->m_active_indices.resize(1);
  this->m_active_indices[0].resize(1);
  this->m_active_indices[0][0] = 0;
}

template <typename T>
void HypoHillPlaneStrain<T>::init_variables_impl() {

  int const ndims = this->m_num_dims;
  int const TC_idx = 0;
  int const alpha_idx = 1;
  int const TC_zz_idx = 2;

  Tensor<T> const TC = minitensor::zero<T>(ndims);
  T const alpha = 0.;
  T const TC_zz = 0.;

  this->set_sym_tensor_xi(TC_idx, TC);
  this->set_scalar_xi(alpha_idx, alpha);
  this->set_scalar_xi(TC_zz_idx, TC_zz);

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
int HypoHillPlaneStrain<double>::solve_nonlinear(RCP<GlobalResidual<double>>) {
  return 0;
}

template <>
int HypoHillPlaneStrain<FADT>::solve_nonlinear(RCP<GlobalResidual<FADT>> global) {

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
    FADT const TC_zz_old = this->scalar_xi_prev(2);
    Tensor<FADT> const d = eval_d(global);
    Tensor<FADT> const TC = TC_old + lambda * trace(d) * I + 2. * mu * d;
    FADT const alpha = alpha_old;
    FADT const TC_zz = TC_zz_old + lambda * trace(d);
    this->set_sym_tensor_xi(0, TC);
    this->set_scalar_xi(1, alpha);
    this->set_scalar_xi(2, TC_zz);
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
    std::cout << "HypoHillPlaneStrain:solve_nonlinear failed in "  << iter << " iterations\n";
    return -1;
  }

  return path;

}

template <>
int HypoHillPlaneStrain<DFADT>::solve_nonlinear(RCP<GlobalResidual<DFADT>>) {
  return 0;
}

template <typename T>
int HypoHillPlaneStrain<T>::evaluate(
    RCP<GlobalResidual<T>> global,
    bool force_path,
    int path_in) {

  int path = ELASTIC;
  int const ndims = this->m_num_dims;

  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const Y = this->m_params[2];
  T const S = this->m_params[3];
  T const D = this->m_params[4];
  T const R00 = this->m_params[5];
  T const R11 = this->m_params[6];
  T const R22 = this->m_params[7];
  T const R01 = this->m_params[8];
  T const lambda = compute_lambda(E, nu);
  T const mu = compute_mu(E, nu);

  Tensor<T> const TC_old = this->sym_tensor_xi_prev(0);
  T const alpha_old = this->scalar_xi_prev(1);
  T const TC_zz_old = this->scalar_xi_prev(2);

  Tensor<T> const TC = this->sym_tensor_xi(0);
  T const alpha = this->scalar_xi(1);
  T const TC_zz = this->scalar_xi(2);

  Tensor<T> TC_3D = insert_2D_tensor_into_3D(TC);
  TC_3D(2, 2) = TC_zz;

  // out-of-plane shear coefficients
  T const R02 = 1.;
  T const R12 = 1.;
  Vector<T> const hill_params = compute_hill_params(R00, R11, R22,
      R01, R02, R12);

  T const phi = compute_hill_value(TC_3D, hill_params);
  T const sigma_yield = Y + S * (1. - std::exp(-D * alpha));
  T const f = (phi - sigma_yield) / val(mu);

  Tensor<T> R_TC;
  T R_alpha;
  T R_TC_zz;

  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const d = eval_d(global);
  R_TC = TC - TC_old - lambda * trace(d) * I - 2. * mu * d;
  R_TC_zz = TC_zz - TC_zz_old - lambda * trace(d);

  if (!force_path) {
    // plastic step
    if (f > m_abs_tol || std::abs(f) < m_abs_tol) {
      Tensor<T> n_3D = compute_hill_normal(TC_3D, hill_params, phi);
      Tensor<T> const n_2D = extract_2D_tensor_from_3D(n_3D);
      T const dgam = alpha - alpha_old;
      Tensor<T> const dp_2D = dgam * n_2D;
      T const dp_zz = -trace(dp_2D);
      R_TC += 2. * mu * dp_2D;
      R_alpha = f;
      R_TC_zz += 2. * mu * dp_zz;
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
      Tensor<T> n_3D = compute_hill_normal(TC_3D, hill_params, phi);
      Tensor<T> const n_2D = extract_2D_tensor_from_3D(n_3D);
      T const dgam = alpha - alpha_old;
      Tensor<T> const dp_2D = dgam * n_2D;
      T const dp_zz = -trace(dp_2D);
      R_TC += 2. * mu * dp_2D;
      R_alpha = f;
      R_TC_zz += 2. * mu * dp_zz;
      path = PLASTIC;
    }
    // elastic step
    else {
      R_alpha = alpha - alpha_old;
    }
  }

  this->set_sym_tensor_R(0, R_TC);
  this->set_scalar_R(1, R_alpha);
  this->set_scalar_R(2, R_TC_zz);

  return path;

}

template <typename T>
Tensor<T> HypoHillPlaneStrain<T>::rotated_cauchy(RCP<GlobalResidual<T>> global) {
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
Tensor<T> HypoHillPlaneStrain<T>::cauchy(RCP<GlobalResidual<T>> global) {
  int const pressure_idx = 1;
  T const p = global->scalar_x(pressure_idx);
  int const ndims = this->m_num_dims;
  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const dev_RC = this->dev_cauchy(global);
  Tensor<T> const sigma = dev_RC - p * I;
  return sigma;
}

template <typename T>
Tensor<T> HypoHillPlaneStrain<T>::dev_cauchy(RCP<GlobalResidual<T>> global) {
  Tensor<T> const RC = this->rotated_cauchy(global);
  int const ndims = this->m_num_dims;
  Tensor<T> const I = minitensor::eye<T>(ndims);
  T const hydro_cauchy = this->hydro_cauchy(global);
  return RC - hydro_cauchy * I;
}

template <typename T>
T HypoHillPlaneStrain<T>::hydro_cauchy(RCP<GlobalResidual<T>> global) {
  Tensor<T> const RC = this->rotated_cauchy(global);
  T const TC_zz = this->scalar_xi(this->m_z_stress_idx);
  return (trace(RC) + TC_zz) / 3.;
}

template <typename T>
T HypoHillPlaneStrain<T>::pressure_scale_factor() {
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const kappa = compute_kappa(E, nu);
  return kappa;
}

template class HypoHillPlaneStrain<double>;
template class HypoHillPlaneStrain<FADT>;
template class HypoHillPlaneStrain<DFADT>;

}
