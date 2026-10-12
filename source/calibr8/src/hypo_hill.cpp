#include <Eigen/Dense>
#include "control.hpp"
#include "defines.hpp"
#include "fad.hpp"
#include "global_residual.hpp"
#include "hypo_hill.hpp"
#include "hypo_kinematics.hpp"
#include "macros.hpp"
#include "material_params.hpp"
#include "yield_functions.hpp"

namespace calibr8 {

using minitensor::dev;
using minitensor::trace;
using minitensor::transpose;

static ParameterList get_valid_local_residual_params() {
  ParameterList p;
  p.set<std::string>("type", "hypo_hill");
  p.set<int>("nonlinear max iters", 0);
  p.set<double>("nonlinear absolute tol", 0.);
  p.set<double>("nonlinear relative tol", 0.);
  p.sublist("materials");
  p.sublist("cylindrical coordinate system points");
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
void HypoHill<T>::compute_cartesian_lab_to_mat_rotation() {
  m_compute_cylindrical_transform = true;
  ParameterList& inputs = this->m_params_list.sublist("cylindrical coordinate system points", true);
  auto const input_cylindrical_cs_origin = inputs.get<Teuchos::Array<double>>("origin").toVector();
  auto const input_cylindrical_cs_point_on_z_axis = inputs.get<Teuchos::Array<double>>("point on z axis").toVector();
  auto const input_cylindrical_cs_point_on_x_axis = inputs.get<Teuchos::Array<double>>("point on x axis").toVector();
  ALWAYS_ASSERT(input_cylindrical_cs_origin.size() == 3);
  ALWAYS_ASSERT(input_cylindrical_cs_point_on_z_axis.size() == 3);
  ALWAYS_ASSERT(input_cylindrical_cs_point_on_x_axis.size() == 3);

  Eigen::Vector3d const cylindrical_cs_origin = Eigen::Vector3d::Map(
      input_cylindrical_cs_origin.data(), 3
  );
  Eigen::Vector3d const cylindrical_cs_point_on_z_axis = Eigen::Vector3d::Map(
      input_cylindrical_cs_point_on_z_axis.data(), 3
  );
  Eigen::Vector3d const cylindrical_cs_point_on_x_axis = Eigen::Vector3d::Map(
      input_cylindrical_cs_point_on_x_axis.data(), 3
  );

  auto const cylindrical_cs_x_dir = (cylindrical_cs_point_on_x_axis - cylindrical_cs_origin).normalized();
  auto const cylindrical_cs_z_dir = (cylindrical_cs_point_on_z_axis - cylindrical_cs_origin).normalized();
  double const xz_dot = cylindrical_cs_x_dir.dot(cylindrical_cs_z_dir);
  if (std::abs(xz_dot) > 1.0e-12)
    fail("cylindrical coordinate system: 'point on x axis' and 'point on z axis' "
         "must give orthogonal directions from 'origin', but x.z = %.3e", xz_dot);
  auto const cylindrical_cs_y_dir = cylindrical_cs_z_dir.cross(cylindrical_cs_x_dir);

  m_cartesian_lab_to_mat_rotation <<
      cylindrical_cs_x_dir[0], cylindrical_cs_x_dir[1], cylindrical_cs_x_dir[2],
      cylindrical_cs_y_dir[0], cylindrical_cs_y_dir[1], cylindrical_cs_y_dir[2],
      cylindrical_cs_z_dir[0], cylindrical_cs_z_dir[1], cylindrical_cs_z_dir[2];

  m_cyl_origin = cylindrical_cs_origin;
}

template <typename T>
void HypoHill<T>::compute_Q(RCP<GlobalResidual<T>> global) {
  if (m_compute_cylindrical_transform) {

    // global coordinates of current material point
    auto const& x_global = global->pt_global_coords();

    // cylindrical coordinates must be computed from (x - origin)
    auto const x_rel = x_global - m_cyl_origin;

    // rotate into the local Cartesian frame aligned with the cylindrical system
    auto const x_local = m_cartesian_lab_to_mat_rotation * x_rel;
    double const x = x_local(0);
    double const y = x_local(1);
    double const theta = std::atan2(y, x);

    // rows of rotation matrix are the local basis vectors expressed in lab coords
    auto const& e_x = m_cartesian_lab_to_mat_rotation.row(0);
    auto const& e_y = m_cartesian_lab_to_mat_rotation.row(1);
    auto const& e_z = m_cartesian_lab_to_mat_rotation.row(2);

    // cylindrical basis at this point, expressed in lab coordinates
    auto const e_rho   = std::cos(theta) * e_x + std::sin(theta) * e_y;
    auto const e_theta = -std::sin(theta) * e_x + std::cos(theta) * e_y;
    auto const& e_zeta = e_z;

    // assemble Q: lab -> material (cylindrical) rotation
    m_Q(0, 0) = e_rho(0);
    m_Q(0, 1) = e_rho(1);
    m_Q(0, 2) = e_rho(2);
    m_Q(1, 0) = e_theta(0);
    m_Q(1, 1) = e_theta(1);
    m_Q(1, 2) = e_theta(2);
    m_Q(2, 0) = e_zeta(0);
    m_Q(2, 1) = e_zeta(1);
    m_Q(2, 2) = e_zeta(2);
  }
}

template <typename T>
HypoHill<T>::HypoHill(ParameterList const& inputs, int ndims) {

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

  if (inputs.isSublist("cylindrical coordinate system points")) {
    compute_cartesian_lab_to_mat_rotation();
  }

}

template <typename T>
HypoHill<T>::~HypoHill() {
}

template <typename T>
void HypoHill<T>::init_params() {

  // 2 elastic + Y + 6 HypoHill + S + D
  int const num_params = 11;
  this->m_params.resize(num_params);
  this->m_param_names.resize(num_params);

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
void HypoHill<T>::init_variables_impl() {

  int const ndims = this->m_num_dims;
  int const TC_idx = 0;
  int const alpha_idx = 1;

  Tensor<T> const TC = minitensor::zero<T>(ndims);
  T const alpha = 0.0;

  this->set_sym_tensor_xi(TC_idx, TC);
  this->set_scalar_xi(alpha_idx, alpha);

}

template <typename T>
Tensor<T> HypoHill<T>::eval_d(RCP<GlobalResidual<T>> global) {
  if (m_kinematics_cached) return m_d;
  compute_Q(global);
  Tensor<T> const d_unrotated =
      compute_unrotated_rate_of_deformation_increment(
          global->F(), global->F_prev());
  return m_Q * d_unrotated * transpose(m_Q);
}


template <>
int HypoHill<double>::solve_nonlinear(RCP<GlobalResidual<double>>) {
  return 0;
}

template <>
int HypoHill<FADT>::solve_nonlinear(RCP<GlobalResidual<FADT>> global) {

  int path;

  m_d = eval_d(global);
  m_kinematics_cached = true;

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
    // std::cout << "HypoHill:solve_nonlinear failed in "  << iter << " iterations\n";
    m_kinematics_cached = false;
    return -1;
  }

  m_kinematics_cached = false;
  return path;

}

template <>
int HypoHill<DFADT>::solve_nonlinear(RCP<GlobalResidual<DFADT>>) {
  return 0;
}

template <typename T>
int HypoHill<T>::evaluate(
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
  T const lambda = compute_lambda(E, nu);
  T const mu = compute_mu(E, nu);

  Vector<T> const hill_params = compute_hill_params(R00, R11, R22,
      R01, R02, R12);

  Tensor<T> const TC_old = this->sym_tensor_xi_prev(0);
  T const alpha_old = this->scalar_xi_prev(1);

  Tensor<T> const TC = this->sym_tensor_xi(0);
  T const alpha = this->scalar_xi(1);

  T const hill = compute_hill_value(TC, hill_params);
  T const sigma_yield = Y + S * (1. - std::exp(-D * alpha));
  T const f = (hill - sigma_yield) / val(mu);

  Tensor<T> R_TC;
  T R_alpha;

  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const d = eval_d(global);
  R_TC = TC - TC_old - lambda * trace(d) * I - 2. * mu * d;
  R_TC /= val(mu);

  if (!force_path) {
    // plastic step
    if (f > m_abs_tol || std::abs(f) < m_abs_tol) {
      T const dgam = alpha - alpha_old;
      Tensor<T> const n = compute_hill_normal(TC, hill_params, hill);
      R_TC += (2. * mu * dgam * n) / val(mu);
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
      Tensor<T> const n = compute_hill_normal(TC, hill_params, hill);
      R_TC += (2. * mu * dgam * n) / val(mu);
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
Tensor<T> HypoHill<T>::rotated_cauchy(RCP<GlobalResidual<T>> global) {
  Tensor<T> const TC = this->sym_tensor_xi(0);
  Tensor<T> const& R = global->R();
  compute_Q(global);
  Tensor<T> const RC = R * transpose(m_Q) * TC * m_Q * transpose(R);
  return RC;
}

template <typename T>
Tensor<T> HypoHill<T>::cauchy(RCP<GlobalResidual<T>> global) {
  int const pressure_idx = 1;
  T const p = global->scalar_x(pressure_idx);
  int const ndims = this->m_num_dims;
  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const dev_RC = this->dev_cauchy(global);
  Tensor<T> const sigma = dev_RC - p * I;
  return sigma;
}

template <typename T>
Tensor<T> HypoHill<T>::dev_cauchy(RCP<GlobalResidual<T>> global) {
  Tensor<T> const RC = this->rotated_cauchy(global);
  return dev(RC);
}

template <typename T>
T HypoHill<T>::hydro_cauchy(RCP<GlobalResidual<T>> global) {
  Tensor<T> const RC = this->rotated_cauchy(global);
  return trace(RC) / 3.;
}


template <typename T>
T HypoHill<T>::pressure_scale_factor() {
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const kappa = compute_kappa(E, nu);
  return kappa;
}


template class HypoHill<double>;
template class HypoHill<FADT>;
template class HypoHill<DFADT>;

}
