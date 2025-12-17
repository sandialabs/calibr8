#include <Eigen/Dense>
#include "control.hpp"
#include "defines.hpp"
#include "fad.hpp"
#include "global_residual.hpp"
#include "hypo_barlat.hpp"
#include "macros.hpp"
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
  p.set<std::string>("type", "hypo_barlat");
  p.set<int>("nonlinear max iters", 0);
  p.set<double>("nonlinear absolute tol", 0.);
  p.set<double>("nonlinear relative tol", 0.);
  p.sublist("materials");
  p.set<double>("line search beta", 1.0e-4);
  p.set<double>("line search eta", 0.5);
  p.set<int>("line search max evals", 10);
  p.set<bool>("line search print", false);
  p.sublist("cylindrical coordinate system points");
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

  p.set<double>("sp_01", 0.);
  p.set<double>("sp_02", 0.);
  p.set<double>("sp_10", 0.);
  p.set<double>("sp_12", 0.);
  p.set<double>("sp_20", 0.);
  p.set<double>("sp_21", 0.);
  p.set<double>("sp_33", 0.);
  p.set<double>("sp_44", 0.);
  p.set<double>("sp_55", 0.);

  p.set<double>("dp_01", 0.);
  p.set<double>("dp_02", 0.);
  p.set<double>("dp_10", 0.);
  p.set<double>("dp_12", 0.);
  p.set<double>("dp_20", 0.);
  p.set<double>("dp_21", 0.);
  p.set<double>("dp_33", 0.);
  p.set<double>("dp_44", 0.);
  p.set<double>("dp_55", 0.);

  return p;
}

template <typename T>
void HypoBarlat<T>::compute_cartesian_lab_to_mat_rotation() {
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
  auto const cylindrical_cs_y_dir = cylindrical_cs_z_dir.cross(cylindrical_cs_x_dir);

  m_cartesian_lab_to_mat_rotation <<
      cylindrical_cs_x_dir[0], cylindrical_cs_x_dir[1], cylindrical_cs_x_dir[2],
      cylindrical_cs_y_dir[0], cylindrical_cs_y_dir[1], cylindrical_cs_y_dir[2],
      cylindrical_cs_z_dir[0], cylindrical_cs_z_dir[1], cylindrical_cs_z_dir[2];
}


template <typename T>
HypoBarlat<T>::HypoBarlat(ParameterList const& inputs, int ndims) {

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

  if (inputs.isSublist("cylindrical coordinate system points")) {
    compute_cartesian_lab_to_mat_rotation();
  }
}

template <typename T>
HypoBarlat<T>::~HypoBarlat() {
}

template <typename T>
void HypoBarlat<T>::init_params() {

  int const num_params = 25;
  this->m_params.resize(num_params);
  this->m_param_names.resize(num_params);

  this->m_param_names[0] = "E";
  this->m_param_names[1] = "nu";
  this->m_param_names[2] = "Y";
  this->m_param_names[3] = "a";
  this->m_param_names[4] = "K";
  this->m_param_names[5] = "S";
  this->m_param_names[6] = "D";

  this->m_param_names[7] = "sp_01";
  this->m_param_names[8] = "sp_02";
  this->m_param_names[9] = "sp_10";
  this->m_param_names[10] = "sp_12";
  this->m_param_names[11] = "sp_20";
  this->m_param_names[12] = "sp_21";
  this->m_param_names[13] = "sp_33";
  this->m_param_names[14] = "sp_44";
  this->m_param_names[15] = "sp_55";

  this->m_param_names[16] = "dp_01";
  this->m_param_names[17] = "dp_02";
  this->m_param_names[18] = "dp_10";
  this->m_param_names[19] = "dp_12";
  this->m_param_names[20] = "dp_20";
  this->m_param_names[21] = "dp_21";
  this->m_param_names[22] = "dp_33";
  this->m_param_names[23] = "dp_44";
  this->m_param_names[24] = "dp_55";

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

    this->m_param_values[es][7] = material_params.get<double>("sp_01");
    this->m_param_values[es][8] = material_params.get<double>("sp_02");
    this->m_param_values[es][9] = material_params.get<double>("sp_10");
    this->m_param_values[es][10] = material_params.get<double>("sp_12");
    this->m_param_values[es][11] = material_params.get<double>("sp_20");
    this->m_param_values[es][12] = material_params.get<double>("sp_21");
    this->m_param_values[es][13] = material_params.get<double>("sp_33");
    this->m_param_values[es][14] = material_params.get<double>("sp_44");
    this->m_param_values[es][15] = material_params.get<double>("sp_55");

    this->m_param_values[es][16] = material_params.get<double>("dp_01");
    this->m_param_values[es][17] = material_params.get<double>("dp_02");
    this->m_param_values[es][18] = material_params.get<double>("dp_10");
    this->m_param_values[es][19] = material_params.get<double>("dp_12");
    this->m_param_values[es][20] = material_params.get<double>("dp_20");
    this->m_param_values[es][21] = material_params.get<double>("dp_21");
    this->m_param_values[es][22] = material_params.get<double>("dp_33");
    this->m_param_values[es][23] = material_params.get<double>("dp_44");
    this->m_param_values[es][24] = material_params.get<double>("dp_55");
  }

  this->m_active_indices.resize(1);
  this->m_active_indices[0].resize(1);
  this->m_active_indices[0][0] = 0;
}

template <typename T>
void HypoBarlat<T>::init_variables_impl() {

  int const ndims = this->m_num_dims;
  int const TC_idx = 0;
  int const alpha_idx = 1;

  Tensor<T> const TC = minitensor::zero<T>(ndims);
  T const alpha = 0.0;

  this->set_sym_tensor_xi(TC_idx, TC);
  this->set_scalar_xi(alpha_idx, alpha);

}

template <typename T>
void HypoBarlat<T>::compute_Q(RCP<GlobalResidual<T>> global) {
  if (m_compute_cylindrical_transform) {
    auto const& pt_global_coords = global->pt_global_coords();
    auto const cartesian_mat_coords = m_cartesian_lab_to_mat_rotation * pt_global_coords;

    double x = cartesian_mat_coords(0);
    double y = cartesian_mat_coords(1);
    double rho = std::sqrt(x*x + y*y);
    double theta = std::atan2(y, x);

    auto const& e_x = m_cartesian_lab_to_mat_rotation.row(0);
    auto const& e_y = m_cartesian_lab_to_mat_rotation.row(1);
    auto const e_rho = std::cos(theta) * e_x + std::sin(theta) * e_y;
    auto const e_theta = -std::sin(theta) * e_x + std::cos(theta) * e_y;
    auto const& e_zeta = m_cartesian_lab_to_mat_rotation.row(2);

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
Tensor<T> HypoBarlat<T>::eval_d(RCP<GlobalResidual<T>> global) {
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
  compute_Q(global);
  Tensor<T> const d = m_Q * transpose(R) * D * R * transpose(m_Q);
  return d;
}


template <>
int HypoBarlat<double>::solve_nonlinear(RCP<GlobalResidual<double>>) {
  return 0;
}

template <>
int HypoBarlat<FADT>::solve_nonlinear(RCP<GlobalResidual<FADT>> global) {

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
    // double-check this
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
    // std::cout << "HypoBarlat:solve_nonlinear failed in "  << iter << " iterations\n";
    return -1;
  }

  return path;
}

template <>
int HypoBarlat<DFADT>::solve_nonlinear(RCP<GlobalResidual<DFADT>>) {
  return 0;
}

template <typename T>
int HypoBarlat<T>::evaluate(
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

  T const sp_01 = this->m_params[7];
  T const sp_02 = this->m_params[8];
  T const sp_10 = this->m_params[9];
  T const sp_12 = this->m_params[10];
  T const sp_20 = this->m_params[11];
  T const sp_21 = this->m_params[12];
  T const sp_33 = this->m_params[13];
  T const sp_44 = this->m_params[14];
  T const sp_55 = this->m_params[15];

  Vector<T> const sp_barlat_params = collect_barlat_params(
      sp_01, sp_02, sp_10, sp_12, sp_20, sp_21, sp_33, sp_44, sp_55
  );

  T const dp_01 = this->m_params[16];
  T const dp_02 = this->m_params[17];
  T const dp_10 = this->m_params[18];
  T const dp_12 = this->m_params[19];
  T const dp_20 = this->m_params[20];
  T const dp_21 = this->m_params[21];
  T const dp_33 = this->m_params[22];
  T const dp_44 = this->m_params[23];
  T const dp_55 = this->m_params[24];

  Vector<T> const dp_barlat_params = collect_barlat_params(
      dp_01, dp_02, dp_10, dp_12, dp_20, dp_21, dp_33, dp_44, dp_55
  );

  Tensor<T> const TC_old = this->sym_tensor_xi_prev(0);
  T const alpha_old = this->scalar_xi_prev(1);

  Tensor<T> const TC = this->sym_tensor_xi(0);
  T const alpha = this->scalar_xi(1);

  T phi = 0.;
  Tensor<T> n = zero<T>(3);
  evaluate_barlat_phi_and_normal(
      TC, sp_barlat_params, dp_barlat_params, a, phi, n
  );

  T const scale_factor = 2. * mu;

  T const flow_stress = Y + K * alpha + S * (1. - std::exp(-D * alpha));
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
Tensor<T> HypoBarlat<T>::rotated_cauchy(RCP<GlobalResidual<T>> global) {
  int const ndims = this->m_num_dims;
  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const grad_u = global->grad_vector_x(0);
  Tensor<T> const F = grad_u + I;
  Tensor<T> const TC = this->sym_tensor_xi(0);
  Tensor<T> const R = minitensor::polar_rotation(F);
  compute_Q(global);
  Tensor<T> const RC = R * transpose(m_Q) * TC * m_Q * transpose(R);
  return RC;
}

template <typename T>
Tensor<T> HypoBarlat<T>::cauchy(RCP<GlobalResidual<T>> global) {
  int const pressure_idx = 1;
  T const p = global->scalar_x(pressure_idx);
  int const ndims = this->m_num_dims;
  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const dev_RC = this->dev_cauchy(global);
  Tensor<T> const sigma = dev_RC - p * I;
  return sigma;
}

template <typename T>
Tensor<T> HypoBarlat<T>::dev_cauchy(RCP<GlobalResidual<T>> global) {
  Tensor<T> const RC = this->rotated_cauchy(global);
  return dev(RC);
}

template <typename T>
T HypoBarlat<T>::hydro_cauchy(RCP<GlobalResidual<T>> global) {
  Tensor<T> const RC = this->rotated_cauchy(global);
  return trace(RC) / 3.;
}


template <typename T>
T HypoBarlat<T>::pressure_scale_factor() {
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const kappa = compute_kappa(E, nu);
  return kappa;
}


template class HypoBarlat<double>;
template class HypoBarlat<FADT>;
template class HypoBarlat<DFADT>;

}
