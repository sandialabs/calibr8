#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include "control.hpp"
#include "defines.hpp"
#include "fad.hpp"
#include "global_residual.hpp"
#include "hypo_barlat.hpp"
#include "hypo_kinematics.hpp"
#include "macros.hpp"
#include "material_params.hpp"
#include "static_tensor.hpp"
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
  p.sublist("line search");
  p.sublist("cylindrical coordinate system points");
  p.set<std::string>("MLEP file", "");
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

  m_cyl_origin = cylindrical_cs_origin;
  m_cartesian_lab_to_mat_rotation <<
      cylindrical_cs_x_dir[0], cylindrical_cs_x_dir[1], cylindrical_cs_x_dir[2],
      cylindrical_cs_y_dir[0], cylindrical_cs_y_dir[1], cylindrical_cs_y_dir[2],
      cylindrical_cs_z_dir[0], cylindrical_cs_z_dir[1], cylindrical_cs_z_dir[2];
}

template <typename T>
void HypoBarlat<T>::read_mlep_data(std::string const& filename) {
  std::ifstream file(filename);
  std::string line;

  while (std::getline(file, line)) {
    std::istringstream lineStream(line);
    std::string xStr, yStr;

    std::getline(lineStream, xStr, ',');
    std::getline(lineStream, yStr, ',');

    m_mlep_x.push_back(std::stod(xStr));
    m_mlep_y.push_back(std::stod(yStr));
  }
}

template <typename T>
T HypoBarlat<T>::evaluate_mlep_hardening(T const& alpha) {
  double const alpha_val = val(alpha);
  T H = 0.;
  if (alpha_val <= m_mlep_x.front()) {
    H = m_mlep_y.front();
  }
  if (alpha_val >= m_mlep_x.back()) {
    H = m_mlep_y.back();
  }

  for (size_t i = 0; i < m_mlep_x.size() - 1; ++i) {
    if (alpha_val >= m_mlep_x[i] && alpha_val <= m_mlep_x[i + 1]) {
      T const t = (alpha - m_mlep_x[i]) / (m_mlep_x[i + 1] - m_mlep_x[i]);
      H = m_mlep_y[i] + t * (m_mlep_y[i + 1] - m_mlep_y[i]);
    }
  }

  return H;
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
  m_ls_params = read_line_search_params(this->m_params_list.sublist("line search"));
  m_ls_params.tag = "(local) ";

  if (inputs.isSublist("cylindrical coordinate system points")) {
    compute_cartesian_lab_to_mat_rotation();
  }
  if (inputs.isParameter("MLEP file")) {
    auto const mlep_file = inputs.get<std::string>("MLEP file");
    read_mlep_data(mlep_file);
    m_use_mlep = true;
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

    // Global coordinates of current material point
    auto const& x_global = global->pt_global_coords();   // Eigen::Vector3d-like

    // --- KEY FIX: translate so cylindrical origin is treated correctly ---
    // cylindrical coordinates must be computed from (x - origin), not from x
    auto const x_rel = x_global - m_cyl_origin;

    // Rotate into the local Cartesian frame aligned with the cylindrical system
    auto const x_loc = m_cartesian_lab_to_mat_rotation * x_rel;

    double const x = x_loc(0);
    double const y = x_loc(1);
    // double const rho = std::sqrt(x*x + y*y);  // only needed if you use it later
    double const theta = std::atan2(y, x);

    // Rows of rotation matrix are the local basis vectors expressed in lab coords
    auto const& e_x = m_cartesian_lab_to_mat_rotation.row(0);
    auto const& e_y = m_cartesian_lab_to_mat_rotation.row(1);
    auto const& e_z = m_cartesian_lab_to_mat_rotation.row(2);

    // Cylindrical basis at this point, expressed in lab coordinates
    auto const e_rho   = std::cos(theta) * e_x + std::sin(theta) * e_y;
    auto const e_theta = -std::sin(theta) * e_x + std::cos(theta) * e_y;
    auto const& e_zeta = e_z;

    // Assemble Q: lab -> material (cylindrical) rotation
    m_Q(0, 0) = e_rho(0);   m_Q(0, 1) = e_rho(1);   m_Q(0, 2) = e_rho(2);
    m_Q(1, 0) = e_theta(0); m_Q(1, 1) = e_theta(1); m_Q(1, 2) = e_theta(2);
    m_Q(2, 0) = e_zeta(0);  m_Q(2, 1) = e_zeta(1);  m_Q(2, 2) = e_zeta(2);
  }
}

template <typename T>
Tensor<T> HypoBarlat<T>::eval_d(RCP<GlobalResidual<T>> global) {
  if (m_kinematics_cached) return m_d;
  compute_Q(global);
  Tensor<T> const d_unrotated = compute_unrotated_rate_of_deformation(
      global->F(), global->F_prev(), global->R());
  return m_Q * d_unrotated * transpose(m_Q);
}

template <>
int HypoBarlat<double>::solve_nonlinear(RCP<GlobalResidual<double>>) {
  return 0;
}

template <>
int HypoBarlat<FADT>::solve_nonlinear(RCP<GlobalResidual<FADT>> global) {

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
  double C_norm_0 = 1.;
  bool converged = false;

  while ((iter <= m_max_iters) && (!converged)) {

    if (iter == 1) {
      path = this->evaluate(global);
    } else {
      this->evaluate(global, true, path);
    }

    double const C_norm = this->norm_residual();
    double C_norm_prev;
    if (iter == 1) {
      C_norm_0 = C_norm;
      C_norm_prev = 10. * C_norm;
    } else {
      C_norm_prev = C_norm;
    }
    if (C_norm_prev < C_norm) {
      print("(local) Newton iter %d: RESIDUAL INCREASE!!!", iter);
      print("C_norm_prev = %e, C_norm = %e", C_norm_prev, C_norm);
    }

    double const C_norm_rel = C_norm / C_norm_0;
    if ((C_norm_rel < m_rel_tol) || (C_norm < m_abs_tol)) {
      converged = true;
      break;
    }

    EMatrix const J = this->eigen_jacobian(this->m_num_dofs);
    EVector const R = this->eigen_residual();
    EVector const dxi = J.fullPivLu().solve(-R);

    this->add_to_sym_tensor_xi(0, dxi);
    this->add_to_scalar_xi(1, dxi);

    {
      double const C_0 = C_norm;
      double const psi_0 = 0.5 * C_0 * C_0;
      double const dpsi_0 = -2. * psi_0;

      double alpha_applied = 1.;   // the full Newton step was applied above
      auto eval = [&](double alpha, double& phi, double& slope) -> bool {
        double const alpha_diff = alpha - alpha_applied;
        alpha_applied = alpha;
        this->add_to_sym_tensor_xi(0, alpha_diff * dxi);
        this->add_to_scalar_xi(1, alpha_diff * dxi);
        path = this->evaluate(global, true, path);
        double const C_alpha = this->norm_residual();
        phi = 0.5 * C_alpha * C_alpha;
        EVector const C = this->eigen_residual();
        EMatrix const J = this->eigen_jacobian(this->m_num_dofs);
        slope = C.dot(J * dxi);          // phi'(alpha) = C . (J dxi)
        return true;
      };

      double const alpha = line_search(m_ls_params, psi_0, dpsi_0, eval);
      // Move the local state to the accepted step.
      double const alpha_diff = alpha - alpha_applied;
      this->add_to_sym_tensor_xi(0, alpha_diff * dxi);
      this->add_to_scalar_xi(1, alpha_diff * dxi);
    }

    iter++;

  }

  m_kinematics_cached = false;

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

  minitensor::Vector<T, 9> const sp_barlat_params = collect_barlat_params(
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

  minitensor::Vector<T, 9> const dp_barlat_params = collect_barlat_params(
      dp_01, dp_02, dp_10, dp_12, dp_20, dp_21, dp_33, dp_44, dp_55
  );

  minitensor::Tensor<T, 3> const TC_old = to_static<3>(this->sym_tensor_xi_prev(0));
  T const alpha_old = this->scalar_xi_prev(1);

  minitensor::Tensor<T, 3> const TC = to_static<3>(this->sym_tensor_xi(0));
  T const alpha = this->scalar_xi(1);

  T phi = 0.;
  BarlatEigenDecomp<T> decomp;
  evaluate_barlat_phi(TC, sp_barlat_params, dp_barlat_params, a, phi, decomp);

  T const scale_factor = 2. * mu;

  T flow_stress;
  if (m_use_mlep) {
    flow_stress = evaluate_mlep_hardening(alpha);
  } else {
    flow_stress = Y + K * alpha + S * (1. - std::exp(-D * alpha));
  }
  T const f = (phi - flow_stress) / scale_factor;

  minitensor::Tensor<T, 3> R_TC;
  T R_alpha;

  minitensor::Tensor<T, 3> const I = minitensor::eye<T, 3>();
  minitensor::Tensor<T, 3> const d = to_static<3>(eval_d(global));
  R_TC = (TC - TC_old - lambda * trace(d) * I - 2. * mu * d) / scale_factor;


  if (!force_path) {
    // plastic step
    if (f > m_abs_tol || std::abs(f) < m_abs_tol) {
      T const dgam = alpha - alpha_old;
      minitensor::Tensor<T, 3> const n =
          evaluate_barlat_normal(decomp, phi, sp_barlat_params, dp_barlat_params, a);
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
      minitensor::Tensor<T, 3> const n =
          evaluate_barlat_normal(decomp, phi, sp_barlat_params, dp_barlat_params, a);
      // scale_factor in R_TC removes the (2. * mu) multiplier
      R_TC += dgam * n;
      R_alpha = f;
    }
    // elastic step
    else {
      R_alpha = alpha - alpha_old;
    }
  }

  this->set_sym_tensor_R(0, to_dynamic(R_TC));
  this->set_scalar_R(1, R_alpha);

  return path;

}

template <typename T>
Tensor<T> HypoBarlat<T>::rotated_cauchy(RCP<GlobalResidual<T>> global) {
  Tensor<T> const TC = this->sym_tensor_xi(0);
  Tensor<T> const& R = global->R();
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
