#include <fstream>
#include <Eigen/Dense>
#include "control.hpp"
#include "defines.hpp"
#include "fad.hpp"
#include "global_residual.hpp"
#include "hybrid_hypo_hill.hpp"
#include "hypo_kinematics.hpp"
#include "ICNN.hpp"
#include "material_params.hpp"
#include "yield_functions.hpp"

namespace calibr8 {

using minitensor::dev;
using minitensor::trace;
using minitensor::transpose;

static ParameterList get_valid_local_residual_params() {
  ParameterList p;
  p.set<std::string>("type", "hybrid_hypo_hill");
  p.set<int>("nonlinear max iters", 0);
  p.set<double>("nonlinear absolute tol", 0.);
  p.set<double>("nonlinear relative tol", 0.);
  p.sublist("materials");
  p.sublist("embedded model");
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
HybridHypoHill<T>::HybridHypoHill(ParameterList const& inputs, int ndims) {

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
}

template <typename T>
HybridHypoHill<T>::~HybridHypoHill() {
}

template <typename T>
void HybridHypoHill<T>::init_params() {

  int const num_params = 11;
  this->m_params.resize(num_params);
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

  ParameterList& embedded_model_params =
      this->m_params_list.sublist("embedded model", true);
  m_activation = embedded_model_params.get<std::string>("activation function");
  m_topology = embedded_model_params.get<Teuchos::Array<int>>("topology").toVector();
  m_nn_input_scale = embedded_model_params.get<double>("input scale");
  m_nn_output_scale = embedded_model_params.get<double>("output scale");
  m_symmetrize = embedded_model_params.get<bool>("symmetrize", true);
  m_icnn = rcp(new ML::FICNN<T>(m_activation.c_str(), m_topology));
  this->m_num_dfad_params = m_icnn->get_params().size();

  bool const read_nn_params =
      embedded_model_params.get<bool>("read parameters", false);
  if (read_nn_params) {
    EVector nn_params(this->m_num_dfad_params);
    std::ifstream nn_params_file("nn_params.in");
    std::string line;
    int i = 0;
    while (getline(nn_params_file, line)) {
      nn_params(i) = std::stod(line);
      ++i;
    }
    set_dfad_params(nn_params);
  }
}

template <typename T>
void HybridHypoHill<T>::set_dfad_params(EVector const& nn_values) {
  int const num_embedded_params = this->m_num_dfad_params;
  Eigen::Matrix<T, Eigen::Dynamic, 1> nn_params(num_embedded_params);
  for (int i = 0; i < num_embedded_params; ++i) {
    nn_params(i) = nn_values(i);
  }
  m_icnn->set_params(nn_params);
}

template <typename T>
EVector HybridHypoHill<T>::get_dfad_params() const {
  int const num_embedded_params = this->m_num_dfad_params;
  EVector nn_values(num_embedded_params);
  auto const& nn_params = m_icnn->get_params();
  for (int i = 0; i < num_embedded_params; ++i) {
    nn_values(i) = val(nn_params(i));
  }
  return nn_values;
}

template <typename T>
void HybridHypoHill<T>::init_variables_impl() {

  int const ndims = this->m_num_dims;
  int const TC_idx = 0;
  int const alpha_idx = 1;

  Tensor<T> const TC = minitensor::zero<T>(ndims);
  T const alpha = 0.0;

  this->set_sym_tensor_xi(TC_idx, TC);
  this->set_scalar_xi(alpha_idx, alpha);
}

template <typename T>
Tensor<T> HybridHypoHill<T>::eval_d(RCP<GlobalResidual<T>> global) {
  if (m_kinematics_cached) return m_d;
  return compute_unrotated_rate_of_deformation_increment(
      global->F(), global->F_prev());
}

template <>
int HybridHypoHill<double>::solve_nonlinear(RCP<GlobalResidual<double>>) {
  return 0;
}

template <>
int HybridHypoHill<FADT>::solve_nonlinear(RCP<GlobalResidual<FADT>> global) {

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

    iter++;
  }

  if ((iter > m_max_iters) && (!converged)) {
    m_kinematics_cached = false;
    return -1;
  }

  m_kinematics_cached = false;
  return path;
}

template <>
int HybridHypoHill<DFADT>::solve_nonlinear(RCP<GlobalResidual<DFADT>>) {
  return 0;
}

template <typename T>
void icnn_value_and_grad(
    ML::FICNN<T>& nn,
    char const* activation,
    std::vector<int> const& topology,
    double in_scale,
    double out_scale,
    bool symmetrize,
    Tensor<T> const& stress,
    T& value,
    Tensor<T>& grad) {

  using RAD = Sacado::Rad::ADvar<T>;

  ML::FICNN<RAD> nn_rad(activation, topology);
  Eigen::Matrix<T, Eigen::Dynamic, 1> const params = nn.get_params();
  Eigen::Matrix<RAD, Eigen::Dynamic, 1> rad_params(params.size());
  for (int i = 0; i < params.size(); ++i) {
    rad_params(i) = RAD(params(i));
  }
  nn_rad.set_params(rad_params);

  Eigen::Matrix<RAD, Eigen::Dynamic, 1> s(6);
  s(0) = RAD(stress(0, 0));
  s(1) = RAD(stress(1, 1));
  s(2) = RAD(stress(2, 2));
  s(3) = RAD(stress(0, 1));
  s(4) = RAD(stress(1, 2));
  s(5) = RAD(stress(2, 0));

  RAD const mean = (s(0) + s(1) + s(2)) / 3.;
  Eigen::Matrix<RAD, Eigen::Dynamic, 1> input(6);
  input(0) = in_scale * (s(0) - mean);
  input(1) = in_scale * (s(1) - mean);
  input(2) = in_scale * (s(2) - mean);
  input(3) = in_scale * s(3);
  input(4) = in_scale * s(4);
  input(5) = in_scale * s(5);

  RAD const phi_plus = nn_rad.evaluate(input)(0);
  RAD phi = phi_plus;
  if (symmetrize) {
    Eigen::Matrix<RAD, Eigen::Dynamic, 1> input_neg(6);
    for (int i = 0; i < 6; ++i) {
      input_neg(i) = -1. * input(i);
    }
    phi = 0.5 * (phi_plus + nn_rad.evaluate(input_neg)(0));
  }
  Sacado::Rad::ADvar<T>::Gradcomp();

  Eigen::Matrix<T, Eigen::Dynamic, 1> zero(6);
  for (int i = 0; i < 6; ++i) {
    zero(i) = T(0.);
  }
  T const out_zero = nn.evaluate(zero)(0);
  value = out_scale * (phi.val() - out_zero);

  grad = minitensor::zero<T>(3);
  grad(0, 0) = out_scale * s(0).adj();
  grad(1, 1) = out_scale * s(1).adj();
  grad(2, 2) = out_scale * s(2).adj();
  grad(0, 1) = grad(1, 0) = 0.5 * out_scale * s(3).adj();
  grad(1, 2) = grad(2, 1) = 0.5 * out_scale * s(4).adj();
  grad(2, 0) = grad(0, 2) = 0.5 * out_scale * s(5).adj();
}

template <typename T>
T HybridHypoHill<T>::effective_stress_and_normal(
    Tensor<T> const& TC,
    Vector<T> const& hill_params,
    T const& Y,
    Tensor<T>& n) {

  int const ndims = this->m_num_dims;

  T const hill_TC = compute_hill_value(TC, hill_params);

  double const stress_tol = 1.0e-12 * val(Y);
  if (val(hill_TC) < stress_tol) {
    n = minitensor::zero<T>(ndims);
    return hill_TC;
  }

  int const max_beta_iters = 50;
  double const beta_rel_tol = 1.0e-12;
  T c = T(0.);
  Tensor<T> grad_c = minitensor::zero<T>(ndims);
  T const beta = solve_effective_stress_scale(
      TC, hill_TC, Y, max_beta_iters, beta_rel_tol,
      [this](Tensor<T> const& u, T& value, Tensor<T>& gradient) {
        icnn_value_and_grad(*m_icnn, m_activation.c_str(), m_topology,
            m_nn_input_scale, m_nn_output_scale, m_symmetrize, u, value,
            gradient);
      },
      c, grad_c);

  T const sigma_eff = hill_TC + c / beta;

  Tensor<T> const grad_phi =
      compute_hill_normal(TC, hill_params, hill_TC) + grad_c;
  T grad_phi_dot_TC = T(0.);
  for (int i = 0; i < ndims; ++i) {
    for (int j = 0; j < ndims; ++j) {
      grad_phi_dot_TC += grad_phi(i, j) * TC(i, j);
    }
  }
  n = (sigma_eff / grad_phi_dot_TC) * grad_phi;

  return sigma_eff;
}

template <typename T>
int HybridHypoHill<T>::evaluate(
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

  Tensor<T> n = minitensor::zero<T>(ndims);
  T const sigma_eff = this->effective_stress_and_normal(TC, hill_params, Y, n);
  T const sigma_yield = Y + S * (1. - std::exp(-D * alpha));
  T const f = (sigma_eff - sigma_yield) / val(mu);

  Tensor<T> R_TC;
  T R_alpha;

  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const d = eval_d(global);
  R_TC = TC - TC_old - lambda * trace(d) * I - 2. * mu * d;
  R_TC /= val(mu);

  if (!force_path) {
    if (f > m_abs_tol || std::abs(f) < m_abs_tol) {
      T const dgam = alpha - alpha_old;
      R_TC += (2. * mu * dgam * n) / val(mu);
      R_alpha = f;
      path = PLASTIC;
    } else {
      R_alpha = alpha - alpha_old;
      path = ELASTIC;
    }
  } else {
    path = path_in;
    if (path == PLASTIC) {
      T const dgam = alpha - alpha_old;
      R_TC += (2. * mu * dgam * n) / val(mu);
      R_alpha = f;
    } else {
      R_alpha = alpha - alpha_old;
    }
  }

  this->set_sym_tensor_R(0, R_TC);
  this->set_scalar_R(1, R_alpha);

  return path;
}

template <typename T>
Tensor<T> HybridHypoHill<T>::rotated_cauchy(RCP<GlobalResidual<T>> global) {
  Tensor<T> const TC = this->sym_tensor_xi(0);
  Tensor<T> const& R = global->R();
  Tensor<T> const RC = R * TC * transpose(R);
  return RC;
}

template <typename T>
Tensor<T> HybridHypoHill<T>::cauchy(RCP<GlobalResidual<T>> global) {
  int const pressure_idx = 1;
  T const p = global->scalar_x(pressure_idx);
  int const ndims = this->m_num_dims;
  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const dev_RC = this->dev_cauchy(global);
  Tensor<T> const sigma = dev_RC - p * I;
  return sigma;
}

template <typename T>
Tensor<T> HybridHypoHill<T>::dev_cauchy(RCP<GlobalResidual<T>> global) {
  Tensor<T> const RC = this->rotated_cauchy(global);
  return dev(RC);
}

template <typename T>
T HybridHypoHill<T>::hydro_cauchy(RCP<GlobalResidual<T>> global) {
  Tensor<T> const RC = this->rotated_cauchy(global);
  return trace(RC) / 3.;
}

template <typename T>
T HybridHypoHill<T>::pressure_scale_factor() {
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const kappa = compute_kappa(E, nu);
  return kappa;
}

template class HybridHypoHill<double>;
template class HybridHypoHill<FADT>;
template class HybridHypoHill<DFADT>;

template void icnn_value_and_grad(
    ML::FICNN<double>&, char const*, std::vector<int> const&, double, double,
    bool, Tensor<double> const&, double&, Tensor<double>&);
template void icnn_value_and_grad(
    ML::FICNN<FADT>&, char const*, std::vector<int> const&, double, double,
    bool, Tensor<FADT> const&, FADT&, Tensor<FADT>&);
template void icnn_value_and_grad(
    ML::FICNN<DFADT>&, char const*, std::vector<int> const&, double, double,
    bool, Tensor<DFADT> const&, DFADT&, Tensor<DFADT>&);

}
