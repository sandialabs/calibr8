#include <iomanip>
#include <fstream>
#include <Eigen/Dense>
#include <PCU.h>
#include "control.hpp"
#include "defines.hpp"
#include "fad.hpp"
#include "global_residual.hpp"
#include "hybrid_hyper_J2_plane_stress.hpp"
#include "macros.hpp"
#include "material_params.hpp"
#include "yield_functions.hpp"

namespace calibr8 {

static ParameterList get_valid_local_residual_params() {
  ParameterList p;
  p.set<std::string>("type", "hyper_J2_plane_stress");
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
  return p;
}


template <typename T>
HybridHyperJ2PlaneStress<T>::HybridHyperJ2PlaneStress(ParameterList const& inputs, int ndims) {

  this->m_params_list = inputs;
  this->m_params_list.validateParameters(get_valid_local_residual_params(), 0);

  int const num_residuals = 4;

  this->m_num_residuals = num_residuals;
  this->m_num_eqs.resize(num_residuals);
  this->m_var_types.resize(num_residuals);
  this->m_resid_names.resize(num_residuals);

  this->m_resid_names[0] = "zeta";
  this->m_var_types[0] = SYM_TENSOR;
  this->m_num_eqs[0] = get_num_eqs(SYM_TENSOR, ndims);

  this->m_resid_names[1] = "Ie";
  this->m_var_types[1] = SCALAR;
  this->m_num_eqs[1] = get_num_eqs(SCALAR, ndims);

  this->m_resid_names[2] = "lambda_z";
  this->m_var_types[2] = SCALAR;
  this->m_num_eqs[2] = get_num_eqs(SCALAR, ndims);
  this->m_z_stretch_idx = 2;

  this->m_resid_names[3] = "alpha";
  this->m_var_types[3] = SCALAR;
  this->m_num_eqs[3] = get_num_eqs(SCALAR, ndims);

  m_max_iters = inputs.get<int>("nonlinear max iters");
  m_abs_tol = inputs.get<double>("nonlinear absolute tol");
  m_rel_tol = inputs.get<double>("nonlinear relative tol");

}

template <typename T>
void HybridHyperJ2PlaneStress<T>::init_params() {

  int const num_params = 3;
  this->m_params.resize(num_params);
  this->m_param_names.resize(num_params);

  this->m_param_names.resize(num_params);
  this->m_param_names[0] = "E";
  this->m_param_names[1] = "nu";
  this->m_param_names[2] = "Y";

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
  }

  this->m_active_indices.resize(1);
  this->m_active_indices[0].resize(1);
  this->m_active_indices[0][0] = 0;

  ParameterList& embedded_model_params =
      this->m_params_list.sublist("embedded model", true);
  const char* activation = embedded_model_params.get<std::string>("activation function").c_str();
  Array1D<int> const topology = embedded_model_params.get<Teuchos::Array<int>>("topology").toVector();
  m_nn_input_scale = embedded_model_params.get<double>("input scale");
  m_nn_output_scale = embedded_model_params.get<double>("output scale");
  bool write_nn_params = embedded_model_params.get<bool>("write parameters", false);
  bool read_nn_params = embedded_model_params.get<bool>("read parameters", false);
  ALWAYS_ASSERT((write_nn_params + read_nn_params) < 2);
  bool positive_weights = true;
  m_neural_network = rcp(new ML::FFNN<T>(activation, topology, positive_weights));
  this->m_num_dfad_params = m_neural_network->get_num_params();
  if (write_nn_params && PCU_Comm_Self() == 0) {
    auto& nn_params = m_neural_network->get_params();
    std::string const nn_params_filename = "nn_params.out";
    std::ofstream nn_params_file;
    nn_params_file.open(nn_params_filename);
    nn_params_file << std::scientific << std::setprecision(17);
    for (int i = 0; i < this->m_num_dfad_params; ++i) {
      nn_params_file << val(nn_params(i)) << "\n";
    }
    nn_params_file.close();
  }
  if (read_nn_params) {
    Eigen::Matrix<T, Eigen::Dynamic, 1> nn_params(this->m_num_dfad_params);
    std::string const nn_params_filename = "nn_params.in";
    std::ifstream nn_params_file(nn_params_filename);
    std::string line;
    int i = 0;
    while (getline(nn_params_file, line)) {
      nn_params(i) = std::stod(line);
      ++i;
    }
    m_neural_network->set_params(nn_params);
  }
}

template <typename T>
void HybridHyperJ2PlaneStress<T>::set_embedded_params(EVector const& nn_values) {
  int const num_embedded_params = this->m_num_dfad_params;
  Eigen::Matrix<T, Eigen::Dynamic, 1> nn_params(num_embedded_params);
  for (int i = 0; i < num_embedded_params; ++i) {
    nn_params(i) = nn_values(i);
  }
  m_neural_network->set_params(nn_params);
}

template <typename T>
EVector HybridHyperJ2PlaneStress<T>::get_embedded_params() const {
  int const num_embedded_params = this->m_num_dfad_params;
  EVector nn_values(num_embedded_params);
  auto const& nn_params = m_neural_network->get_params();
  for (int i = 0; i < num_embedded_params; ++i) {
    nn_values(i) = val(nn_params(i));
  }
  return nn_values;
}

template <typename T>
HybridHyperJ2PlaneStress<T>::~HybridHyperJ2PlaneStress() {
}

template <typename T>
void HybridHyperJ2PlaneStress<T>::init_variables_impl() {

  int const ndims = this->m_num_dims;
  int const zeta_idx = 0;
  int const Ie_idx = 1;
  int const lambda_z_idx = 2;
  int const alpha_idx = 3;

  Tensor<T> const zeta = minitensor::zero<T>(ndims);
  T const Ie = 1.0;
  T const lambda_z = 1.0;
  T const alpha = 0.0;

  this->set_sym_tensor_xi(zeta_idx, zeta);
  this->set_scalar_xi(Ie_idx, Ie);
  this->set_scalar_xi(lambda_z_idx, lambda_z);
  this->set_scalar_xi(alpha_idx, alpha);
}

template <typename T>
T HybridHyperJ2PlaneStress<T>::nn_hardening(T alpha) {
  Eigen::Matrix<T, Eigen::Dynamic, 1> input_vec(1);
  Eigen::Matrix<T, Eigen::Dynamic, 1> zero_vec(1);
  input_vec(0) = m_nn_input_scale * alpha;
  zero_vec(0) = 0. * alpha;

  return m_nn_output_scale * (m_neural_network->evaluate(input_vec)[0]
      - m_neural_network->evaluate(zero_vec)[0]);
}

template <typename T>
void eval_be_bar_plane_stress(
    RCP<GlobalResidual<T>> global,
    Tensor<T> const& zeta_2D,
    T const& Ie,
    T const& lambda_z_prev,
    T const& lambda_z,
    T& J_2D,
    Tensor<T>& be_bar) {

  Tensor<T> const I_2D = minitensor::eye<T>(2);
  Tensor<T> const I = minitensor::eye<T>(3);
  Tensor<T> const grad_u = global->grad_vector_x(0);
  Tensor<T> const grad_u_prev = global->grad_vector_x_prev(0);
  Tensor<T> const F_2D = grad_u + I_2D;
  J_2D = minitensor::det(F_2D);
  Tensor<T> const F_prev_2D = grad_u_prev + I_2D;
  Tensor<T> F_3D = insert_2D_tensor_into_3D(F_2D);
  Tensor<T> F_prev_3D = insert_2D_tensor_into_3D(F_prev_2D);
  F_3D(2, 2) = lambda_z;
  F_prev_3D(2, 2) = lambda_z_prev;
  Tensor<T> const rF = F_3D * minitensor::inverse(F_prev_3D);
  T const det_rF = minitensor::det(rF);
  T const det_rF_13 = cbrt(det_rF);
  Tensor<T> const rF_bar = rF / det_rF_13;
  Tensor<T> const rF_barT = minitensor::transpose(rF_bar);
  Tensor<T> zeta_3D = insert_2D_tensor_into_3D(zeta_2D);
  zeta_3D(2, 2) = -trace(zeta_2D);
  be_bar = rF_bar * (zeta_3D + Ie * I) * rF_barT;
}

template <>
int HybridHyperJ2PlaneStress<double>::solve_nonlinear(RCP<GlobalResidual<double>>) {
  return 0;
}

template <>
int HybridHyperJ2PlaneStress<FADT>::solve_nonlinear(RCP<GlobalResidual<FADT>> global) {

  int path;

  // pick an initial guess for the local variables
  {
    Tensor<FADT> const zeta_old = this->sym_tensor_xi_prev(0);
    FADT const Ie_old = this->scalar_xi_prev(1);
    FADT const lambda_z_old = this->scalar_xi_prev(2);
    FADT const alpha_old = this->scalar_xi_prev(3);

    int const ndims = m_num_dims;
    FADT J_2D;
    Tensor<FADT> be_bar_trial;
    FADT be_bar_zz_trial;
    FADT const lambda_z = this->scalar_xi(2);
    eval_be_bar_plane_stress(global, zeta_old, Ie_old, lambda_z_old, lambda_z,
        J_2D, be_bar_trial);
    FADT const Ie_trial = minitensor::trace(be_bar_trial) / 3.;
    Tensor<FADT> const I = minitensor::eye<FADT>(ndims);
    Tensor<FADT> const be_bar_trial_2D = extract_2D_tensor_from_3D(be_bar_trial);
    Tensor<FADT> const zeta_trial = be_bar_trial_2D - Ie_trial * I;
    FADT const alpha_trial = alpha_old;
    this->set_sym_tensor_xi(0, zeta_trial);
    this->set_scalar_xi(1, Ie_trial);
    this->set_scalar_xi(3, alpha_trial);
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
    this->add_to_scalar_xi(3, dxi);

    iter++;

  }

  if ((iter > m_max_iters) && (!converged)) {
    std::cout << "HybridHyperJ2PlaneStress:solve_nonlinear failed in "  << iter << " iterations\n";
    return -1;
  }

  return path;

}

template <>
int HybridHyperJ2PlaneStress<DFADT>::solve_nonlinear(RCP<GlobalResidual<DFADT>>) {
  return 0;
}

template <typename T>
int HybridHyperJ2PlaneStress<T>::evaluate(
    RCP<GlobalResidual<T>> global,
    bool force_path,
    int path_in) {

  int path = ELASTIC;
  int const ndims = this->m_num_dims;
  double const sqrt_23 = std::sqrt(2./3.);
  double const sqrt_32 = std::sqrt(3./2.);

  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const Y = this->m_params[2];
  T const mu = compute_mu(E, nu);
  T const kappa = compute_kappa(E, nu);

  Tensor<T> const zeta_old = this->sym_tensor_xi_prev(0);
  T const Ie_old = this->scalar_xi_prev(1);
  T const lambda_z_old = this->scalar_xi_prev(2);
  T const alpha_old = this->scalar_xi_prev(3);

  Tensor<T> const zeta = this->sym_tensor_xi(0);
  T const Ie = this->scalar_xi(1);
  T const lambda_z = this->scalar_xi(2);
  T const alpha = this->scalar_xi(3);

  Tensor<T> const I = minitensor::eye<T>(3);
  T J_2D;

  Tensor<T> be_bar_trial;
  eval_be_bar_plane_stress(global, zeta_old, Ie_old, lambda_z_old, lambda_z,
      J_2D, be_bar_trial);
  T const Ie_trial = minitensor::trace(be_bar_trial) / 3.;
  Tensor<T> const zeta_trial_3D = be_bar_trial - Ie_trial * I;
  Tensor<T> const zeta_trial_2D = extract_2D_tensor_from_3D(zeta_trial_3D);

  Tensor<T> zeta_3D = insert_2D_tensor_into_3D(zeta);
  T const zeta_zz = -trace(zeta);
  zeta_3D(2, 2) = zeta_zz;
  Tensor<T> const be_bar = zeta_3D + Ie * I;

  Tensor<T> const s = mu * zeta_3D;
  T const s_mag = minitensor::norm(s);
  T const sigma_yield = Y + nn_hardening(alpha);
  T const f = (s_mag - sqrt_23 * sigma_yield) / val(mu);

  Tensor<T> R_zeta;
  T R_Ie;
  T R_lambda_z;
  T R_alpha;

  T const mat_factor = kappa / (2. * mu);
  R_lambda_z = lambda_z - std::sqrt((1. - zeta_zz / mat_factor)
      / std::pow(J_2D, 2));

  if (!force_path) {
    // plastic step
    if (f > m_abs_tol || std::abs(f) < m_abs_tol) {
      Tensor<T> const n_2D = mu * zeta / s_mag;
      T const dgam = sqrt_32 * (alpha - alpha_old);
      R_zeta = zeta - zeta_trial_2D + 2. * dgam * Ie * n_2D;
      R_Ie = det(be_bar) - 1.;
      R_alpha = f;
      path = PLASTIC;
    }
    // elastic step
    else {
      R_zeta = zeta - zeta_trial_2D;
      R_Ie = Ie - Ie_trial;
      R_alpha = alpha - alpha_old;
      path = ELASTIC;
    }
  }

  // force the path
  else {
    path = path_in;
    // plastic step
    if (path == PLASTIC) {
      Tensor<T> const n_2D = mu * zeta / s_mag;
      T const dgam = sqrt_32 * (alpha - alpha_old);
      R_zeta = zeta - zeta_trial_2D + 2. * dgam * Ie * n_2D;
      R_Ie = det(be_bar) - 1.;
      R_alpha = f;
    }
    // elastic step
    else {
      R_zeta = zeta - zeta_trial_2D;
      R_Ie = Ie - Ie_trial;
      R_alpha = alpha - alpha_old;
    }
  }

  this->set_sym_tensor_R(0, R_zeta);
  this->set_scalar_R(1, R_Ie);
  this->set_scalar_R(2, R_lambda_z);
  this->set_scalar_R(3, R_alpha);

  return path;

}

template <typename T>
Tensor<T> HybridHyperJ2PlaneStress<T>::cauchy(RCP<GlobalResidual<T>> global) {
  int const ndims = global->num_dims();
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const mu = compute_mu(E, nu);
  T const kappa = compute_kappa(E, nu);
  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const grad_u = global->grad_vector_x(0);
  Tensor<T> const F = grad_u + I;
  T const lambda_z = this->scalar_xi(this->m_z_stretch_idx);
  T const J = minitensor::det(F) * lambda_z;
  Tensor<T> const zeta = this->sym_tensor_xi(0);
  return mu * zeta / J + kappa / 2. * (J - 1. / J) * I;
}

template <typename T>
Tensor<T> HybridHyperJ2PlaneStress<T>::dev_cauchy(RCP<GlobalResidual<T>> global) {
  int const ndims = global->num_dims();
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const mu = compute_mu(E, nu);
  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const grad_u = global->grad_vector_x(0);
  Tensor<T> const F = grad_u + I;
  T const lambda_z = this->scalar_xi(this->m_z_stretch_idx);
  T const J = minitensor::det(F) * lambda_z;
  Tensor<T> const zeta = this->sym_tensor_xi(0);
  return mu * zeta / J;
}

template <typename T>
T HybridHyperJ2PlaneStress<T>::hydro_cauchy(RCP<GlobalResidual<T>> global) {
  int const ndims = global->num_dims();
  T const E = this->m_params[0];
  T const nu = this->m_params[1];
  T const kappa = compute_kappa(E, nu);
  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const grad_u = global->grad_vector_x(0);
  Tensor<T> const F = grad_u + I;
  T const lambda_z = this->scalar_xi(this->m_z_stretch_idx);
  T const J = minitensor::det(F) * lambda_z;
  return kappa / 2. * (J - 1. / J);
}

template <typename T>
T HybridHyperJ2PlaneStress<T>::pressure_scale_factor() { return 0.; }

template class HybridHyperJ2PlaneStress<double>;
template class HybridHyperJ2PlaneStress<FADT>;
template class HybridHyperJ2PlaneStress<DFADT>;

}
