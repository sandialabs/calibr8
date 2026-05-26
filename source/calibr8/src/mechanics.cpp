#include <apf.h>
#include <apfMesh.h>
#include "control.hpp"
#include "defines.hpp"
#include "local_residual.hpp"
#include "material_params.hpp"
#include "mechanics.hpp"

namespace calibr8 {

using minitensor::det;
using minitensor::inverse;
using minitensor::transpose;

template <typename T>
Mechanics<T>::Mechanics(ParameterList const& params, int ndims) {

  auto p = params;
  bool is_mixed = p.get<bool>("mixed formulation", true);
  if (is_mixed) {
    m_mode = MIXED;
  } else {
    m_mode = DISPLACEMENT;
  }

  int const num_residuals = (m_mode == MIXED) ? 2 : 1;

  this->m_num_residuals = num_residuals;
  this->m_num_eqs.resize(num_residuals);
  this->m_var_types.resize(num_residuals);
  this->m_resid_names.resize(num_residuals);

  this->m_resid_names[0] = "u";
  this->m_var_types[0] = VECTOR;
  this->m_num_eqs[0] = get_num_eqs(VECTOR, ndims);

  if (m_mode == MIXED) {
    this->m_resid_names[1] = "p";
    this->m_var_types[1] = SCALAR;
    this->m_num_eqs[1] = get_num_eqs(SCALAR, ndims);

    int const num_ip_sets = 2;
    this->m_ip_sets.resize(num_ip_sets);
    // quadrature order for each integration point set
    this->m_ip_sets[0] = 1;
    this->m_ip_sets[1] = 2;
    m_stabilization_multiplier = p.get<double>("stabilization multiplier", 1.);

  } else {
    int const num_ip_sets = 1;
    this->m_ip_sets.resize(num_ip_sets);
    // quadrature order for each integration point set
    this->m_ip_sets[0] = 1;
  }
}

template <typename T>
Mechanics<T>::~Mechanics() {
}

template <typename T>
void Mechanics<T>::compute_kinematics() {
  int const ndims = this->m_num_dims;
  int const momentum_idx = 0;

  if (this->m_F.get_dimension() != ndims) {
    this->m_F = Tensor<T>(ndims);
    this->m_F_prev = Tensor<T>(ndims);
    this->m_cof_F = Tensor<T>(ndims);
  }

  for (int k = 0; k < ndims; ++k) {
    for (int l = 0; l < ndims; ++l) {
      this->m_F(k, l) = this->m_grad_x[momentum_idx][k][l];
      this->m_F_prev(k, l) = this->m_grad_x_prev[momentum_idx][k][l];
    }
    this->m_F(k, k) += T(1.0);
    this->m_F_prev(k, k) += T(1.0);
  }

  this->m_det_F = det(this->m_F);

  Tensor<T> const& F = this->m_F;
  Tensor<T>& C = this->m_cof_F;
  if (ndims == 3) {
    C(0,0) =  F(1,1)*F(2,2) - F(1,2)*F(2,1);
    C(0,1) = -F(1,0)*F(2,2) + F(1,2)*F(2,0);
    C(0,2) =  F(1,0)*F(2,1) - F(1,1)*F(2,0);
    C(1,0) = -F(0,1)*F(2,2) + F(0,2)*F(2,1);
    C(1,1) =  F(0,0)*F(2,2) - F(0,2)*F(2,0);
    C(1,2) = -F(0,0)*F(2,1) + F(0,1)*F(2,0);
    C(2,0) =  F(0,1)*F(1,2) - F(0,2)*F(1,1);
    C(2,1) = -F(0,0)*F(1,2) + F(0,2)*F(1,0);
    C(2,2) =  F(0,0)*F(1,1) - F(0,1)*F(1,0);
  } else {
    C(0,0) =  F(1,1);
    C(0,1) = -F(1,0);
    C(1,0) = -F(0,1);
    C(1,1) =  F(0,0);
  }
}

static double get_size(apf::Mesh* mesh, apf::MeshElement* me) {
  double h = 0.;
  apf::Downward edges;
  apf::MeshEntity* ent = apf::getMeshEntity(me);
  int const nedges = mesh->getDownward(ent, 1, edges);
  for (int e = 0; e < nedges; ++e) {
    double const l = apf::measure(mesh, edges[e]);
    h += l * l;
  }
  return std::sqrt(h / nedges);
}

template <typename T>
void Mechanics<T>::evaluate_displacement(
    RCP<LocalResidual<T>> local,
    apf::Vector3 const&,
    double w,
    double dv,
    int ip_set) {

  // gather information from this class
  int const ndims = this->m_num_dims;
  int const nnodes = this->m_num_nodes;
  int const momentum_idx = 0;

  // compute stress measures
  RCP<GlobalResidual<T>> global = rcp(this, false);
  Tensor<T> stress = local->cauchy(global);

  // PK1 = J*sigma*F^{-T} = sigma*cof(F)
  if (local->is_finite_deformation()) stress = stress * this->cof_F();

  // compute the balance of linear momentum residual
  for (int n = 0; n < nnodes; ++n) {
    for (int i = 0; i < ndims; ++i) {
      for (int j = 0; j < ndims; ++j) {
        double const dbasis_dx = this->grad_weight(momentum_idx, n, i, j);
        this->R_nodal(momentum_idx, n, i) +=
          stress(i, j) * dbasis_dx * w * dv;
      }
    }
  }
}

template <typename T>
void Mechanics<T>::evaluate_mixed(
    RCP<LocalResidual<T>> local,
    apf::Vector3 const&,
    double w,
    double dv,
    int ip_set) {

  // gather information from this class
  int const ndims = this->m_num_dims;
  int const nnodes = this->m_num_nodes;
  int const pressure_idx = 1;

  // gather material properties
  T const E = local->params(0);
  T const nu = local->params(1);
  T const mu = compute_mu(E, nu);

  T const p = this->scalar_x(pressure_idx);
  T pressure_scale_factor = local->pressure_scale_factor();

  // coupled ip set (lowest quadrature order)
  if (ip_set == 0) {

    // gather variables from this residual quantities
    Vector<T> const grad_p = this->grad_scalar_x(pressure_idx);

    Tensor<T> const I = minitensor::eye<T>(ndims);

    // compute the constant part of the pressure residual
    RCP<GlobalResidual<T>> global = rcp(this, false);
    T hydro_cauchy = local->hydro_cauchy(global);

    for (int n = 0; n < nnodes; ++n) {
      int const eq = 0;
      double const basis = this->weight(pressure_idx, n, eq);
      this->R_nodal(pressure_idx, n, eq) -=
        hydro_cauchy / pressure_scale_factor * basis * w * dv;
    }

    // compute the pressure stabilization residual
    double h = 0.;
    if (this->m_stabilization_h == CURRENT) {
      h = get_size(this->m_mesh, this->m_mesh_elem);
    } else if (this->m_stabilization_h == BASE) {
      apf::Field* f_h = this->m_mesh->findField("h");
      apf::MeshEntity* ent = apf::getMeshEntity(this->m_mesh_elem);
      h = apf::getScalar(f_h, ent, 0);
    }

    T const tau = m_stabilization_multiplier * 0.5 * h * h / mu;
    Tensor<T> stab_matrix = tau * I;
    if (local->is_finite_deformation()) {
      Tensor<T> const& cof_F = this->cof_F();
      stab_matrix = stab_matrix * (transpose(cof_F) * cof_F) / this->det_F();
    }

    for (int n = 0; n < nnodes; ++n) {
      for (int i = 0; i < ndims; ++i) {
        for (int j = 0; j < ndims; ++j) {
          int const eq = 0;
          double const dbasis_dx = this->grad_stab_weight(pressure_idx, n, eq, i);
          this->R_nodal(pressure_idx, n, eq) -=
            stab_matrix(i, j) *  grad_p(j) * dbasis_dx * w * dv;
        }
      }
    }

  } else if (ip_set == 1) {

    // compute the linear part of the pressure residual
    for (int n = 0; n < nnodes; ++n) {
      int const eq = 0;
      double const basis = this->weight(pressure_idx, n, eq);
      this->R_nodal(pressure_idx, n, eq) -=
        p / pressure_scale_factor * basis * w * dv;
    }
  } else {
    fail("unimplemented ip set\n");
  }
}

template <typename T>
void Mechanics<T>::evaluate(
    RCP<LocalResidual<T>> local,
    apf::Vector3 const& iota,
    double w,
    double dv,
    int ip_set) {

  if (ip_set == 0) evaluate_displacement(local, iota, w, dv, ip_set);
  if (m_mode == MIXED) evaluate_mixed(local, iota, w, dv, ip_set);

}

template class Mechanics<double>;
template class Mechanics<FADT>;
template class Mechanics<DFADT>;

}
