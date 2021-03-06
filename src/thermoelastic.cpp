#include <apf.h>
#include <apfMesh.h>
#include "control.hpp"
#include "defines.hpp"
#include "local_residual.hpp"
#include "material_params.hpp"
#include "thermoelastic.hpp"

namespace calibr8 {

using minitensor::transpose;

template <typename T>
Thermoelastic<T>::Thermoelastic(ParameterList const&, int ndims) {

  int const num_residuals = 2;

  this->m_num_residuals = num_residuals;
  this->m_num_eqs.resize(num_residuals);
  this->m_var_types.resize(num_residuals);
  this->m_resid_names.resize(num_residuals);

  this->m_resid_names[0] = "u";
  this->m_var_types[0] = VECTOR;
  this->m_num_eqs[0] = get_num_eqs(VECTOR, ndims);

  this->m_resid_names[1] = "p";
  this->m_var_types[1] = SCALAR;
  this->m_num_eqs[1] = get_num_eqs(SCALAR, ndims);

  int const num_ip_sets = 2;
  this->m_ip_sets.resize(num_ip_sets);
  // quadrature order for each integration point set
  this->m_ip_sets[0] = 1;
  this->m_ip_sets[1] = 2;

}

template <typename T>
Thermoelastic<T>::~Thermoelastic() {
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
void Thermoelastic<T>::evaluate(
    RCP<LocalResidual<T>> local,
    apf::Vector3 const&,
    double w,
    double dv,
    int ip_set) {

  // gather information from this class
  int const ndims = this->m_num_dims;
  int const nnodes = this->m_num_nodes;
  int const momentum_idx = 0;
  int const pressure_idx = 1;

  // gather material properties
  T const E = local->params(0);
  T const nu = local->params(1);
  T const mu = compute_mu(E, nu);
  T const kappa = compute_kappa(E, nu);
  T const cte = local->params(2);
  T const delta_T = local->params(3);

  // coupled ip set (lowest quadrature order)
  if (ip_set == 0) {

    // gather variables from this residual quantities
    T const p = this->scalar_x(pressure_idx);
    Vector<T> const grad_p = this->grad_scalar_x(pressure_idx);
    Tensor<T> const grad_u = this->grad_vector_x(momentum_idx);

    // compute stress measures
    RCP<GlobalResidual<T>> global = rcp(this, false);
    Tensor<T> stress = local->cauchy(global, p);

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

    Tensor<T> const eps = 0.5 * (grad_u + minitensor::transpose(grad_u));

    // compute the linear part of the pressure residual
    for (int n = 0; n < nnodes; ++n) {
      int const eq  = 0;
      double const basis = this->weight(pressure_idx, n, eq);
      this->R_nodal(pressure_idx, n, eq) -=
        (minitensor::trace(eps) - 3. * cte * delta_T) * basis * w * dv;
    }

    // compute the stabilization to the pressure residual
    double const h = get_size(this->m_mesh, this->m_mesh_elem);
    T const tau = 0.5 * h * h / mu;
    for (int n = 0; n < nnodes; ++n) {
      for (int i = 0; i < ndims; ++i) {
        for (int j = 0; j < ndims; ++j) {
          int const eq = 0;
          double const dbasis_dx = this->grad_stab_weight(pressure_idx, n, eq, j);
          this->R_nodal(pressure_idx, n, eq) -=
            tau * grad_p(i) * dbasis_dx * w * dv;
        }
      }
    }
  }

  else if (ip_set == 1) {
    // gather variables from this residual quantities
    T const p = this->scalar_x(pressure_idx);

    // compute the linear part of the pressure residual
    for (int n = 0; n < nnodes; ++n) {
      int const eq = 0;
      double const basis = this->weight(pressure_idx, n, eq);
      this->R_nodal(pressure_idx, n, eq) -=
        p / kappa * basis * w * dv;
    }
  }

  else {
    fail("unimplemented ip set\n");
  }

}

template class Thermoelastic<double>;
template class Thermoelastic<FADT>;

}
