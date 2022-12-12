#include <apf.h>
#include <apfMesh.h>
#include "control.hpp"
#include "defines.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "material_params.hpp"
#include "mechanics_disp.hpp"

namespace calibr8 {

using minitensor::det;
using minitensor::inverse;
using minitensor::transpose;

template <typename T>
MechanicsDisp<T>::MechanicsDisp(ParameterList const&, int ndims) {

  int const num_residuals = 1;

  this->m_num_residuals = num_residuals;
  this->m_num_eqs.resize(num_residuals);
  this->m_var_types.resize(num_residuals);
  this->m_resid_names.resize(num_residuals);

  this->m_resid_names[0] = "u";
  this->m_var_types[0] = VECTOR;
  this->m_num_eqs[0] = get_num_eqs(VECTOR, ndims);

  int const num_ip_sets = 1;
  this->m_ip_sets.resize(num_ip_sets);
  // quadrature order for each integration point set
  this->m_ip_sets[0] = 1;

}

template <typename T>
MechanicsDisp<T>::~MechanicsDisp() {
}

template <typename T>
void MechanicsDisp<T>::evaluate(
    RCP<LocalResidual<T>> local,
    apf::Vector3 const&,
    double w,
    double dv,
    int ip_set) {

  // gather information from this class
  int const ndims = this->m_num_dims;
  int const nnodes = this->m_num_nodes;
  int const momentum_idx = 0;

  // coupled ip set (lowest quadrature order)
  ALWAYS_ASSERT(ip_set == 0);

  // gather variables from this residual quantities
  Tensor<T> const grad_u = this->grad_vector_x(momentum_idx);

  // compute kinematic quantities
  Tensor<T> const I = minitensor::eye<T>(ndims);
  Tensor<T> const F = grad_u + I;
  Tensor<T> const F_inv = inverse(F);
  Tensor<T> const F_invT = transpose(F_inv);
  T const J = det(F);

  // compute stress measures
  RCP<GlobalResidual<T>> global = rcp(this, false);
  // Cauchy for these models is dev_cauchy
  Tensor<T> stress = local->dev_cauchy(global);

  if (local->is_finite_deformation()) {
    stress = J * stress * F_invT;
    if (local->is_plane_stress()) {
      const int lambda_z_idx = 2;
      T const lambda_z = local->scalar_xi(lambda_z_idx);
      stress *= lambda_z;
    }
  }

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

template class MechanicsDisp<double>;
template class MechanicsDisp<FADT>;

}
