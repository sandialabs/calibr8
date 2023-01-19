#include "control.hpp"
#include "error_weight.hpp"

namespace calibr8 {

ErrorWeight::ErrorWeight(
    apf::FieldShape* shape,
    int ndim,
    int nresids,
    int nnodes,
    Array1D<int> const& neqs,
    Array1D<apf::Field*> const& z) :
  Weight(shape)
{
  m_z = z;
  m_ndim = ndim;
  m_nresids = nresids;
  m_nnodes = nnodes;
  m_neqs = neqs;
  resize(values, nresids, nnodes, neqs);
  resize(gradients, nresids, nnodes, neqs, ndim);
}

void ErrorWeight::evaluate(apf::MeshElement* me, apf::Vector3 const& iota) {
  double z_scal;
  apf::Vector3 z_scal_grad;
  apf::Vector3 z_vec;
  apf::Matrix3x3 z_vec_grad;
  apf::Matrix3x3 z_vec_gradT;
  apf::getBF(m_shape, me, iota, m_basis);
  apf::getGradBF(m_shape, me, iota, m_grad_basis);
  for (int r = 0; r < m_nresids; ++r) {
    apf::Element* z_elem = apf::createElement(m_z[r], me);
    int const type = apf::getValueType(m_z[r]);
    if (type == apf::SCALAR) {
      z_scal = apf::getScalar(z_elem, iota);
      apf::getGrad(z_elem, iota, z_scal_grad);
      for (int n = 0; n < m_nnodes; ++n) {
        values[r][n][0] = z_scal * m_basis[n];
        for (int i = 0; i < m_ndim; ++i) {
          gradients[r][n][0][i] =
            z_scal_grad[i] * m_basis[n] +
            z_scal * m_grad_basis[n][i];
        }
      }
    } else if (type == apf::VECTOR) {
      apf::getVector(z_elem, iota, z_vec);
      apf::getVectorGrad(z_elem, iota, z_vec_gradT);
      z_vec_grad = apf::transpose(z_vec_gradT);
      for (int n = 0; n < m_nnodes; ++n) {
        for (int i = 0; i < m_ndim; ++i) {
          values[r][n][i] = z_vec[i] * m_basis[n];
          for (int j = 0; j < m_ndim; ++j) {
            gradients[r][n][i][j] =
              z_vec_grad[i][j] * m_basis[n] +
              z_vec[i] * m_grad_basis[n][j];
          }
        }
      }
    } else {
      fail("unsupported error weight field type");
    }
    apf::destroyElement(z_elem);
  }
}

double ErrorWeight::val(int i, int n, int eq) {
  return values[i][n][eq];
}

double ErrorWeight::grad(int i, int n, int eq, int dim) {
  return gradients[i][n][eq][dim];
}

}
