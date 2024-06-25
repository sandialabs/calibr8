#include "weight.hpp"

namespace calibr8 {

Weight::Weight(apf::FieldShape* shape) {
  m_shape = shape;
}

void Weight::evaluate(apf::MeshElement* me, apf::Vector3 const& iota) {
  apf::getBF(m_shape, me, iota, m_basis);
  apf::getGradBF(m_shape, me, iota, m_grad_basis);
}

double Weight::val(int i, int n, int eq) {
  (void)i;
  (void)eq;
  return m_basis[n];
}

double Weight::grad(int i, int n, int eq, int dim) {
  (void)i;
  (void)eq;
  return m_grad_basis[n][dim];
}

}
