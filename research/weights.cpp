#include "weights.hpp"

namespace calibr8 {

Weight::Weight(apf::FieldShape* shape) {
  m_shape = shape;
}

void Weight::evaluate(apf::MeshElement* me, apf::Vector3 const& xi) {
  apf::getBF(m_shape, me, xi, m_BF);
  apf::getGradBF(m_shape, me, xi, m_gBF);
}

double Weight::val(int node, int eq) {
  (void)eq;
  return m_BF[node];
}

double Weight::grad(int node, int eq, int dim) {
  return m_gBF[node][dim];
}

}
