#include <apf.h>
#include <apfMesh.h>
#include <apfShape.h>

#include "weight.hpp"

namespace calibr8 {

Weight::Weight(apf::FieldShape* shape) {
  m_shape = shape;
}

void Weight::at_point(apf::MeshElement* me, apf::Vector3 const& xi) {
  apf::getBF(m_shape, me, xi, m_basis);
  apf::getGradBF(m_shape, me, xi, m_grad_basis);
}

double Weight::val(int n, int eq) {
  (void)eq;
  return m_basis[n];
}

double Weight::grad(int n, int eq, int dim) {
  (void)eq;
  return m_grad_basis[n][dim];
}

AdjointWeight::AdjointWeight(apf::FieldShape* PU, apf::Field* z) :
  Weight(PU),
  m_z(z)
{
  if (apf::countComponents(z) != 1) {
    throw std::runtime_error("not a scalar weight");
  }
}

void AdjointWeight::at_point(apf::MeshElement* me, apf::Vector3 const& xi) {

  apf::Element* z_elem = apf::createElement(m_z, me);
  auto mesh = apf::getMesh(m_z);
  auto entity = apf::getMeshEntity(me);
  auto entity_type = mesh->getType(entity);
  int const nnodes = m_shape->getEntityShape(entity_type)->countNodes();
  int const ndims = mesh->getDimension();
  m_vals.resize(nnodes);
  m_grads.resize(nnodes);

  // bng - debug
  //m_vals.resize(6);
  //m_grads.resize(6);
  //for (int i = 0; i < 6; ++i) {
  //  m_vals[i] = 0.;
  //  m_grads[i] = apf::Vector3(0,0,0);
  //}

  apf::getBF(m_shape, me, xi, m_basis);
  apf::getGradBF(m_shape, me, xi, m_grad_basis);

  double z = apf::getScalar(z_elem, xi);
  apf::Vector3 grad_z(0,0,0);
  apf::getGrad(z_elem, xi, grad_z);

  for (int n = 0; n < nnodes; ++n) {
    m_vals[n] = z * m_basis[n];
    for (int i = 0; i < ndims; ++i) {
      m_grads[n][i] = grad_z[i] * m_basis[n] + z * m_grad_basis[n][i];
    }
  }

  apf::destroyElement(z_elem);

}

double AdjointWeight::val(int n, int eq) {
  (void)eq;
  return m_vals[n];
}

double AdjointWeight::grad(int n, int eq, int dim) {
  (void)eq;
  return m_grads[n][dim];
}

}
