#include "weights.hpp"

namespace calibr8 {

Weight::Weight(apf::FieldShape* shape) {
  m_shape = shape;
}

Weight::~Weight() {
}

void Weight::in_elem(apf::MeshElement* me, RCP<Disc>) {
  m_mesh_elem = me;
}

void Weight::evaluate(apf::Vector3 const& xi) {
  apf::getBF(m_shape, m_mesh_elem, xi, m_BF);
  apf::getGradBF(m_shape, m_mesh_elem, xi, m_gBF);
}

double Weight::val(int node, int eq) {
  (void)eq;
  return m_BF[node];
}

double Weight::grad(int node, int eq, int dim) {
  return m_gBF[node][dim];
}

void Weight::out_elem() {
  m_mesh_elem = nullptr;
}

AdjointWeight::AdjointWeight(apf::FieldShape* shape) :
  Weight(shape) {
}

AdjointWeight::~AdjointWeight() {
}

void AdjointWeight::in_elem(apf::MeshElement* me, RCP<Disc> disc) {
  //TODO: fill in
  m_mesh_elem = me;
}

void AdjointWeight::gather(RCP<Disc> disc, RCP<VectorT> Z) {
  // TODO: fill in
  (void)disc;
  (void)Z;
}

void AdjointWeight::evaluate(apf::Vector3 const& xi) {
  apf::getBF(m_shape, this->m_mesh_elem, xi, this->m_BF);
  apf::getGradBF(m_shape, this->m_mesh_elem, xi, this->m_gBF);
  // TODO: fill in
}

double AdjointWeight::val(int node, int eq) {
  return m_vals[node][eq];
}

double AdjointWeight::grad(int node, int eq, int dim) {
  return m_grads[node][eq][dim];
}

void AdjointWeight::out_elem() {
  // TODO: fill in
}

}
