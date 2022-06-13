#include "disc.hpp"
#include "fad.hpp"
#include "global_residual.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "normal_traction.hpp"
#include "ree.h"

namespace calibr8 {

template <typename T>
NormalTraction<T>::NormalTraction(ParameterList const& params) {
  m_side_set = params.get<std::string>("side set");
  m_elem_set = params.get<std::string>("elem set");
}

template <typename T>
NormalTraction<T>::~NormalTraction() {
}

template <typename T>
void NormalTraction<T>::before_elems(RCP<Disc> disc, int step) {
  this->m_mesh = disc->apf_mesh();
  this->m_num_dims = disc->num_dims();
  this->m_shape = disc->gv_shape();
  this->m_step = step;
  this->setup_mapping(m_side_set, disc, m_mapping);
  m_elem_set_idx = disc->elem_set_idx(m_elem_set);
}

template <typename T>
void NormalTraction<T>::evaluate(
    int elem_set,
    int elem,
    RCP<GlobalResidual<T>> global,
    RCP<LocalResidual<T>> local,
    apf::Vector3 const& iota_input,
    double,
    double) {

  this->initialize_value_pt();

  // only do one side of the interface (if non-manifold)
  if (elem_set != m_elem_set_idx) return;

  // get the id of the facet wrt element if this facet is on the QoI side
  // do not evaluate if the facet is not adjacent to the QoI side
  int const facet_id = m_mapping[elem_set][elem];
  if (facet_id < 0) return;

  // store some information contained in this class as local variables
  int const ndims = this->m_num_dims;
  apf::Mesh* mesh = this->m_mesh;
  apf::MeshElement* mesh_elem = this->m_mesh_elem;

  // grab the face to integrate over
  apf::Downward elem_faces;
  apf::MeshEntity* elem_entity = apf::getMeshEntity(mesh_elem);
  mesh->getDownward(elem_entity, ndims - 1, elem_faces);
  apf::MeshEntity* face = elem_faces[facet_id];
  apf::MeshElement* me = apf::createMeshElement(mesh, face);

  // get a single 1st order point on the face
  apf::Vector3 iota_face;
  apf::getIntPoint(me, 1, 0, iota_face);
  double const w = apf::getIntWeight(me, 1, 0);
  double const dv = apf::getDV(me, iota_face);

  // map the integration point on the face parametric space
  // to the element parametric space
  apf::Vector3 iota_elem = boundaryToElementXi(
      mesh, face, elem_entity, iota_face);

  // compute the normal at this integration point
  apf::Vector3 n = ree::computeFaceOutwardNormal(
      mesh, elem_entity, face, iota_face);

  // evaluate the stress at the integration point
  int const pressure_idx = 1;
  global->interpolate(iota_elem);
  T const p = global->scalar_x(pressure_idx);
  Tensor<T> const sigma = local->cauchy(global, p);

  // compute the normal load at the point
  T load = T(0.);
  for (int i = 0; i < ndims; ++i) {
    for (int j = 0; j < ndims; ++j) {
      load += n[i] * sigma(i,j) * n[j];
    }
  }

  // contribute to the point value
  this->value_pt += load * w * dv;

  // clean up allocated memory
  apf::destroyMeshElement(me);

  // reset the state in global residual to what it was on input
  global->interpolate(iota_input);

}

template class NormalTraction<double>;
template class NormalTraction<FADT>;

}
