#include "disc.hpp"
#include "global_residual.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "surface_mismatch.hpp"

namespace calibr8 {

template <typename T>
SurfaceMismatch<T>::SurfaceMismatch(ParameterList const& params) {
  m_side_set = params.get<std::string>("side set");
}

template <typename T>
SurfaceMismatch<T>::~SurfaceMismatch() {
}

template <typename T>
void SurfaceMismatch<T>::before_elems(RCP<Disc> disc, int step) {

  // set the discretization-based information
  this->m_mesh = disc->apf_mesh();
  this->m_num_dims = disc->num_dims();
  this->m_shape = disc->gv_shape();
  this->m_step = step;

  // initialize the surface mesh information
  if (!is_initd) {
    int ndims = this->m_num_dims;
    apf::Mesh* mesh = disc->apf_mesh();
    apf::Downward downward_faces;
    m_mapping.resize(disc->num_elem_sets());
    SideSet const& sides = disc->sides(m_side_set);
    for (int es = 0; es < disc->num_elem_sets(); ++es) {
      std::string const& es_name = disc->elem_set_name(es);
      ElemSet const& elems = disc->elems(es_name);
      m_mapping[es].resize(elems.size());
      for (size_t elem = 0; elem < elems.size(); ++elem) {
        m_mapping[es][elem] = -1;
        apf::MeshEntity* elem_entity = elems[elem];
        int ndown = mesh->getDownward(elem_entity, ndims - 1, downward_faces);
        for (int down = 0; down < ndown; ++down) {
          apf::MeshEntity* downward_entity = downward_faces[down];
          for (apf::MeshEntity* side : sides) {
            if (side == downward_entity) {
              m_mapping[es][elem] = down;
            }
          }
        }
      }
    }
    is_initd = true;
  }

}

template <typename T>
void SurfaceMismatch<T>::evaluate(
    int elem_set,
    int elem,
    RCP<GlobalResidual<T>> global,
    RCP<LocalResidual<T>> local,
    apf::Vector3 const& iota_input,
    double,
    double) {

  // initialize the QoI contribution to 0
  T const dummy1 = global->vector_x(0)[0];
  T const dummy2 = local->first_value();
  T const dummy3 = local->params(0);
  this->value_pt = 0. * (dummy1 + dummy2 + dummy3);

  // get the id of the facet wrt element if this facet is on the QoI side
  // do not evaluate if the facet is not adjacent to the QoI side
  int const facet_id = m_mapping[elem_set][elem];
  if (facet_id < 0) return;

  // store some information contained in this class as local variables
  int const ndims = this->m_num_dims;
  int const step = this->m_step;
  apf::Mesh* mesh = this->m_mesh;
  apf::MeshElement* mesh_elem = this->m_mesh_elem;

  // grab the field for the measured displacement data at the step
  std::string name = "measured_" + std::to_string(step);
  apf::Field* f_meas = mesh->findField(name.c_str());
  ALWAYS_ASSERT(f_meas);
  apf::Element* e_meas = createElement(f_meas, mesh_elem);

  // grab the face to integrate over
  apf::Downward elem_faces;
  apf::MeshEntity* elem_entity = apf::getMeshEntity(mesh_elem);
  mesh->getDownward(elem_entity, ndims - 1, elem_faces);
  apf::MeshEntity* face = elem_faces[facet_id];

  // get quadrature information over the face
  int const q_order = 2;
  apf::MeshElement* me = apf::createMeshElement(mesh, face);
  int const num_qps = apf::countIntPoints(me, q_order);

  // numerically integrate the QoI over the face
  for (int pt = 0; pt < num_qps; ++pt) {

    // get the integration point info on the face
    apf::Vector3 iota_face;
    apf::getIntPoint(me, q_order, pt, iota_face);
    double const w = apf::getIntWeight(me, q_order, pt);
    double const dv = apf::getDV(me, iota_face);

    // map the integration point on the face parametric space to
    // the element parametric space
    apf::Vector3 iota_elem = boundaryToElementXi(
        mesh, face, elem_entity, iota_face);

    // interpolate the global variable solution to face integration point
    // we assume displacements are index 0
    int const disp_idx = 0;
    global->interpolate(iota_elem);
    Vector<T> const u_fem = global->vector_x(disp_idx);

    // interpolate the measured displacement data to the point
    apf::Vector3 u_meas;
    apf::getVector(e_meas, iota_elem, u_meas);

    // compute the QoI contribution at the point
    T const qoi =
      (u_fem[0] - u_meas[0]) * (u_fem[0] - u_meas[0]) +
      (u_fem[1] - u_meas[1]) * (u_fem[1] - u_meas[1]) +
      (u_fem[2] - u_meas[2]) * (u_fem[2] - u_meas[2]);
    
    // compute the difference between the FEM displacement and
    // the measured input displacement data
    this->value_pt += qoi * w * dv;

  }

  // clean up allocated memory
  apf::destroyElement(e_meas);
  apf::destroyMeshElement(me);

  // reset the state in global residual to what it was on input
  global->interpolate(iota_input);


}

template class SurfaceMismatch<double>;
template class SurfaceMismatch<FADT>;

}
