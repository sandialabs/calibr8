#include "disc.hpp"
#include "global_residual.hpp"
#include "load_mismatch.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "material_params.hpp"
#include "ree.h"

namespace calibr8 {

template <typename T>
LoadMismatch<T>::LoadMismatch(ParameterList const& params) {
  m_side_set = params.get<std::string>("side set");
  // DTS: how to set default value?
  m_predict_load = params.get<bool>("predict load");
}

template <typename T>
LoadMismatch<T>::~LoadMismatch() {
}

template <typename T>
void LoadMismatch<T>::before_elems(RCP<Disc> disc, int step) {

  // set the discretization-based information
  this->m_mesh = disc->apf_mesh();
  this->m_num_dims = disc->num_dims();
  this->m_shape = disc->gv_shape();
  this->m_step = step;
  m_load_mismatch_computed = false;
  //this->vec_value_pt = Vector<T>(this->m_num_dims);

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
void LoadMismatch<T>::evaluate(
    int elem_set,
    int elem,
    RCP<GlobalResidual<T>> global,
    RCP<LocalResidual<T>> local,
    apf::Vector3 const& iota_input,
    double,
    double) {


  // initialize the QoI contribution to 0
  int const ndims = this->m_num_dims;
  T const dummy1 = global->vector_x(0)[0];
  T const dummy2 = local->first_value();
  this->value_pt = 0. * (dummy1 + dummy2);
  //for (int i = 0; i < ndims; ++i) {
  //  this->vec_value_pt(i) = 0. * (dummy1 + dummy2);
  //}

  // get the id of the facet wrt element if this facet is on the QoI side
  // do not evaluate if the facet is not adjacent to the QoI side
  int const facet_id = m_mapping[elem_set][elem];
  if (facet_id < 0) return;

  // store some information contained in this class as local variables
  int const step = this->m_step;
  apf::Mesh* mesh = this->m_mesh;
  apf::MeshElement* mesh_elem = this->m_mesh_elem;

  // grab the field for the measured displacement data at the step
  // std::string name = "measured_" + std::to_string(step);
  // apf::Field* f_meas = mesh->findField(name.c_str());
  // ALWAYS_ASSERT(f_meas);
  // apf::Element* e_meas = createElement(f_meas, mesh_elem);

  // grab the face to integrate over
  apf::Downward elem_faces;
  apf::MeshEntity* elem_entity = apf::getMeshEntity(mesh_elem);
  mesh->getDownward(elem_entity, ndims - 1, elem_faces);
  apf::MeshEntity* face = elem_faces[facet_id];

  // get quadrature information over the face
  // int const q_order = 2;
  int const q_order = 1;
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
    // we assume displacements are index 0 and pressure is index 1
    int const disp_idx = 0;
    int const pressure_idx = 1;
    global->interpolate(iota_elem);
    Tensor<T> const grad_u = global->grad_vector_x(disp_idx);
    T const p = global->scalar_x(pressure_idx);

    // gather material properties
    T const E = local->params(0);
    T const nu = local->params(1);
    T const mu = compute_mu(E, nu);

    // compute the stress
    Tensor<T> stress = local->cauchy(global, p);
    if (local->is_finite_deformation()) {
      Tensor<T> const I = minitensor::eye<T>(ndims);
      Tensor<T> const F = grad_u + I;
      Tensor<T> const F_inv = inverse(F);
      Tensor<T> const F_invT = transpose(F_inv);
      T const J = det(F);
      stress = J * stress * F_invT;
    }

    apf::Vector3 N = ree::computeFaceOutwardNormal(mesh, elem_entity, face,
        iota_face);

    // compute the normal load
    T load = T(0.);
    for (int i = 0; i < ndims; ++i) {
      for (int j = 0; j < ndims; ++j) {
        load += N[i] * stress(i, j) * N[j];
      }
    }

    // integrate the load over the surface
    this->value_pt = load * w * dv;

  }

  // clean up allocated memory
  // apf::destroyElement(e_meas);
  apf::destroyMeshElement(me);

  // reset the state in global residual to what it was on input
  global->interpolate(iota_input);


}

template <typename T>
void LoadMismatch<T>::finalize(int step, double& J) {
  if (m_predict_load) {
    print("Load on surface %s at step %d = %.16e",
        m_side_set.c_str(), step, J);
  } else {
    // compute H_{diff}(i) = H(i) - H^{meas}(i) (store this for each step?)
    // finalized value -> \sum_i^n 0.5 * ||H_{diff}||^2
    (void) step;
    (void) J;
  }
}

template class LoadMismatch<double>;
template class LoadMismatch<FADT>;

}
