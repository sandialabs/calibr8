#include <PCU.h>
#include <fstream>
#include <iomanip>
#include "disc.hpp"
#include "global_residual.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "material_params.hpp"
#include "reaction.hpp"

namespace calibr8 {

template <typename T>
Reaction<T>::Reaction(ParameterList const& params) {
  m_coord_idx = params.get<int>("coordinate index");
  m_coord_value = params.get<double>("coordinate value");
  if (params.isParameter("coordinate tolerance")) {
    m_coord_tol = params.get<double>("coordinate tolerance");
  }
  m_reaction_force_comp = params.get<int>("reaction force component");
  if (params.isParameter("compute torque")) {
    m_compute_torque = params.get<bool>("compute torque");
  }
}

template <typename T>
Reaction<T>::~Reaction() {
}

template <typename T>
void Reaction<T>::before_elems(RCP<Disc> disc, int step) {

  // set the discretization-based information
  this->m_mesh = disc->apf_mesh();
  this->m_num_dims = disc->num_dims();
  this->m_shape = disc->gv_shape();
  this->m_step = step;

  if (!is_initd) {
    is_initd = this->setup_coord_based_node_mapping(m_coord_idx, m_coord_value,
        m_coord_tol, disc, m_mapping);
  }

}

template <typename T>
T Reaction<T>::compute_load(
    int elem_set,
    int elem,
    RCP<GlobalResidual<T>> global,
    RCP<LocalResidual<T>> local,
    apf::Vector3 const& iota,
    double w,
    double dv) {

  T load_pt = T(0.);

  // store some information contained in this class as local variables
  int const ndims = this->m_num_dims;
  int const step = this->m_step;
  apf::Mesh* mesh = this->m_mesh;
  apf::MeshElement* mesh_elem = this->m_mesh_elem;
  apf::MeshEntity* elem_entity = apf::getMeshEntity(mesh_elem);

  // get the nodes for the element
  apf::Downward elem_verts;
  int const num_elem_nodes = mesh->getDownward(elem_entity, 0, elem_verts);

  // grab the nodes for the reactions
  Array1D<int> const& node_ids = m_mapping[elem_set][elem];
  int const num_nodes = node_ids.size();

  // compute the internal force vector
  global->zero_residual();
  global->evaluate(local, iota, w, dv, 0);

  // sum relevant entries
  int const disp_idx = 0;
  apf::Vector3 r(0, 0, 0);
  for (size_t i = 0; i < num_nodes; ++i) {
    int const node_id = node_ids[i];
    if (m_compute_torque) {
      mesh->getPoint(elem_verts[node_id], 0, r);
      load_pt += compute_torque(global, r, node_id);
    } else {
      load_pt += global->R_nodal(disp_idx, node_id, m_reaction_force_comp);
    }
  }

  return load_pt;
}

template <typename T>
T Reaction<T>::compute_torque(
    RCP<GlobalResidual<T>> global,
    apf::Vector3 const& r,
    int const node_id)
{
  int const disp_idx = 0;
  if (m_reaction_force_comp == 2) {
    T const F_0 = global->R_nodal(disp_idx, node_id, 0);
    T const F_1 = global->R_nodal(disp_idx, node_id, 1);
    return r[0] * F_1 - r[1] * F_0;
  } else if (m_reaction_force_comp == 0) {
    T const F_1 = global->R_nodal(disp_idx, node_id, 1);
    T const F_2 = global->R_nodal(disp_idx, node_id, 2);
    return r[1] * F_2 - r[2] * F_1;
  } else if (m_reaction_force_comp == 1) {
    T const F_0 = global->R_nodal(disp_idx, node_id, 0);
    T const F_2 = global->R_nodal(disp_idx, node_id, 2);
    return r[2] * F_0 - r[0] * F_2;
  } else {
    fail("invalid specified reaction force component for torque computation");
  }
}

template <typename T>
void Reaction<T>::evaluate(
    int elem_set,
    int elem,
    RCP<GlobalResidual<T>> global,
    RCP<LocalResidual<T>> local,
    apf::Vector3 const& iota,
    double w,
    double dv) {

  this->initialize_value_pt();

  // get the id of the node wrt element if this node is on the QoI node set
  // do not evaluate if the element contains no nodes in the QoI node set
  int const node_id = m_mapping[elem_set][elem][0];
  if (node_id < 0) return;

  this->value_pt = compute_load(elem_set, elem, global, local, iota, w, dv);
}

template class Reaction<double>;
template class Reaction<FADT>;

}
