#include "macros.hpp"
#include "point_wise.hpp"
#include "disc.hpp"
#include "nested.hpp"
#include "global_residual.hpp"
#include "local_residual.hpp"
#include "state.hpp"

namespace calibr8 {

template <typename T>
PointWise<T>::PointWise(ParameterList const& params) {
  m_node_set = params.get<std::string>("node set");
  m_component = params.get<int>("component");
  m_step = params.get<int>("step");
}

template <typename T>
PointWise<T>::~PointWise() {
}

template <typename T>
void PointWise<T>::before_elems(RCP<Disc> disc, int step) {
  m_disc = disc;
  m_current_step = step;
}

template <typename T>
void PointWise<T>::evaluate(
    int,
    int,
    RCP<GlobalResidual<T>>,
    RCP<LocalResidual<T>>,
    apf::Vector3 const&,
    double,
    double) {
  this->initialize_value_pt();
}

apf::MeshEntity* get_owned_vertex(
    std::string const& node_set,
    RCP<Disc> disc) {
  apf::MeshEntity* owned_vtx = 0;
  NodeSet const& nodes = disc->nodes(node_set);
  if (nodes.size() > 0) {
    ALWAYS_ASSERT(nodes.size() == 1);
    apf::MeshEntity* vtx = nodes[0].entity;
    if (disc->apf_mesh()->isOwned(vtx)) {
      owned_vtx = vtx;
    }
  }
  return owned_vtx;
}

template <typename T>
void PointWise<T>::postprocess(double& J) {
  if (m_current_step != m_step) return;
  apf::MeshEntity* owned_vtx = get_owned_vertex(m_node_set, m_disc);
  if (!owned_vtx) return;
  apf::Vector3 u;
  static constexpr int const momentum_idx = 0;
  apf::Field* disp = nullptr;
  if (m_disc->type() == VERIFICATION) {
    RCP<NestedDisc> nested = Teuchos::rcp_static_cast<NestedDisc>(m_disc);
    ALWAYS_ASSERT(nested != Teuchos::null);
    disp = (nested->primal_fine(m_step)).global[momentum_idx];
  } else {
    disp = (m_disc->primal(m_step)).global[momentum_idx];
  }
  apf::getVector(disp, owned_vtx, 0, u);
  J = u[m_component];
}

template <typename T>
void PointWise<T>::modify_state(RCP<State> state) {
  if (m_current_step != m_step) return;
  apf::MeshEntity* owned_vtx = get_owned_vertex(m_node_set, m_disc);
  if (!owned_vtx) return;
  static constexpr int const momentum_idx = 0;
  apf::Node node{owned_vtx, 0};
  LO const row = m_disc->get_lid(node, momentum_idx, m_component);
  auto dJdx = (state->la->b[GHOST][momentum_idx])->get1dViewNonConst();
  dJdx[row] = -1.;
}

template class PointWise<double>;
template class PointWise<FADT>;

}
