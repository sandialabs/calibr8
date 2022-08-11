#include "qoi.hpp"
#include "residual.hpp"
#include "sol_avg.hpp"

namespace calibr8 {

template <typename T>
QoI<T>::QoI() {
}

template <typename T>
QoI<T>::~QoI() {
}

template <typename T>
void QoI<T>::reset() {
  m_value = 0.;
}

template <typename T>
void QoI<T>::in_elem(
    apf::MeshElement* me,
    RCP<Residual<T>> residual,
    RCP<Disc> disc) {
  m_mesh_elem = me;
  apf::MeshEntity* ent = apf::getMeshEntity(me);
  m_elem_value = 0.;
  m_neqs = residual->num_eqs();
  m_nnodes = disc->get_num_nodes(m_space, ent);
}

template <typename T>
void QoI<T>::out_elem() {
  this->m_value += m_elem_value;
  m_mesh_elem = nullptr;
  m_neqs = -1;
  m_nnodes = -1;
}

template <>
void QoI<double>::scatter(RCP<Disc>, System*) {
}

template <>
void QoI<FADT>::scatter(RCP<Disc> disc, System* sys) {
  RCP<VectorT> J = sys->b;
  auto J_data = J->get1dViewNonConst();
  apf::MeshEntity* ent = apf::getMeshEntity(m_mesh_elem);
  for (int node = 0; node < m_nnodes; ++node) {
    for (int eq = 0; eq < this->m_neqs; ++eq) {
      LO const row = disc->get_lid(m_space, ent, node, eq);
      int idx = get_index(node, eq, this->m_neqs);
      J_data[row] += m_elem_value.fastAccessDx(idx);
    }
  }
}

template <typename T>
double QoI<T>::value() {
  return val(m_value);
}

template <typename T>
RCP<QoI<T>> create_QoI(ParameterList const& params) {
  std::string const type = params.get<std::string>("type");
  if (type == "solution average") {
    return rcp(new SolAvg<T>(params));
  } else {
    return Teuchos::null;
  }
}

template class QoI<double>;
template class QoI<FADT>;

template RCP<QoI<double>> create_QoI(ParameterList const&);
template RCP<QoI<FADT>> create_QoI(ParameterList const&);

}
