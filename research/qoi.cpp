#include "qoi.hpp"
#include "qoi_gradient.hpp"
#include "qoi_point.hpp"
#include "qoi_sqrt_gradient.hpp"
#include "qoi_value.hpp"
#include "residual.hpp"

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
void QoI<T>::set_space(int space, RCP<Disc> disc) {
  m_space = space;
  if (m_space == -1) {
    m_nnodes = -1;
  } else {
    m_nnodes = disc->get_num_nodes(space);
  }
}

template <typename T>
void strip_derivs(T& val);

template <> void strip_derivs(double&) {}

template <> void strip_derivs(FADT& val) {
  for (int i = 0; i < nmax_derivs; ++i) {
    val.fastAccessDx(i) = 0.;
  }
}

template <> void strip_derivs(FAD2T& val) {
  for (int i = 0; i < nmax_derivs; ++i) {
    val.fastAccessDx(i).val() = 0.;
    for (int j = 0; j < nmax_derivs; ++j) {
      val.fastAccessDx(i).fastAccessDx(j) = 0.;
    }
  }
}

template <typename T>
void QoI<T>::in_elem(
    apf::MeshElement* me,
    RCP<Residual<T>> residual,
    RCP<Disc> disc) {
  m_mesh_elem = me;
  apf::MeshEntity* ent = apf::getMeshEntity(me);
  m_elem_value = T(0.);
  strip_derivs(m_elem_value);
  m_neqs = residual->num_eqs();
}

template <typename T>
void QoI<T>::out_elem() {
  this->m_value += m_elem_value;
  m_mesh_elem = nullptr;
  m_neqs = -1;
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

template <>
void QoI<FAD2T>::scatter(RCP<Disc> disc, System* sys) {
  using Teuchos::arrayView;
  RCP<MatrixT> H = sys->A;
  apf::MeshEntity* ent = apf::getMeshEntity(m_mesh_elem);
  for (int row_node = 0; row_node < m_nnodes; ++row_node) {
    for (int row_eq = 0; row_eq < m_neqs; ++row_eq) {
      LO const row = disc->get_lid(m_space, ent, row_node, row_eq);
      int const row_idx = get_index(row_node, row_eq, m_neqs);
      for (int col_node = 0; col_node < m_nnodes; ++col_node) {
        for (int col_eq = 0; col_eq < m_neqs; ++col_eq) {
          LO const col = disc->get_lid(m_space, ent, col_node, col_eq);
          int const col_idx = get_index(col_node, col_eq, m_neqs);
          double const d2val = m_elem_value.fastAccessDx(row_idx).fastAccessDx(col_idx);
          H->sumIntoLocalValues(row, arrayView(&col, 1), arrayView(&d2val, 1));
        }
      }
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
  if (type == "value") {
    return rcp(new QoI_Value<T>(params));
  } else if (type == "gradient") {
    return rcp(new QoI_Gradient<T>(params));
  } else if (type == "sqrt gradient") {
    return rcp(new QoI_SqrtGradient<T>(params));
  } else if (type == "point") {
    return rcp(new QoI_Point<T>(params));
  } else {
    throw std::runtime_error("invalid qoi");
  }
}

template class QoI<double>;
template class QoI<FADT>;
template class QoI<FAD2T>;

template RCP<QoI<double>> create_QoI(ParameterList const&);
template RCP<QoI<FADT>> create_QoI(ParameterList const&);
template RCP<QoI<FAD2T>> create_QoI(ParameterList const&);

}
