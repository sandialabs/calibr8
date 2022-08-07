#include "residual.hpp"
#include "nlpoisson.hpp"

namespace calibr8 {

template <typename T>
Residual<T>::Residual(int ndims) {
  m_ndims = ndims;
}

template <typename T>
Residual<T>::~Residual() {
}

template class Residual<double>;
template class Residual<FADT>;

template <typename T>
void resize(Array2D<T>& v, int ni, int nj) {
  v.resize(ni);
  for (int i = 0; i < ni; ++i) {
    v[i].resize(nj);
  }
}

static int get_idx(int node, int eq, int neqs) {
  return node*neqs + eq;
}

template <typename T> double val(T const& in);
template <> double val<double>(double const& in) { return in; }
template <> double val<FADT>(FADT const& in) { return in.val(); }

template <>
void Residual<double>::gather(apf::MeshElement* me, RCP<Disc> disc, RCP<VectorT> u) {
  apf::MeshEntity* ent = apf::getMeshEntity(me);
  m_nnodes = disc->get_num_nodes(m_space, ent);
  m_ndofs = m_nnodes * m_neqs;
  resize(m_vals, m_nnodes, m_neqs);
  resize(m_resid, m_nnodes, m_neqs);
  auto u_data = u->get1dViewNonConst();
  for (int node = 0; node < m_nnodes; ++node) {
    for (int eq = 0; eq < m_neqs; ++eq) {
      LO row = disc->get_lid(m_space, ent, node, eq);
      m_vals[node][eq] = u_data[row];
      m_resid[node][eq] = 0.;
    }
  }
}

template <>
void Residual<FADT>::gather(apf::MeshElement* me, RCP<Disc> disc, RCP<VectorT> u) {
  apf::MeshEntity* ent = apf::getMeshEntity(me);
  m_nnodes = disc->get_num_nodes(m_space, ent);
  m_ndofs = m_nnodes * m_neqs;
  resize(m_vals, m_nnodes, m_neqs);
  resize(m_resid, m_nnodes, m_neqs);
  auto u_data = u->get1dViewNonConst();
  for (int node = 0; node < m_nnodes; ++node) {
    for (int eq = 0; eq < m_neqs; ++eq) {
      int const idx = get_idx(node, eq, m_neqs);
      LO const row = disc->get_lid(m_space, ent, node, eq);
      m_vals[node][eq].diff(idx, m_ndofs);
      m_vals[node][eq].val() = u_data[row];
      m_resid[node][eq] = 0.;
    }
  }
}

template <typename T>
void Residual<T>::scatter_residual(apf::MeshElement* me, RCP<Disc> disc, RCP<VectorT> r) {
  apf::MeshEntity* ent = apf::getMeshEntity(me);
  auto r_data = r->get1dViewNonConst();
  for (int node = 0; node < m_nnodes; ++node) {
    for (int eq = 0; eq < m_neqs; ++eq) {
      LO const row = disc->get_lid(m_space, ent, node, eq);
      r_data[row] += val(m_resid[node][eq]);
    }
  }
}

template <>
void Residual<double>::scatter_jacobian(apf::MeshElement*, RCP<Disc>, RCP<MatrixT>) {
}

template <>
void Residual<FADT>::scatter_jacobian(apf::MeshElement* me, RCP<Disc> disc, RCP<MatrixT> J) {
  using Teuchos::arrayView;
  apf::MeshEntity* ent = apf::getMeshEntity(me);
  for (int row_node = 0; row_node < m_nnodes; ++row_node) {
    for (int row_eq = 0; row_eq < m_neqs; ++row_eq) {
      LO const row = disc->get_lid(m_space, ent, row_node, row_eq);
      FADT const val = m_resid[row_node][row_eq];
      for (int col_node = 0; col_node < m_nnodes; ++col_node) {
        for (int col_eq = 0; col_eq < m_neqs; ++col_eq) {
          LO const col = disc->get_lid(m_space, ent, col_node, col_eq);
          int const idx = get_idx(col_node, col_eq, m_neqs);
          double const dval = val.fastAccessDx(idx);
          J->sumIntoLocalValues(row, arrayView(&col, 1), arrayView(&dval, 1));
        }
      }
    }
  }
}

template <>
void Residual<double>::scatter_adjoint(apf::MeshElement*, RCP<Disc>, RCP<MatrixT>) {
}

template <>
void Residual<FADT>::scatter_adjoint(apf::MeshElement* me, RCP<Disc> disc, RCP<MatrixT> J) {
  using Teuchos::arrayView;
  apf::MeshEntity* ent = apf::getMeshEntity(me);
  for (int row_node = 0; row_node < m_nnodes; ++row_node) {
    for (int row_eq = 0; row_eq < m_neqs; ++row_eq) {
      LO const row = disc->get_lid(m_space, ent, row_node, row_eq);
      FADT const val = m_resid[row_node][row_eq];
      for (int col_node = 0; col_node < m_nnodes; ++col_node) {
        for (int col_eq = 0; col_eq < m_neqs; ++col_eq) {
          LO const col = disc->get_lid(m_space, ent, col_node, col_eq);
          int const idx = get_idx(col_node, col_eq, m_neqs);
          double const dval = val.fastAccessDx(idx);
          J->sumIntoLocalValues(col, arrayView(&row, 1), arrayView(&dval, 1));
        }
      }
    }
  }
}

template <>
void Residual<double>::scatter(apf::MeshElement* me, RCP<Disc> disc, RCP<System> sys) {
  if (m_mode == RESIDUAL) {
    scatter_residual(me, disc, sys->b[m_space][GHOST]);
  } else {
    throw std::runtime_error("invalid mode");
  }
}

template <>
void Residual<FADT>::scatter(apf::MeshElement* me, RCP<Disc> disc, RCP<System> sys) {
  if (m_mode == JACOBIAN) {
    scatter_residual(me, disc, sys->b[m_space][GHOST]);
    scatter_jacobian(me, disc, sys->A[m_space][GHOST]);
  } else if (m_mode == ADJOINT) {
    scatter_adjoint(me, disc, sys->A[m_space][GHOST]);
  } else {
    throw std::runtime_error("invalid mode");
  }
}

template <typename T>
RCP<Residual<T>> create_residual(ParameterList const& params, int ndims) {
  std::string const type = params.get<std::string>("type");
  if (type == "nonlinear poisson") {
    return rcp(new NLPoisson<T>(params, ndims));
  } else {
    return Teuchos::null;
  }
}

template RCP<Residual<double>> create_residual(ParameterList const& params, int ndims);
template RCP<Residual<FADT>> create_residual(ParameterList const& params, int ndims);

}
