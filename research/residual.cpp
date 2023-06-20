#include "residual.hpp"

#include "nlpoisson.hpp"
#include "nlelasticity.hpp"

namespace calibr8 {

template <typename T>
Residual<T>::Residual(int ndims) {
  m_ndims = ndims;
}

template <typename T>
Residual<T>::~Residual() {
}

template <typename T>
void resize(Array2D<T>& v, int ni, int nj) {
  v.resize(ni);
  for (int i = 0; i < ni; ++i) {
    v[i].resize(nj);
  }
}

template <typename T>
void zero(Array2D<T>& v) {
  for (size_t i = 0; i < v.size(); ++i) {
    for (size_t j = 0; j < v[i].size(); ++j) {
      v[i][j] = 0.;
    }
  }
}

template <typename T>
void Residual<T>::set_space(int space, RCP<Disc> disc) {
  m_space = space;
  if (m_space == -1) {
    m_nnodes = -1;
    m_ndofs = -1;
    m_vals.resize(0);
    m_resid.resize(0);
  } else {
    m_nnodes = disc->get_num_nodes(space);
    m_ndofs = m_nnodes * m_neqs;
    m_vals.resize(0);
    m_resid.resize(0);
    resize(m_vals, m_nnodes, m_neqs);
    resize(m_resid, m_nnodes, m_neqs);
  }
}

int get_index(int node, int eq, int neqs) {
  return node*neqs + eq;
}

template <> double val<double>(double const& in) { return in; }
template <> double val<FADT>(FADT const& in) { return in.val(); }
template <> double val<FAD2T>(FAD2T const& in) { return in.val().val(); }

template <typename T>
void Residual<T>::in_elem(apf::MeshElement* me, RCP<Disc> disc) {
  m_mesh_elem = me;
}

template <typename T>
void Residual<T>::out_elem() {
  m_mesh_elem = nullptr;
}

template <>
void Residual<double>::gather(RCP<Disc> disc, RCP<VectorT> u) {
  apf::MeshEntity* ent = apf::getMeshEntity(m_mesh_elem);
  auto u_data = u->get1dView();
  for (int node = 0; node < m_nnodes; ++node) {
    for (int eq = 0; eq < m_neqs; ++eq) {
      LO const row = disc->get_lid(m_space, ent, node, eq);
      m_vals[node][eq] = u_data[row];
      m_resid[node][eq] = 0.;
    }
  }
}

template <>
void Residual<FADT>::gather(RCP<Disc> disc, RCP<VectorT> u) {
  apf::MeshEntity* ent = apf::getMeshEntity(m_mesh_elem);
  auto u_data = u->get1dView();
  for (int node = 0; node < m_nnodes; ++node) {
    for (int eq = 0; eq < m_neqs; ++eq) {
      int const idx = get_index(node, eq, m_neqs);
      LO const row = disc->get_lid(m_space, ent, node, eq);
      m_vals[node][eq].diff(idx, m_ndofs);
      m_vals[node][eq].val() = u_data[row];
      m_resid[node][eq] = 0.;
    }
  }
}

template <>
void Residual<FAD2T>::gather(RCP<Disc> disc, RCP<VectorT> u) {
  apf::MeshEntity* ent = apf::getMeshEntity(m_mesh_elem);
  auto u_data = u->get1dView();
  for (int node = 0; node < m_nnodes; ++node) {
    for (int eq = 0; eq < m_neqs; ++eq) {
      int const idx = get_index(node, eq, m_neqs);
      LO const row = disc->get_lid(m_space, ent, node, eq);
      FADT first;
      first.val() = u_data[row];
      first.diff(idx, m_ndofs);
      m_vals[node][eq].val() = first;
      m_vals[node][eq].diff(idx, m_ndofs);
      m_resid[node][eq] = 0.;
    }
  }
}

template <typename T>
void Residual<T>::interp_basis(apf::Vector3 const& xi, RCP<Disc> disc) {
  apf::FieldShape* shape = disc->shape(m_space);
  apf::getBF(shape, m_mesh_elem, xi, m_BF);
  apf::getGradBF(shape, m_mesh_elem, xi, m_gBF);
}

template <typename T>
Array1D<T> Residual<T>::interp(apf::Vector3 const& xi) {
  Array1D<T> u(m_neqs, 0.);
  for (int eq = 0; eq < m_neqs; ++eq) {
    for (int node = 0; node < m_nnodes; ++node) {
      u[eq] += m_vals[node][eq] * m_BF[node];
    }
  }
  return u;
}

template <typename T>
Array2D<T> Residual<T>::interp_grad(apf::Vector3 const& xi) {
  Array2D<T> grad_u;
  resize(grad_u, m_neqs, m_ndims);
  zero(grad_u);
  for (int eq = 0; eq < m_neqs; ++eq) {
    for (int node = 0; node < m_nnodes; ++node) {
      for (int dim = 0; dim < m_ndims; ++dim) {
        grad_u[eq][dim] += m_vals[node][eq] * m_gBF[node][dim];
      }
    }
  }
  return grad_u;
}

template <typename T>
void Residual<T>::scatter_residual(RCP<Disc> disc, RCP<VectorT> r) {
  apf::MeshEntity* ent = apf::getMeshEntity(m_mesh_elem);
  auto r_data = r->get1dViewNonConst();
  for (int node = 0; node < m_nnodes; ++node) {
    for (int eq = 0; eq < m_neqs; ++eq) {
      LO const row = disc->get_lid(m_space, ent, node, eq);
      r_data[row] += val(m_resid[node][eq]);
    }
  }
}

template <>
void Residual<double>::scatter_jacobian(RCP<Disc>, RCP<MatrixT>) {
}

template <>
void Residual<FADT>::scatter_jacobian(RCP<Disc> disc, RCP<MatrixT> J) {
  using Teuchos::arrayView;
  apf::MeshEntity* ent = apf::getMeshEntity(m_mesh_elem);
  for (int row_node = 0; row_node < m_nnodes; ++row_node) {
    for (int row_eq = 0; row_eq < m_neqs; ++row_eq) {
      LO const row = disc->get_lid(m_space, ent, row_node, row_eq);
      FADT const val = m_resid[row_node][row_eq];
      for (int col_node = 0; col_node < m_nnodes; ++col_node) {
        for (int col_eq = 0; col_eq < m_neqs; ++col_eq) {
          LO const col = disc->get_lid(m_space, ent, col_node, col_eq);
          int const idx = get_index(col_node, col_eq, m_neqs);
          double const dval = val.fastAccessDx(idx);
          J->sumIntoLocalValues(row, arrayView(&col, 1), arrayView(&dval, 1));
        }
      }
    }
  }
}

template <>
void Residual<double>::scatter_adjoint(RCP<Disc>, RCP<MatrixT>) {
}

template <>
void Residual<FADT>::scatter_adjoint(RCP<Disc> disc, RCP<MatrixT> J) {
  using Teuchos::arrayView;
  apf::MeshEntity* ent = apf::getMeshEntity(m_mesh_elem);
  for (int row_node = 0; row_node < m_nnodes; ++row_node) {
    for (int row_eq = 0; row_eq < m_neqs; ++row_eq) {
      LO const row = disc->get_lid(m_space, ent, row_node, row_eq);
      FADT const val = m_resid[row_node][row_eq];
      for (int col_node = 0; col_node < m_nnodes; ++col_node) {
        for (int col_eq = 0; col_eq < m_neqs; ++col_eq) {
          LO const col = disc->get_lid(m_space, ent, col_node, col_eq);
          int const idx = get_index(col_node, col_eq, m_neqs);
          double const dval = val.fastAccessDx(idx);
          J->sumIntoLocalValues(col, arrayView(&row, 1), arrayView(&dval, 1));
        }
      }
    }
  }
}

template <>
void Residual<double>::scatter(RCP<Disc> disc, System const& sys) {
  if (m_mode == RESIDUAL) {
    scatter_residual(disc, sys.b);
  } else {
    throw std::runtime_error("invalid mode");
  }
}

template <>
void Residual<FADT>::scatter(RCP<Disc> disc, System const& sys) {
  if (m_mode == JACOBIAN) {
    scatter_residual(disc, sys.b);
    scatter_jacobian(disc, sys.A);
  } else if (m_mode == ADJOINT) {
    scatter_adjoint(disc, sys.A);
  } else {
    throw std::runtime_error("invalid mode");
  }
}

template <>
void Residual<FAD2T>::scatter(RCP<Disc>, System const&) {
}

template <typename T>
RCP<Residual<T>> create_residual(ParameterList const& params, int ndims) {
  std::string const type = params.get<std::string>("type");
  if (type == "nonlinear poisson") {
    return rcp(new NLPoisson<T>(params, ndims));
  } else if (type == "nonlinear elasticity") {
    return rcp(new NLElasticity<T>(params, ndims));
  } else {
    throw std::runtime_error("invalid residual");
  }
}

template class Residual<double>;
template class Residual<FADT>;
template class Residual<FAD2T>;

template RCP<Residual<double>> create_residual(ParameterList const& params, int ndims);
template RCP<Residual<FADT>> create_residual(ParameterList const& params, int ndims);
template RCP<Residual<FAD2T>> create_residual(ParameterList const& params, int ndims);

}
