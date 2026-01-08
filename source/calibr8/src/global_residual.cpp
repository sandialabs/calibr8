#include "control.hpp"
#include "defines.hpp"
#include "disc.hpp"
#include "fad.hpp"
#include "global_residual.hpp"
#include "macros.hpp"
#include "mechanics.hpp"
#include "mechanics_plane_stress.hpp"

namespace calibr8 {

template <typename T>
GlobalResidual<T>::GlobalResidual() {
}

template <typename T>
GlobalResidual<T>::~GlobalResidual() {
}

template <typename T>
int GlobalResidual<T>::dx_idx(int i, int node, int eq) const {
  return m_dx_offsets[i] + (node * m_num_eqs[i] + eq);
}

template <typename T>
T GlobalResidual<T>::scalar_x(int i) const {
  DEBUG_ASSERT(m_var_types[i] == SCALAR);
  return m_x[i][0];
}

template <typename T>
Vector<T> GlobalResidual<T>::vector_x(int i) const {
  DEBUG_ASSERT(m_var_types[i] == VECTOR);
  Vector<T> val(m_num_dims);
  for (int dim = 0; dim < m_num_dims; ++dim) {
    val(dim) = m_x[i][dim];
  }
  return val;
}

template <typename T>
Vector<T> GlobalResidual<T>::grad_scalar_x(int i) const {
  DEBUG_ASSERT(m_var_types[i] == SCALAR);
  Vector<T> val(m_num_dims);
  for (int dim = 0; dim < m_num_dims; ++dim) {
    val(dim) = m_grad_x[i][0][dim];
  }
  return val;
}

template <typename T>
Tensor<T> GlobalResidual<T>::grad_vector_x(int i) const {
  DEBUG_ASSERT(m_var_types[i] == VECTOR);
  Tensor<T> val(m_num_dims);
  for (int k = 0; k < m_num_dims; ++k) {
    for (int l = 0; l < m_num_dims; ++l) {
      val(k, l) = m_grad_x[i][k][l];
    }
  }
  return val;
}

template <typename T>
T GlobalResidual<T>::scalar_x_prev(int i) const {
  DEBUG_ASSERT(m_var_types[i] == SCALAR);
  return m_x_prev[i][0];
}

template <typename T>
Vector<T> GlobalResidual<T>::vector_x_prev(int i) const {
  DEBUG_ASSERT(m_var_types[i] == VECTOR);
  Vector<T> val(m_num_dims);
  for (int dim = 0; dim < m_num_dims; ++dim) {
    val(dim) = m_x_prev[i][dim];
  }
  return val;
}

template <typename T>
Vector<T> GlobalResidual<T>::grad_scalar_x_prev(int i) const {
  DEBUG_ASSERT(m_var_types[i] == SCALAR);
  Vector<T> val(m_num_dims);
  for (int dim = 0; dim < m_num_dims; ++dim) {
    val(dim) = m_grad_x_prev[i][0][dim];
  }
  return val;
}

template <typename T>
Tensor<T> GlobalResidual<T>::grad_vector_x_prev(int i) const {
  DEBUG_ASSERT(m_var_types[i] == VECTOR);
  Tensor<T> val(m_num_dims);
  for (int k = 0; k < m_num_dims; ++k) {
    for (int l = 0; l < m_num_dims; ++l) {
      val(k, l) = m_grad_x_prev[i][k][l];
    }
  }
  return val;
}

template <typename T>
void GlobalResidual<T>::before_elems(
    RCP<Disc> disc,
    int mode,
    Array1D<apf::Field*> const& adjoint_fields) {

  // set discretization-based information
  m_num_dims = disc->num_dims();
  m_num_nodes = disc->num_gv_nodes_per_elem();
  m_mesh = disc->apf_mesh();
  m_shape = disc->gv_shape();

  // create weighting functions
  if (mode == NORMAL_WEIGHT) {
      m_weight = new Weight(m_shape);
      m_stab_weight = new Weight(m_shape);
  } else if (mode == ERROR_WEIGHT) {
      m_weight = new ErrorWeight(m_shape, m_num_dims, m_num_residuals,
          m_num_nodes, m_num_eqs, adjoint_fields);
      m_stab_weight = new ErrorWeight(m_shape, m_num_dims, m_num_residuals,
          m_num_nodes, m_num_eqs, adjoint_fields);
  }

  // resize the nodal quantities
  resize(m_x_nodal, m_num_residuals, m_num_nodes, m_num_eqs);
  resize(m_R_nodal, m_num_residuals, m_num_nodes, m_num_eqs);
  resize(m_x_prev_nodal, m_num_residuals, m_num_nodes, m_num_eqs);

  // resize the interpolated value quantities
  resize(m_x, m_num_residuals, m_num_eqs);
  resize(m_x_prev, m_num_residuals, m_num_eqs);
  resize(m_grad_x, m_num_residuals, m_num_eqs, m_num_dims);
  resize(m_grad_x_prev, m_num_residuals, m_num_eqs, m_num_dims);

  // initialize the offsets for indexing into derivative arrays
  m_num_dofs = 0;
  m_dx_offsets.resize(m_num_residuals);
  for (int i = 0; i < m_num_residuals; ++i) {
    m_dx_offsets[i] = m_num_dofs;
    m_num_dofs += m_num_eqs[i] * m_num_nodes;
  }

}

template <typename T>
void GlobalResidual<T>::set_elem(apf::MeshElement* mesh_elem) {
  m_mesh_elem = mesh_elem;
}

template <>
void GlobalResidual<double>::zero_residual() {
  for (int i = 0; i < m_num_residuals; ++i) {
    int const neqs = m_num_eqs[i];
    for (int n = 0; n < m_num_nodes; ++n) {
      for (int eq = 0; eq < neqs; ++eq) {
        m_R_nodal[i][n][eq] = 0.;
      }
    }
  }
}

template <>
void GlobalResidual<FADT>::zero_residual() {
  for (int i = 0; i < m_num_residuals; ++i) {
    int const neqs = m_num_eqs[i];
    for (int n = 0; n < m_num_nodes; ++n) {
      for (int eq = 0; eq < neqs; ++eq) {
        m_R_nodal[i][n][eq] = 0.;
        for (int k = 0; k < nmax_derivs; ++k) {
          m_R_nodal[i][n][eq].fastAccessDx(k) = 0.;
        }
      }
    }
  }
}

template <>
void GlobalResidual<DFADT>::zero_residual() {}

template <typename T>
void GlobalResidual<T>::gather(
    Array1D<apf::Field*> const& x,
    Array1D<apf::Field*> const& x_prev) {
  for (int i = 0; i < m_num_residuals; ++i) {
    int const neqs = m_num_eqs[i];
    int const type = m_var_types[i];
    Array2D<double> const x_vals =
      get_nodal_components(x[i], m_mesh_elem, type, m_num_nodes);
    Array2D<double> const x_prev_vals =
      get_nodal_components(x_prev[i], m_mesh_elem, type, m_num_nodes);
    for (int n = 0; n < m_num_nodes; ++n) {
      for (int eq = 0; eq < neqs; ++eq) {
        m_x_nodal[i][n][eq] = x_vals[n][eq];
        m_x_prev_nodal[i][n][eq] = x_prev_vals[n][eq];
      }
    }
  }
}

template <>
int GlobalResidual<double>::seed_wrt_x() {
  return -1;
}

template <>
int GlobalResidual<FADT>::seed_wrt_x() {
  for (int i = 0; i < m_num_residuals; ++i) {
    for (int n = 0; n < m_num_nodes; ++n) {
      for (int eq = 0; eq < m_num_eqs[i]; ++eq) {
        int const dof = dx_idx(i, n, eq);
        m_x_nodal[i][n][eq].diff(dof, m_num_dofs);
      }
    }
  }
  return m_num_dofs;
}

template <>
int GlobalResidual<DFADT>::seed_wrt_x() {
  return -1;
}

template <>
void GlobalResidual<double>::unseed_wrt_x() {}

template <>
void GlobalResidual<FADT>::unseed_wrt_x() {
  for (int i = 0; i < m_num_residuals; ++i) {
    for (int n = 0; n < m_num_nodes; ++n) {
      for (int eq = 0; eq < m_num_eqs[i]; ++eq) {
        m_x_nodal[i][n][eq] = m_x_nodal[i][n][eq].val();
        for (int idx = 0; idx < nmax_derivs; ++idx) {
          m_x_nodal[i][n][eq].fastAccessDx(idx) = 0.;
          m_R_nodal[i][n][eq].fastAccessDx(idx) = 0.;
        }
      }
    }
  }
}

template <>
void GlobalResidual<DFADT>::unseed_wrt_x() {}

template <>
int GlobalResidual<double>::seed_wrt_x_prev() {
  return -1;
}

template <>
int GlobalResidual<FADT>::seed_wrt_x_prev() {
  for (int i = 0; i < m_num_residuals; ++i) {
    for (int n = 0; n < m_num_nodes; ++n) {
      for (int eq = 0; eq < m_num_eqs[i]; ++eq) {
        int const dof = dx_idx(i, n, eq);
        m_x_prev_nodal[i][n][eq].diff(dof, m_num_dofs);
      }
    }
  }
  return m_num_dofs;
}

template <>
int GlobalResidual<DFADT>::seed_wrt_x_prev() {
  return -1;
}

template <>
void GlobalResidual<double>::unseed_wrt_x_prev() {}

template <>
void GlobalResidual<FADT>::unseed_wrt_x_prev() {
  for (int i = 0; i < m_num_residuals; ++i) {
    for (int n = 0; n < m_num_nodes; ++n) {
      for (int eq = 0; eq < m_num_eqs[i]; ++eq) {
        m_x_prev_nodal[i][n][eq] = m_x_prev_nodal[i][n][eq].val();
        for (int idx = 0; idx < nmax_derivs; ++idx) {
          m_x_prev_nodal[i][n][eq].fastAccessDx(idx) = 0.;
          m_R_nodal[i][n][eq].fastAccessDx(idx) = 0.;
        }
      }
    }
  }
}

template <>
void GlobalResidual<DFADT>::unseed_wrt_x_prev() {}

template <typename T>
void GlobalResidual<T>::interpolate(apf::Vector3 const& iota) {

  m_weight->evaluate(m_mesh_elem, iota);
  m_stab_weight->evaluate(m_mesh_elem, iota);

  // interpolate the global state variables
  for (int i = 0; i < m_num_residuals; ++i) {
    for (int eq = 0; eq < m_num_eqs[i]; ++eq) {
      m_x[i][eq] = x_nodal(i, 0, eq) * this->weight(i, 0, eq);
      m_x_prev[i][eq] = x_prev_nodal(i, 0, eq) * this->weight(i, 0, eq);
      for (int n = 1; n < m_num_nodes; ++n) {
        m_x[i][eq] += x_nodal(i, n, eq) * this->weight(i, n, eq);
        m_x_prev[i][eq] += x_prev_nodal(i, n, eq) * this->weight(i, n, eq);
      }
    }
  }

  // interpolate the global state variable gradients
  for (int i = 0; i < m_num_residuals; ++i) {
    for (int eq = 0; eq < m_num_eqs[i]; ++eq) {
      for (int d = 0; d < m_num_dims; ++d) {
        m_grad_x[i][eq][d] =
            x_nodal(i, 0, eq) * this->grad_weight(i, 0, eq, d);
        m_grad_x_prev[i][eq][d] =
            x_prev_nodal(i, 0, eq) * this->grad_weight(i, 0, eq, d);
        for (int n = 1; n < m_num_nodes; ++n) {
          m_grad_x[i][eq][d] +=
              x_nodal(i, n, eq) * this->grad_weight(i, n, eq, d);
          m_grad_x_prev[i][eq][d] +=
              x_prev_nodal(i, n, eq) * this->grad_weight(i, n, eq, d);
        }
      }
    }
  }

  apf::Vector3 x(0, 0, 0);
  apf::mapLocalToGlobal(m_mesh_elem, iota, x);
  m_pt_global_coords(0) = x[0];
  m_pt_global_coords(1) = x[1];
  m_pt_global_coords(2) = x[2];

}

template <typename T>
void GlobalResidual<T>::interpolate_with_error(apf::Vector3 const& iota) {

  m_weight->evaluate(m_mesh_elem, iota);
  m_stab_weight->evaluate(m_mesh_elem, iota);

  apf::NewArray<double> basis;
  apf::NewArray<apf::Vector3> dbasis;
  apf::getBF(m_shape, m_mesh_elem, iota, basis);
  apf::getGradBF(m_shape, m_mesh_elem, iota, dbasis);

  // interpolate the global state variables
  for (int i = 0; i < m_num_residuals; ++i) {
    for (int eq = 0; eq < m_num_eqs[i]; ++eq) {
      m_x[i][eq] = x_nodal(i, 0, eq) * basis[0];
      m_x_prev[i][eq] = x_prev_nodal(i, 0, eq) * basis[0];
      for (int n = 1; n < m_num_nodes; ++n) {
        m_x[i][eq] += x_nodal(i, n, eq) * basis[n];
        m_x_prev[i][eq] += x_prev_nodal(i, n, eq) * basis[n];
      }
    }
  }

  // interpolate the global state variable gradients
  for (int i = 0; i < m_num_residuals; ++i) {
    for (int eq = 0; eq < m_num_eqs[i]; ++eq) {
      for (int d = 0; d < m_num_dims; ++d) {
        m_grad_x[i][eq][d] =
            x_nodal(i, 0, eq) * dbasis[0][d];
        m_grad_x_prev[i][eq][d] =
            x_prev_nodal(i, 0, eq) * dbasis[0][d];
        for (int n = 1; n < m_num_nodes; ++n) {
          m_grad_x[i][eq][d] +=
              x_nodal(i, n, eq) * dbasis[n][d];
          m_grad_x_prev[i][eq][d] +=
              x_prev_nodal(i, n, eq) * dbasis[n][d];
        }
      }
    }
  }

}

template <typename T>
EVector GlobalResidual<T>::eigen_residual() const {
  EVector R(m_num_dofs);
  for (int i = 0; i < m_num_residuals; ++i) {
    for (int n = 0; n < m_num_nodes; ++n) {
      for (int i_eq = 0; i_eq < m_num_eqs[i]; ++i_eq) {
        int const i_idx = dx_idx(i, n, i_eq);
        R[i_idx] = val(m_R_nodal[i][n][i_eq]);
      }
    }
  }
  return R;
}

template <>
EMatrix GlobalResidual<double>::eigen_jacobian(int) const {
  EMatrix empty;
  return empty;
}

template <>
EMatrix GlobalResidual<FADT>::eigen_jacobian(int nderivs) const {
  EMatrix J(m_num_dofs, nderivs);
  J.setZero();
  for (int i = 0; i < m_num_residuals; ++i) {
    for (int n = 0; n < m_num_nodes; ++n) {
      for (int i_eq = 0; i_eq < m_num_eqs[i]; ++i_eq) {
        int const i_idx = dx_idx(i, n, i_eq);
        for (int j = 0; j < nderivs; ++j) {
          J(i_idx, j) = m_R_nodal[i][n][i_eq].fastAccessDx(j);
        }
      }
    }
  }
  return J;
}

template <>
EMatrix GlobalResidual<DFADT>::eigen_jacobian(int) const {
  EMatrix empty;
  return empty;
}

template <typename T>
EVector GlobalResidual<T>::gather_adjoint(Array1D<apf::Field*> const& z) const {
  EVector z_nodes(m_num_dofs);
  for (int i = 0; i < m_num_residuals; ++i) {
    int const neqs = m_num_eqs[i];
    int const type = m_var_types[i];
    Array2D<double> const z_vals =
      get_nodal_components(z[i], m_mesh_elem, type, m_num_nodes);
    for (int n = 0; n < m_num_nodes; ++n) {
      for (int eq = 0; eq < neqs; ++eq) {
        int const idx = dx_idx(i, n, eq);
        z_nodes[idx] = z_vals[n][eq];
      }
    }
  }
  return z_nodes;
}

template <typename T>
EVector GlobalResidual<T>::gather_difference(
    Array1D<apf::Field*> const& x_fine,
    Array1D<apf::Field*> const& x) const {
  EVector diff(m_num_dofs);
  for (int i = 0; i < m_num_residuals; ++i) {
    int const neqs = m_num_eqs[i];
    int const type = m_var_types[i];
    Array2D<double> const x_fine_vals =
      get_nodal_components(x_fine[i], m_mesh_elem, type, m_num_nodes);
    Array2D<double> const x_vals =
      get_nodal_components(x[i], m_mesh_elem, type, m_num_nodes);
    for (int n = 0; n < m_num_nodes; ++n) {
      for (int eq = 0; eq < neqs; ++eq) {
        int const idx = dx_idx(i, n, eq);
        diff[idx] = x_fine_vals[n][eq] - x_vals[n][eq];
      }
    }
  }
  return diff;
}

template <typename T>
void GlobalResidual<T>::scatter_rhs(
    RCP<Disc> disc,
    EVector const& rhs,
    Array1D<RCP<VectorT>>& RHS) {
  apf::MeshEntity* ent = apf::getMeshEntity(m_mesh_elem);
  for (int i = 0; i < m_num_residuals; ++i) {
    auto R_data = RHS[i]->get1dViewNonConst();
    Array2D<LO> const rows = disc->get_element_lids(ent, i);
    for (int n = 0; n < m_num_nodes; ++n) {
      for (int eq = 0; eq < m_num_eqs[i]; ++eq) {
        LO const row = rows[n][eq];
        int const i_idx = dx_idx(i, n, eq);
        R_data[row] += rhs(i_idx);
      }
    }
  }
}

template <typename T>
void GlobalResidual<T>::scatter_sens(
    RCP<Disc> disc,
    EMatrix const& sens,
    Array1D<RCP<MultiVectorT>>& MV) {
  apf::MeshEntity* ent = apf::getMeshEntity(m_mesh_elem);
  int const num_params = sens.cols();
  for (int i = 0; i < m_num_residuals; ++i) {
    Array2D<LO> const rows = disc->get_element_lids(ent, i);
    for (int p = 0 ; p < num_params; ++p) {
      auto dR_data = MV[i]->getVectorNonConst(p)->get1dViewNonConst();
      for (int n = 0; n < m_num_nodes; ++n) {
        for (int eq = 0; eq < m_num_eqs[i]; ++eq) {
          LO const row = rows[n][eq];
          int const i_idx = dx_idx(i, n, eq);
          dR_data[row] += sens(i_idx, p);
        }
      }
    }
  }
}

template <typename T>
void GlobalResidual<T>::assign_rhs(
    RCP<Disc> disc,
    EVector const& rhs,
    Array1D<RCP<VectorT>>& RHS) {
  apf::MeshEntity* ent = apf::getMeshEntity(m_mesh_elem);
  for (int i = 0; i < m_num_residuals; ++i) {
    auto R_data = RHS[i]->get1dViewNonConst();
    Array2D<LO> const rows = disc->get_element_lids(ent, i);
    for (int n = 0; n < m_num_nodes; ++n) {
      for (int eq = 0; eq < m_num_eqs[i]; ++eq) {
        LO const row = rows[n][eq];
        int const i_idx = dx_idx(i, n, eq);
        R_data[row] = rhs[i_idx];
      }
    }
  }
}

template <>
void GlobalResidual<double>::scatter_lhs(
    RCP<Disc>,
    EMatrix const&,
    Array2D<RCP<MatrixT>>&) {}

template <>
void GlobalResidual<FADT>::scatter_lhs(
    RCP<Disc> disc,
    EMatrix const& dtotal,
    Array2D<RCP<MatrixT>>& dR_dx) {

  Teuchos::Array<LO> colIndices;
  Teuchos::Array<double> values;

  colIndices.reserve(m_num_dofs);
  values.reserve(m_num_dofs);

  // get the mesh entity associated with the 'mesh element'
  apf::MeshEntity* ent = apf::getMeshEntity(m_mesh_elem);

  // loop over the first residual index
  for (int i = 0; i < m_num_residuals; ++i) {
    Array2D<LO> const rows = disc->get_element_lids(ent, i);
    for (int i_node = 0; i_node < m_num_nodes; ++i_node) {
      for (int i_eq = 0; i_eq < m_num_eqs[i]; ++i_eq) {
        int const i_idx = dx_idx(i, i_node, i_eq);
        LO const row = rows[i_node][i_eq];

        // loop over the second residual index
        for (int j = 0; j < m_num_residuals; ++j) {
          Array2D<LO> const cols = disc->get_element_lids(ent, j);
          colIndices.clear();
          values.clear();
          for (int j_node = 0; j_node < m_num_nodes; ++j_node) {
            for (int j_eq = 0; j_eq < m_num_eqs[j]; ++j_eq) {
              int const j_idx = dx_idx(j, j_node, j_eq);
              LO const col = cols[j_node][j_eq];
              double const dx = dtotal(i_idx, j_idx);
              colIndices.push_back(col);
              values.push_back(dx);
            }
          }
          dR_dx[i][j]->sumIntoLocalValues(row, colIndices, values);
        }

      }
    }
  }
}

template <>
void GlobalResidual<DFADT>::scatter_lhs(
    RCP<Disc>,
    EMatrix const&,
    Array2D<RCP<MatrixT>>&) {}

template <typename T>
void GlobalResidual<T>::unset_elem() {
  m_mesh_elem = nullptr;
}

template <typename T>
void GlobalResidual<T>::after_elems() {
  ALWAYS_ASSERT(m_weight);
  ALWAYS_ASSERT(m_stab_weight);
  delete m_weight;
  delete m_stab_weight;
  m_num_dims = -1;
  m_num_nodes = -1;
  m_num_dofs = -1;
  m_mesh = nullptr;
  m_shape = nullptr;
  m_x_nodal.resize(0);
  m_R_nodal.resize(0);
  m_x_prev_nodal.resize(0);
  m_dx_offsets.resize(0);
  m_x.resize(0);
  m_x_prev.resize(0);
  m_grad_x.resize(0);
  m_grad_x_prev.resize(0);
}

template <typename T>
RCP<GlobalResidual<T>> create_global_residual(
    ParameterList const& params, int ndims) {
  std::string const type = params.get<std::string>("type");
  if (type == "mechanics") {
    return rcp(new Mechanics<T>(params, ndims));
  } else if (type == "mechanics_plane_stress") {
    return rcp(new MechanicsPlaneStress<T>(params, ndims));
  } else {
    return Teuchos::null;
  }
}

template <typename T>
Array1D<int> GlobalResidual<T>::ip_sets() const {
  return m_ip_sets;
}

template <typename T>
void GlobalResidual<T>::set_stabilization_h(int stabilization_h) {
  m_stabilization_h = stabilization_h;
}

template <typename T>
void GlobalResidual<T>::set_time_info(double time, double dt) {
  m_time = time;
  m_delta_t = dt;
}

template class GlobalResidual<double>;
template class GlobalResidual<FADT>;
template class GlobalResidual<DFADT>;

template RCP<GlobalResidual<double>>
create_global_residual<double>(ParameterList const&, int);

template RCP<GlobalResidual<FADT>>
create_global_residual<FADT>(ParameterList const&, int);

template RCP<GlobalResidual<DFADT>>
create_global_residual<DFADT>(ParameterList const&, int);


}
