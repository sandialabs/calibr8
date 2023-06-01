#include "defines.hpp"
#include "disc.hpp"
#include "elastic.hpp"
#include "fad.hpp"
#include "hyper_J2.hpp"
#include "hyper_J2_plane_strain.hpp"
#include "hyper_J2_plane_stress.hpp"
#include "hypo_hill.hpp"
#include "hypo_hill_plane_strain.hpp"
#include "hypo_hill_plane_stress.hpp"
#include "isotropic_elastic.hpp"
#include "linear_thermoviscoelastic.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "small_hill.hpp"
#include "small_hill_plane_strain.hpp"
#include "small_hill_plane_stress.hpp"
#include "small_J2.hpp"
#include "state.hpp"

namespace calibr8 {

template <typename T>
LocalResidual<T>::LocalResidual() {
}

template <typename T>
LocalResidual<T>::~LocalResidual() {
}

template <typename T>
void LocalResidual<T>::init_variables(
    RCP<State> state,
    bool set_IC) {
  RCP<Disc> disc = state->disc;
  int const num_elem_sets = disc->num_elem_sets();
  m_elem_set_names.resize(num_elem_sets);
  for (int es = 0; es < num_elem_sets; ++es) {
    m_elem_set_names[es] = disc->elem_set_name(es);
  }
  this->init_params();

  if (set_IC) {
    int const model_form = state->model_form;
    Array1D<apf::Field*>* xi;
    if (disc->type() == VERIFICATION) {
      xi = &state->disc->primal_fine(/*step=*/ 0).local[model_form];
    } else {
      xi = &state->disc->primal(/*step=*/ 0).local[model_form];
    }
    int const q_order = state->disc->lv_shape()->getOrder();
    apf::MeshEntity* elem = nullptr;
    apf::Mesh* mesh = state->disc->apf_mesh();
    apf::MeshIterator* elems = mesh->begin(disc->num_dims());
    int const dummy_es = 0;
    this->before_elems(dummy_es, disc);
    while ((elem = mesh->iterate(elems))) {
      apf::MeshElement* me = apf::createMeshElement(mesh, elem);
      this->set_elem(me);
      int const npts = apf::countIntPoints(me, q_order);
      for (int pt = 0; pt < npts; ++pt) {
        this->init_variables_impl();
        this->scatter(pt, *xi);
      }
      this->unset_elem();
      apf::destroyMeshElement(me);
    }
    mesh->end(elems);
  }

}

template <typename T>
void LocalResidual<T>::before_elems(int const es, RCP<Disc> disc) {

  // set discretization-based information
  m_num_dims = disc->num_dims();
  m_shape = disc->lv_shape();

  // resize the 'nodal' (integration point) quantities
  resize(m_xi, m_num_residuals, m_num_eqs);
  resize(m_xi_prev, m_num_residuals, m_num_eqs);
  resize(m_R, m_num_residuals, m_num_eqs);


  // initialize the offsets for indexing into derivative arrays
  m_num_dofs = 0;
  m_dxi_offsets.resize(m_num_residuals);
  for (int i = 0; i < m_num_residuals; ++i) {
    m_dxi_offsets[i] = m_num_dofs;
    m_num_dofs += m_num_eqs[i];
  }

  if (m_num_aux_vars > 0) {
    // resize the 'nodal' (integration point) quantities for the auxiliary variables
    resize(m_chi, m_num_aux_vars, m_aux_var_num_eqs);
    resize(m_chi_prev, m_num_aux_vars, m_aux_var_num_eqs);

    // initialize the offsets for indexing auxiliary variables
    m_num_aux_dofs = 0;
    m_chi_offsets.resize(m_num_aux_vars);
    for (int i = 0; i < m_num_aux_vars; ++i) {
      m_chi_offsets[i] = m_num_aux_dofs;
      m_num_aux_dofs += m_aux_var_num_eqs[i];
    }
  }

  // set the parameters for the element set
  int const num_params = m_params.size();
  for (int p = 0; p < num_params; ++p) {
    m_params[p] = m_param_values[es][p];
  }

}

template <typename T>
int LocalResidual<T>::dxi_idx(int i, int eq) const {
  return m_dxi_offsets[i] + eq;
}

template <typename T>
double LocalResidual<T>::norm_residual() const {
  double norm = 0.;
  for (int i = 0; i < m_num_residuals; ++i) {
    int const neqs = m_num_eqs[i];
    for (int eq = 0; eq < neqs; ++eq) {
      double const v = val(m_R[i][eq]);
      norm += v * v;
    }
  }
  return std::sqrt(norm);
}

template <>
EMatrix LocalResidual<double>::eigen_jacobian(int) const {
  EMatrix empty;
  return empty;
}

template <>
EMatrix LocalResidual<FADT>::eigen_jacobian(int nderivs) const {
  EMatrix J(m_num_dofs, nderivs);
  J.setZero();
  for (int i = 0; i < m_num_residuals; ++i) {
    for (int i_eq = 0; i_eq < m_num_eqs[i]; ++i_eq) {
      int const i_idx = dxi_idx(i, i_eq);
      for (int j = 0; j < nderivs; ++j) {
        J(i_idx, j) = m_R[i][i_eq].fastAccessDx(j);
      }
    }
  }
  return J;
}

template <typename T>
EVector LocalResidual<T>::eigen_residual() const {
  EVector R(m_num_dofs);
  for (int i = 0; i < m_num_residuals; ++i) {
    for (int eq = 0; eq < m_num_eqs[i]; ++eq) {
      int const dof = dxi_idx(i, eq);
      R(dof) = val(m_R[i][eq]);
    }
  }
  return R;
}

template <typename T>
T LocalResidual<T>::first_value() const {
  return m_xi[0][0];
}

template <typename T>
T LocalResidual<T>::scalar_xi(int i) const {
  DEBUG_ASSERT(m_var_types[i] == SCALAR);
  return m_xi[i][0];
}

template <typename T>
Vector<T> LocalResidual<T>::vector_xi(int i) const {
  DEBUG_ASSERT(m_var_types[i] == VECTOR);
  Vector<T> val(m_num_dims);
  for (int dim = 0; dim < m_num_dims; ++dim) {
    val(dim) = m_xi[i][dim];
  }
  return val;
}

template <typename T>
Tensor<T> LocalResidual<T>::sym_tensor_xi(int i) const {
  DEBUG_ASSERT(m_var_types[i] == SYM_TENSOR);
  Tensor<T> val(m_num_dims);
  if (m_num_dims == 2) {
    val(0, 0) = m_xi[i][0];
    val(0, 1) = m_xi[i][1];
    val(1, 0) = m_xi[i][1];
    val(1, 1) = m_xi[i][2];
  }
  if (m_num_dims == 3) {
    val(0, 0) = m_xi[i][0];
    val(0, 1) = m_xi[i][1];
    val(0, 2) = m_xi[i][2];
    val(1, 0) = m_xi[i][1];
    val(1, 1) = m_xi[i][3];
    val(1, 2) = m_xi[i][4];
    val(2, 0) = m_xi[i][2];
    val(2, 1) = m_xi[i][4];
    val(2, 2) = m_xi[i][5];
  }
  return val;
}

template <typename T>
Tensor<T> LocalResidual<T>::tensor_xi(int i) const {
  DEBUG_ASSERT(m_var_types[i] == TENSOR);
  Tensor<T> val(m_num_dims);
  int eq = 0;
  for (int k = 0; k < m_num_dims; ++k) {
    for (int l = 0; l < m_num_dims; ++l) {
      val(k, l) = m_xi[i][eq++];
    }
  }
  return val;
}

template <typename T>
T LocalResidual<T>::scalar_xi_prev(int i) const {
  DEBUG_ASSERT(m_var_types[i] == SCALAR);
  return m_xi_prev[i][0];
}

template <typename T>
Vector<T> LocalResidual<T>::vector_xi_prev(int i) const {
  DEBUG_ASSERT(m_var_types[i] == VECTOR);
  Vector<T> val(m_num_dims);
  for (int dim = 0; dim < m_num_dims; ++dim) {
    val(dim) = m_xi_prev[i][dim];
  }
  return val;
}

template <typename T>
Tensor<T> LocalResidual<T>::sym_tensor_xi_prev(int i) const {
  DEBUG_ASSERT(m_var_types[i] == SYM_TENSOR);
  Tensor<T> val(m_num_dims);
  if (m_num_dims == 2) {
    val(0, 0) = m_xi_prev[i][0];
    val(0, 1) = m_xi_prev[i][1];
    val(1, 0) = m_xi_prev[i][1];
    val(1, 1) = m_xi_prev[i][2];
  }
  if (m_num_dims == 3) {
    val(0, 0) = m_xi_prev[i][0];
    val(0, 1) = m_xi_prev[i][1];
    val(0, 2) = m_xi_prev[i][2];
    val(1, 0) = m_xi_prev[i][1];
    val(1, 1) = m_xi_prev[i][3];
    val(1, 2) = m_xi_prev[i][4];
    val(2, 0) = m_xi_prev[i][2];
    val(2, 1) = m_xi_prev[i][4];
    val(2, 2) = m_xi_prev[i][5];
  }
  return val;
}

template <typename T>
Tensor<T> LocalResidual<T>::tensor_xi_prev(int i) const {
  DEBUG_ASSERT(m_var_types[i] == TENSOR);
  Tensor<T> val(m_num_dims);
  int eq = 0;
  for (int k = 0; k < m_num_dims; ++k) {
    for (int l = 0; l < m_num_dims; ++l) {
      val(k, l) = m_xi_prev[i][eq++];
    }
  }
  return val;
}

template <>
void LocalResidual<double>::set_scalar_xi(int i, double const& xi) {
  DEBUG_ASSERT(m_var_types[i] == SCALAR);
  m_xi[i][0] = xi;
}

template <>
void LocalResidual<FADT>::set_scalar_xi(int i, FADT const& xi) {
  DEBUG_ASSERT(m_var_types[i] == SCALAR);
  m_xi[i][0].val() = xi.val();
}

template <>
void LocalResidual<double>::set_vector_xi(int i, Vector<double> const& xi) {
  DEBUG_ASSERT(m_var_types[i] == VECTOR);
  for (int k = 0; k < m_num_dims; ++k) {
    m_xi[i][k] = xi(k);
  }
}

template <>
void LocalResidual<FADT>::set_vector_xi(int i, Vector<FADT> const& xi) {
  DEBUG_ASSERT(m_var_types[i] == VECTOR);
  for (int k = 0; k < m_num_dims; ++k) {
    m_xi[i][k].val() = xi(k).val();
  }
}

template <>
void LocalResidual<double>::set_sym_tensor_xi(int i, Tensor<double> const& xi) {
  DEBUG_ASSERT(m_var_types[i] == SYM_TENSOR);
  if (m_num_dims == 2) {
    m_xi[i][0] = xi(0, 0);
    m_xi[i][1] = xi(0, 1);
    m_xi[i][2] = xi(1, 1);
  } else {
    m_xi[i][0] = xi(0, 0);
    m_xi[i][1] = xi(0, 1);
    m_xi[i][2] = xi(0, 2);
    m_xi[i][3] = xi(1, 1);
    m_xi[i][4] = xi(1, 2);
    m_xi[i][5] = xi(2, 2);
  }
}

template <>
void LocalResidual<FADT>::set_sym_tensor_xi(int i, Tensor<FADT> const& xi) {
  DEBUG_ASSERT(m_var_types[i] == SYM_TENSOR);
  if (m_num_dims == 2) {
    m_xi[i][0].val() = xi(0, 0).val();
    m_xi[i][1].val() = xi(0, 1).val();
    m_xi[i][2].val() = xi(1, 1).val();
  } else {
    m_xi[i][0].val() = xi(0, 0).val();
    m_xi[i][1].val() = xi(0, 1).val();
    m_xi[i][2].val() = xi(0, 2).val();
    m_xi[i][3].val() = xi(1, 1).val();
    m_xi[i][4].val() = xi(1, 2).val();
    m_xi[i][5].val() = xi(2, 2).val();
  }
}

template <>
void LocalResidual<double>::set_tensor_xi(int i, Tensor<double> const& xi) {
  DEBUG_ASSERT(m_var_types[i] == TENSOR);
  int eq = 0;
  for (int k = 0; k < m_num_dims; ++k) {
    for (int l = 0; l < m_num_dims; ++l) {
      m_xi[i][eq++] = xi(k, l);
    }
  }
}

template <>
void LocalResidual<FADT>::set_tensor_xi(int i, Tensor<FADT> const& xi) {
  DEBUG_ASSERT(m_var_types[i] == TENSOR);
  int eq = 0;
  for (int k = 0; k < m_num_dims; ++k) {
    for (int l = 0; l < m_num_dims; ++l) {
      m_xi[i][eq++].val() = xi(k, l).val();
    }
  }
}

template <>
void LocalResidual<double>::add_to_scalar_xi(int i, EVector const& dxi) {
  DEBUG_ASSERT(m_var_types[i] == SCALAR);
  int const idx = dxi_idx(i, 0);
  m_xi[i][0] += dxi(idx);
}

template <>
void LocalResidual<FADT>::add_to_scalar_xi(int i, EVector const& dxi) {
  DEBUG_ASSERT(m_var_types[i] == SCALAR);
  int const idx = dxi_idx(i, 0);
  m_xi[i][0].val() += dxi(idx);
}

template <>
void LocalResidual<double>::add_to_vector_xi(int i, EVector const& dxi) {
  DEBUG_ASSERT(m_var_types[i] == VECTOR);
  for (int k = 0; k < m_num_dims; ++k) {
    int const idx = dxi_idx(i, k);
    m_xi[i][k] += dxi(idx);
  }
}

template <>
void LocalResidual<FADT>::add_to_vector_xi(int i, EVector const& dxi) {
  DEBUG_ASSERT(m_var_types[i] == VECTOR);
  for (int k = 0; k < m_num_dims; ++k) {
    int const idx = dxi_idx(i, k);
    m_xi[i][k].val() += dxi(idx);
  }
}

template <>
void LocalResidual<double>::add_to_sym_tensor_xi(int i, EVector const& dxi) {
  DEBUG_ASSERT(m_var_types[i] == SYM_TENSOR);
  if (m_num_dims == 2) {
    m_xi[i][0] += dxi( dxi_idx(i, 0) );
    m_xi[i][1] += dxi( dxi_idx(i, 1) );
    m_xi[i][2] += dxi( dxi_idx(i, 2) );
  } else {
    m_xi[i][0] += dxi( dxi_idx(i, 0) );
    m_xi[i][1] += dxi( dxi_idx(i, 1) );
    m_xi[i][2] += dxi( dxi_idx(i, 2) );
    m_xi[i][3] += dxi( dxi_idx(i, 3) );
    m_xi[i][4] += dxi( dxi_idx(i, 4) );
    m_xi[i][5] += dxi( dxi_idx(i, 5) );
  }
}

template <>
void LocalResidual<FADT>::add_to_sym_tensor_xi(int i, EVector const& dxi) {
  DEBUG_ASSERT(m_var_types[i] == SYM_TENSOR);
  if (m_num_dims == 2) {
    m_xi[i][0].val() += dxi( dxi_idx(i, 0) );
    m_xi[i][1].val() += dxi( dxi_idx(i, 1) );
    m_xi[i][2].val() += dxi( dxi_idx(i, 2) );
  } else {
    m_xi[i][0].val() += dxi( dxi_idx(i, 0) );
    m_xi[i][1].val() += dxi( dxi_idx(i, 1) );
    m_xi[i][2].val() += dxi( dxi_idx(i, 2) );
    m_xi[i][3].val() += dxi( dxi_idx(i, 3) );
    m_xi[i][4].val() += dxi( dxi_idx(i, 4) );
    m_xi[i][5].val() += dxi( dxi_idx(i, 5) );
  }
}

template <>
void LocalResidual<double>::add_to_tensor_xi(int i, EVector const& dxi) {
  DEBUG_ASSERT(m_var_types[i] == TENSOR);
  int eq = 0;
  for (int k = 0; k < m_num_dims; ++k) {
    for (int l = 0; l < m_num_dims; ++l) {
      int const idx = dxi_idx(i, eq);
      m_xi[i][eq] += dxi(idx);
      eq++;
    }
  }
}

template <>
void LocalResidual<FADT>::add_to_tensor_xi(int i, EVector const& dxi) {
  DEBUG_ASSERT(m_var_types[i] == TENSOR);
  int eq = 0;
  for (int k = 0; k < m_num_dims; ++k) {
    for (int l = 0; l < m_num_dims; ++l) {
      int const idx = dxi_idx(i, eq);
      m_xi[i][eq].val() += dxi(idx);
      eq++;
    }
  }
}

template <typename T>
void LocalResidual<T>::set_scalar_R(int i, T const& R) {
  DEBUG_ASSERT(m_var_types[i] == SCALAR);
  m_R[i][0] = R;
}

template <typename T>
void LocalResidual<T>::set_vector_R(int i, Vector<T> const& R) {
  DEBUG_ASSERT(m_var_types[i] == VECTOR);
  for (int k = 0; k < m_num_dims; ++k) {
    m_R[i][k] = R(k);
  }
}

template <typename T>
void LocalResidual<T>::set_sym_tensor_R(int i, Tensor<T> const& R) {
  DEBUG_ASSERT(m_var_types[i] == SYM_TENSOR);
  if (m_num_dims == 2) {
    m_R[i][0] = R(0, 0);
    m_R[i][1] = R(0, 1);
    m_R[i][2] = R(1, 1);
  } else {
    m_R[i][0] = R(0, 0);
    m_R[i][1] = R(0, 1);
    m_R[i][2] = R(0, 2);
    m_R[i][3] = R(1, 1);
    m_R[i][4] = R(1, 2);
    m_R[i][5] = R(2, 2);
  }
}

template <typename T>
void LocalResidual<T>::set_tensor_R(int i, Tensor<T> const& R) {
  DEBUG_ASSERT(m_var_types[i] == TENSOR);
  int eq = 0;
  for (int k = 0; k < m_num_dims; ++k) {
    for (int l = 0; l < m_num_dims; ++l) {
      m_R[i][eq++] = R(k, l);
    }
  }
}

template <typename T>
T LocalResidual<T>::scalar_chi(int i) const {
  DEBUG_ASSERT(m_aux_var_types[i] == SCALAR);
  return m_chi[i][0];
}

template <typename T>
Vector<T> LocalResidual<T>::vector_chi(int i) const {
  DEBUG_ASSERT(m_aux_var_types[i] == VECTOR);
  Vector<T> val(m_num_dims);
  for (int dim = 0; dim < m_num_dims; ++dim) {
    val(dim) = m_chi[i][dim];
  }
  return val;
}

template <typename T>
Tensor<T> LocalResidual<T>::sym_tensor_chi(int i) const {
  DEBUG_ASSERT(m_aux_var_types[i] == SYM_TENSOR);
  Tensor<T> val(m_num_dims);
  if (m_num_dims == 2) {
    val(0, 0) = m_chi[i][0];
    val(0, 1) = m_chi[i][1];
    val(1, 0) = m_chi[i][1];
    val(1, 1) = m_chi[i][2];
  }
  if (m_num_dims == 3) {
    val(0, 0) = m_chi[i][0];
    val(0, 1) = m_chi[i][1];
    val(0, 2) = m_chi[i][2];
    val(1, 0) = m_chi[i][1];
    val(1, 1) = m_chi[i][3];
    val(1, 2) = m_chi[i][4];
    val(2, 0) = m_chi[i][2];
    val(2, 1) = m_chi[i][4];
    val(2, 2) = m_chi[i][5];
  }
  return val;
}

template <typename T>
Tensor<T> LocalResidual<T>::tensor_chi(int i) const {
  DEBUG_ASSERT(m_aux_var_types[i] == TENSOR);
  Tensor<T> val(m_num_dims);
  int eq = 0;
  for (int k = 0; k < m_num_dims; ++k) {
    for (int l = 0; l < m_num_dims; ++l) {
      val(k, l) = m_chi[i][eq++];
    }
  }
  return val;
}

template <typename T>
T LocalResidual<T>::scalar_chi_prev(int i) const {
  DEBUG_ASSERT(m_aux_var_types[i] == SCALAR);
  return m_chi_prev[i][0];
}

template <typename T>
Vector<T> LocalResidual<T>::vector_chi_prev(int i) const {
  DEBUG_ASSERT(m_aux_var_types[i] == VECTOR);
  Vector<T> val(m_num_dims);
  for (int dim = 0; dim < m_num_dims; ++dim) {
    val(dim) = m_chi_prev[i][dim];
  }
  return val;
}

template <typename T>
Tensor<T> LocalResidual<T>::sym_tensor_chi_prev(int i) const {
  DEBUG_ASSERT(m_aux_var_types[i] == SYM_TENSOR);
  Tensor<T> val(m_num_dims);
  if (m_num_dims == 2) {
    val(0, 0) = m_chi_prev[i][0];
    val(0, 1) = m_chi_prev[i][1];
    val(1, 0) = m_chi_prev[i][1];
    val(1, 1) = m_chi_prev[i][2];
  }
  if (m_num_dims == 3) {
    val(0, 0) = m_chi_prev[i][0];
    val(0, 1) = m_chi_prev[i][1];
    val(0, 2) = m_chi_prev[i][2];
    val(1, 0) = m_chi_prev[i][1];
    val(1, 1) = m_chi_prev[i][3];
    val(1, 2) = m_chi_prev[i][4];
    val(2, 0) = m_chi_prev[i][2];
    val(2, 1) = m_chi_prev[i][4];
    val(2, 2) = m_chi_prev[i][5];
  }
  return val;
}

template <typename T>
Tensor<T> LocalResidual<T>::tensor_chi_prev(int i) const {
  DEBUG_ASSERT(m_aux_var_types[i] == TENSOR);
  Tensor<T> val(m_num_dims);
  int eq = 0;
  for (int k = 0; k < m_num_dims; ++k) {
    for (int l = 0; l < m_num_dims; ++l) {
      val(k, l) = m_chi_prev[i][eq++];
    }
  }
  return val;
}

template <typename T>
void LocalResidual<T>::set_scalar_chi(int i, T const& chi) {
  DEBUG_ASSERT(m_aux_var_types[i] == SCALAR);
  m_chi[i][0] = val(chi);
}

template <typename T>
void LocalResidual<T>::set_vector_chi(int i, Vector<T> const& chi) {
  DEBUG_ASSERT(m_aux_var_types[i] == VECTOR);
  for (int k = 0; k < m_num_dims; ++k) {
    m_chi[i][k] = val(chi(k));
  }
}

template <typename T>
void LocalResidual<T>::set_sym_tensor_chi(int i, Tensor<T> const& chi) {
  DEBUG_ASSERT(m_aux_var_types[i] == SYM_TENSOR);
  if (m_num_dims == 2) {
    m_chi[i][0] = val(chi(0, 0));
    m_chi[i][1] = val(chi(0, 1));
    m_chi[i][2] = val(chi(1, 1));
  } else {
    m_chi[i][0] = val(chi(0, 0));
    m_chi[i][1] = val(chi(0, 1));
    m_chi[i][2] = val(chi(0, 2));
    m_chi[i][3] = val(chi(1, 1));
    m_chi[i][4] = val(chi(1, 2));
    m_chi[i][5] = val(chi(2, 2));
  }
}

template <typename T>
void LocalResidual<T>::set_tensor_chi(int i, Tensor<T> const& chi) {
  DEBUG_ASSERT(m_aux_var_types[i] == TENSOR);
  int eq = 0;
  for (int k = 0; k < m_num_dims; ++k) {
    for (int l = 0; l < m_num_dims; ++l) {
      m_chi[i][eq++] = val(chi(k, l));
    }
  }
}

template <typename T>
void LocalResidual<T>::set_scalar_chi_prev(int i, T const& chi_prev) {
  DEBUG_ASSERT(m_aux_var_types[i] == SCALAR);
  m_chi_prev[i][0] = val(chi_prev);
}

template <typename T>
void LocalResidual<T>::set_vector_chi_prev(int i, Vector<T> const& chi_prev) {
  DEBUG_ASSERT(m_aux_var_types[i] == VECTOR);
  for (int k = 0; k < m_num_dims; ++k) {
    m_chi_prev[i][k] = val(chi_prev(k));
  }
}

template <typename T>
void LocalResidual<T>::set_sym_tensor_chi_prev(int i, Tensor<T> const& chi_prev) {
  DEBUG_ASSERT(m_aux_var_types[i] == SYM_TENSOR);
  if (m_num_dims == 2) {
    m_chi_prev[i][0] = val(chi_prev(0, 0));
    m_chi_prev[i][1] = val(chi_prev(0, 1));
    m_chi_prev[i][2] = val(chi_prev(1, 1));
  } else {
    m_chi_prev[i][0] = val(chi_prev(0, 0));
    m_chi_prev[i][1] = val(chi_prev(0, 1));
    m_chi_prev[i][2] = val(chi_prev(0, 2));
    m_chi_prev[i][3] = val(chi_prev(1, 1));
    m_chi_prev[i][4] = val(chi_prev(1, 2));
    m_chi_prev[i][5] = val(chi_prev(2, 2));
  }
}

template <typename T>
void LocalResidual<T>::set_tensor_chi_prev(int i, Tensor<T> const& chi_prev) {
  DEBUG_ASSERT(m_aux_var_types[i] == TENSOR);
  int eq = 0;
  for (int k = 0; k < m_num_dims; ++k) {
    for (int l = 0; l < m_num_dims; ++l) {
      m_chi_prev[i][eq++] = val(chi_prev(k, l));
    }
  }
}

template <typename T>
void LocalResidual<T>::set_elem(apf::MeshElement* mesh_elem) {
  m_mesh_elem = mesh_elem;
}

template <typename T>
void LocalResidual<T>::gather(
    int pt,
    Array1D<apf::Field*> const& xi,
    Array1D<apf::Field*> const& xi_prev) {
  apf::MeshEntity* elem = apf::getMeshEntity(m_mesh_elem);
  for (int i = 0; i < m_num_residuals; ++i) {
    int const neqs = m_num_eqs[i];
    int const type = m_var_types[i];
    Array1D<double> const xi_vals =
      get_node_components(xi[i], elem, pt, type);
    Array1D<double> const xi_prev_vals =
      get_node_components(xi_prev[i], elem, pt, type);
    for (int eq = 0; eq < neqs; ++eq) {
      m_R[i][eq] = 0.;
      m_xi[i][eq] = xi_vals[eq];
      m_xi_prev[i][eq] = xi_prev_vals[eq];
    }
  }
}

template <typename T>
void LocalResidual<T>::gather_aux(
    int pt,
    Array1D<apf::Field*> const& chi,
    Array1D<apf::Field*> const& chi_prev) {
  apf::MeshEntity* elem = apf::getMeshEntity(m_mesh_elem);
  for (int i = 0; i < m_num_aux_vars; ++i) {
    int const neqs = m_aux_var_num_eqs[i];
    int const type = m_aux_var_types[i];
    Array1D<double> const chi_vals =
      get_node_components(chi[i], elem, pt, type);
    Array1D<double> const chi_prev_vals =
      get_node_components(chi_prev[i], elem, pt, type);
    for (int eq = 0; eq < neqs; ++eq) {
      m_chi[i][eq] = chi_vals[eq];
      m_chi_prev[i][eq] = chi_prev_vals[eq];
    }
  }
}

template <typename T>
EVector LocalResidual<T>::compute_daux_dchi_diag(int step) {
  EVector daux_dchi_diag = EVector::Zero(1);
  return daux_dchi_diag;
}

template <typename T>
EVector LocalResidual<T>::compute_dlocal_dchi_prev_diag(int step) {
  EVector dlocal_dchi_prev_diag = EVector::Zero(1);
  return dlocal_dchi_prev_diag;
}

template <>
void LocalResidual<double>::scatter(int, Array1D<apf::Field*>&) {
}

template <>
void LocalResidual<FADT>::scatter(int pt, Array1D<apf::Field*>& xi) {
  apf::MeshEntity* elem = apf::getMeshEntity(m_mesh_elem);
  for (int i = 0; i < m_num_residuals; ++i) {
    int const type = m_var_types[i];
    Array1D<double> const xi_vals = get_components(m_xi[i], m_num_dims, type);
    apf::setComponents(xi[i], elem, pt, &(xi_vals[0]));
  }
}

template <typename T>
void LocalResidual<T>::scatter_aux(int pt, Array1D<apf::Field*>& chi) {
  apf::MeshEntity* elem = apf::getMeshEntity(m_mesh_elem);
  for (int i = 0; i < m_num_aux_vars; ++i) {
    int const type = m_aux_var_types[i];
    Array1D<double> const chi_vals = get_components(m_chi[i], m_num_dims, type);
    apf::setComponents(chi[i], elem, pt, &(chi_vals[0]));
  }
}

template <typename T>
void LocalResidual<T>::scatter_aux_prev(int pt, Array1D<apf::Field*>& chi_prev) {
  apf::MeshEntity* elem = apf::getMeshEntity(m_mesh_elem);
  for (int i = 0; i < m_num_aux_vars; ++i) {
    int const type = m_aux_var_types[i];
    Array1D<double> const chi_prev_vals = get_components(m_chi_prev[i], m_num_dims, type);
    apf::setComponents(chi_prev[i], elem, pt, &(chi_prev_vals[0]));
  }
}

template <typename T>
void LocalResidual<T>::scatter_adjoint(
    int pt,
    EVector const& phi_pt,
    Array1D<apf::Field*>& phi) {
  apf::MeshEntity* elem = apf::getMeshEntity(m_mesh_elem);
  for (int i = 0; i < m_num_residuals; ++i) {
    int const neqs = m_num_eqs[i];
    int const type = m_var_types[i];
    Array1D<FADT> phi_vals(m_num_eqs[i]);
    for (int eq = 0; eq < neqs; ++eq) {
      int idx = dxi_idx(i, eq);
      phi_vals[eq] = phi_pt[idx];
    }
    Array1D<double> const apf_phi = get_components(phi_vals, m_num_dims, type);
    apf::setComponents(phi[i], elem, pt, &(apf_phi[0]));
  }
}

template <typename T>
EVector LocalResidual<T>::gather_adjoint(
    int pt,
    Array1D<apf::Field*> const& phi) const {
  EVector phi_pt(m_num_dofs);
  apf::MeshEntity* elem = apf::getMeshEntity(m_mesh_elem);
  for (int i = 0; i < m_num_residuals; ++i) {
    int const neqs = m_num_eqs[i];
    int const type = m_var_types[i];
    Array1D<double> const phi_vals =
      get_node_components(phi[i], elem, pt, type);
    for (int eq = 0; eq < neqs; ++eq) {
      int idx = dxi_idx(i, eq);
      phi_pt[idx] = phi_vals[eq];
    }
  }
  return phi_pt;
}

template <typename T>
EVector LocalResidual<T>::gather_difference(
    int pt,
    Array1D<apf::Field*> const& xi_fine,
    Array1D<apf::Field*> const& xi) const {
  EVector diff(m_num_dofs);
  apf::MeshEntity* elem = apf::getMeshEntity(m_mesh_elem);
  for (int i = 0; i < m_num_residuals; ++i) {
    int const neqs = m_num_eqs[i];
    int const type = m_var_types[i];
    Array1D<double> const xi_fine_vals =
      get_node_components(xi_fine[i], elem, pt, type);
    Array1D<double> const xi_vals =
      get_node_components(xi[i], elem, pt, type);
    for (int eq = 0; eq < neqs; ++eq) {
      int idx = dxi_idx(i, eq);
      diff[idx] = xi_fine_vals[eq] - xi_vals[eq];
    }
  }
  return diff;
}

template <>
int LocalResidual<double>::seed_wrt_xi() {
  return -1;
}

template <>
int LocalResidual<FADT>::seed_wrt_xi() {
  for (int i = 0; i < m_num_residuals; ++i) {
    for (int eq = 0; eq < m_num_eqs[i]; ++eq) {
      int const dof = dxi_idx(i, eq);
      m_xi[i][eq].diff(dof, m_num_dofs);
    }
  }
  return m_num_dofs;
}

template <>
void LocalResidual<double>::unseed_wrt_xi() {
}

template <>
void LocalResidual<FADT>::unseed_wrt_xi() {
  for (int i = 0; i < m_num_residuals; ++i) {
    for (int eq = 0; eq < m_num_eqs[i]; ++eq) {
      m_xi[i][eq] = m_xi[i][eq].val();
      for (int idx = 0; idx < nmax_derivs; ++idx) {
        m_xi[i][eq].fastAccessDx(idx) = 0.;
        m_R[i][eq].fastAccessDx(idx) = 0.;
      }
    }
  }
}

template <>
int LocalResidual<double>::seed_wrt_xi_prev() {
  return -1;
}

template <>
int LocalResidual<FADT>::seed_wrt_xi_prev() {
  for (int i = 0; i < m_num_residuals; ++i) {
    for (int eq = 0; eq < m_num_eqs[i]; ++eq) {
      int const dof = dxi_idx(i, eq);
      m_xi_prev[i][eq].diff(dof, m_num_dofs);
    }
  }
  return m_num_dofs;
}

template <>
void LocalResidual<double>::unseed_wrt_xi_prev() {
}

template <>
void LocalResidual<FADT>::unseed_wrt_xi_prev() {
  for (int i = 0; i < m_num_residuals; ++i) {
    for (int eq = 0; eq < m_num_eqs[i]; ++eq) {
      m_xi_prev[i][eq] = m_xi_prev[i][eq].val();
      for (int idx = 0; idx < nmax_derivs; ++idx) {
        m_xi_prev[i][eq].fastAccessDx(idx) = 0.;
        m_R[i][eq].fastAccessDx(idx) = 0.;
      }
    }
  }
}

template <>
void LocalResidual<double>::seed_wrt_x(EMatrix const&) {
}

template <>
void LocalResidual<FADT>::seed_wrt_x(EMatrix const& dxi_dx) {
  DEBUG_ASSERT(dxi_dx.rows() == m_num_dofs);
  int const nglobal_dofs = dxi_dx.cols();
  for (int i = 0; i < m_num_residuals; ++i) {
    for (int eq = 0; eq < m_num_eqs[i]; ++eq) {
      int const xi_idx = dxi_idx(i, eq);
      double const val = m_xi[i][eq].val();
      m_xi[i][eq].diff(0, nglobal_dofs);
      m_xi[i][eq].val() = val;
      for (int x_idx = 0; x_idx < nglobal_dofs; ++x_idx) {
        m_xi[i][eq].fastAccessDx(x_idx) = dxi_dx(xi_idx, x_idx);
      }
    }
  }
}

template <>
int LocalResidual<double>::seed_wrt_params(int const es) {
  return -1;
}

template <>
int LocalResidual<FADT>::seed_wrt_params(int const es) {
  int const num_es_active_params = m_active_indices[es].size();
  for (int p = 0; p < num_es_active_params; ++p) {
    int const active_idx = m_active_indices[es][p];
    m_params[active_idx].diff(p, num_es_active_params);
  }
  return num_es_active_params;
}

template <>
void LocalResidual<double>::unseed_wrt_params(int const es) {
}

template <>
void LocalResidual<FADT>::unseed_wrt_params(int const es) {
  int const num_es_active_params = m_active_indices[es].size();
  for (int p = 0; p < num_es_active_params; ++p) {
    int const active_idx = m_active_indices[es][p];
    m_params[active_idx] = m_params[active_idx].val();
  }
  // for safety?
  for (int i = 0; i < m_num_residuals; ++i) {
    for (int eq = 0; eq < m_num_eqs[i]; ++eq) {
      for (int idx = 0; idx < nmax_derivs; ++idx) {
        m_R[i][eq].fastAccessDx(idx) = 0.;
      }
    }
  }
}

template <typename T>
void LocalResidual<T>::unset_elem() {
  m_mesh_elem = nullptr;
}

template <typename T>
void LocalResidual<T>::after_elems() {
  m_num_dims = -1;
  m_num_dofs = -1;
  m_shape = nullptr;
  m_xi.resize(0);
  m_xi_prev.resize(0);
  m_dxi_offsets.resize(0);
  m_num_aux_dofs = -1;
  m_chi.resize(0);
  m_chi_prev.resize(0);
  m_chi_offsets.resize(0);
}

template <typename T>
RCP<LocalResidual<T>> create_local_residual(
    ParameterList const& params,
    int ndims) {
  std::string const type = params.get<std::string>("type");
  if (type == "elastic") {
    return rcp(new Elastic<T>(params, ndims));
  } else if (type == "hyper_J2") {
    return rcp(new HyperJ2<T>(params, ndims));
  } else if (type == "hyper_J2_plane_strain") {
    return rcp(new HyperJ2PlaneStrain<T>(params, ndims));
  } else if (type == "hyper_J2_plane_stress") {
    return rcp(new HyperJ2PlaneStress<T>(params, ndims));
  } else if (type == "hypo_hill") {
    return rcp(new HypoHill<T>(params, ndims));
  } else if (type == "hypo_hill_plane_strain") {
    return rcp(new HypoHillPlaneStrain<T>(params, ndims));
  } else if (type == "hypo_hill_plane_stress") {
    return rcp(new HypoHillPlaneStress<T>(params, ndims));
  } else if (type == "isotropic_elastic") {
    return rcp(new IsotropicElastic<T>(params, ndims));
  } else if (type == "linear_thermoviscoelastic") {
    return rcp(new LTVE<T>(params, ndims));
  } else if (type == "small_hill") {
    return rcp(new SmallHill<T>(params, ndims));
  } else if (type == "small_hill_plane_strain") {
    return rcp(new SmallHillPlaneStrain<T>(params, ndims));
  } else if (type == "small_hill_plane_stress") {
    return rcp(new SmallHillPlaneStress<T>(params, ndims));
  } else if (type == "small_J2") {
    return rcp(new SmallJ2<T>(params, ndims));
  } else {
    fail("unknown local residual name: %s", type.c_str());
    return Teuchos::null;
  }
}

template class LocalResidual<double>;
template class LocalResidual<FADT>;

template RCP<LocalResidual<double>>
create_local_residual(ParameterList const&, int);

template RCP<LocalResidual<FADT>>
create_local_residual(ParameterList const&, int);

}
