#include "disc.hpp"
#include "weights.hpp"

namespace calibr8 {

static void resize(Array1D<double>& v, int ni) {
  v.resize(ni);
}

static void resize(Array2D<double>& v, int ni, int nj) {
  v.resize(ni);
  for (int i = 0; i < ni; ++i) {
    v[i].resize(nj);
  }
}

static void resize(Array3D<double>& v, int ni, int nj, int nk) {
  v.resize(ni);
  for (int i = 0; i < ni; ++i) {
    v[i].resize(nj);
    for (int j = 0; j < nj; ++j) {
      v[j].resize(nk);
    }
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

Weight::Weight(apf::FieldShape* shape) {
  m_shape = shape;
}

Weight::~Weight() {
}

void Weight::in_elem(apf::MeshElement* me, RCP<Disc>) {
  m_mesh_elem = me;
}

void Weight::evaluate(apf::Vector3 const& xi) {
  apf::getBF(m_shape, m_mesh_elem, xi, m_BF);
  apf::getGradBF(m_shape, m_mesh_elem, xi, m_gBF);
}

double Weight::val(int node, int eq) {
  (void)eq;
  return m_BF[node];
}

double Weight::grad(int node, int eq, int dim) {
  return m_gBF[node][dim];
}

void Weight::out_elem() {
  m_mesh_elem = nullptr;
}

AdjointWeight::AdjointWeight(apf::FieldShape* shape) :
  Weight(shape) {
}

AdjointWeight::~AdjointWeight() {
}

void AdjointWeight::in_elem(apf::MeshElement* me, RCP<Disc> disc) {
  m_mesh_elem = me;
  apf::MeshEntity* ent = apf::getMeshEntity(me);
  m_space = disc->get_space(m_shape);
  m_neqs = disc->num_eqs();
  m_nnodes = disc->get_num_nodes(m_space, ent);
  m_ndims = disc->num_dims();
  resize(m_z_vals, m_nnodes, m_neqs);
  resize(m_vals, m_nnodes, m_neqs);
  resize(m_grads, m_nnodes, m_neqs, m_ndims);
}

void AdjointWeight::gather(RCP<Disc> disc, RCP<VectorT> Z) {
  apf::MeshEntity* ent = apf::getMeshEntity(m_mesh_elem);
  auto z_data = Z->get1dView();
  for (int node = 0; node < m_nnodes; ++node) {
    for (int eq = 0; eq < m_neqs; ++eq) {
      LO const row = disc->get_lid(m_space, ent, node, eq);
      m_z_vals[node][eq] = z_data[row];
    }
  }
}

Array1D<double> AdjointWeight::interp_z(apf::Vector3 const& xi) {
  Array1D<double> z(m_neqs, 0.);
  apf::getBF(m_shape, m_mesh_elem, xi, m_BF);
  for (int node = 0; node < m_nnodes; ++node) {
    for (int eq = 0; eq < m_neqs; ++eq) {
      z[eq] += m_z_vals[node][eq] * m_BF[node];
    }
  }
  return z;
}

Array2D<double> AdjointWeight::interp_grad_z(apf::Vector3 const& xi) {
  Array2D<double> grad_z;
  resize(grad_z, m_neqs, m_ndims);
  zero(grad_z);
  apf::getGradBF(m_shape, m_mesh_elem, xi, m_gBF);
  for (int node = 0; node < m_nnodes; ++node) {
    for (int eq = 0; eq < m_neqs; ++eq) {
      for (int dim = 0; dim < m_ndims; ++dim) {
        grad_z[eq][dim] += m_z_vals[node][eq] * m_gBF[node][dim];
      }
    }
  }
  return grad_z;
}

void AdjointWeight::evaluate(apf::Vector3 const& xi) {
  Array1D<double> z = interp_z(xi);
  Array2D<double> grad_z = interp_grad_z(xi);
  for (int node = 0; node < m_nnodes; ++node) {
    for (int eq = 0; eq < m_neqs; ++eq) {
      m_vals[node][eq] = z[eq] * m_BF[node];
      for (int dim = 0; dim < m_ndims; ++dim) {
        m_grads[node][eq][dim] =
          grad_z[eq][dim] * m_BF[node] + z[eq] * m_gBF[node][dim];
      }
    }
  }
}

double AdjointWeight::val(int node, int eq) {
  return m_vals[node][eq];
}

double AdjointWeight::grad(int node, int eq, int dim) {
  return m_grads[node][eq][dim];
}

void AdjointWeight::out_elem() {
  m_mesh_elem = nullptr;
  m_nnodes = -1;
  m_ndims = -1;
  m_neqs = -1;
}

}
