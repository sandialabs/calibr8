#include "bcs.hpp"
#include "control.hpp"

namespace calibr8 {

static double get_val(
    RCP<Disc> disc,
    std::string const& val,
    apf::Node const& node) {
  apf::Vector3 x;
  apf::MeshEntity* e = node.entity;
  apf::Mesh* m = disc->apf_mesh();
  m->getPoint(e, 0, x);
  double v = eval(val, x[0], x[1], x[2], 0.);
  return v;
}

struct DBC {
  int eq = -1;
  std::string set = "";
  std::string val = "";
};

template <class T>
DBC get_dbc(ParameterList const& dbcs, T const& it) {
  DBC r;
  auto entry = dbcs.entry(it);
  auto a = Teuchos::getValue<Teuchos::Array<std::string>>(entry);
  r.eq = std::stoi(a[0]);
  r.set = a[1];
  r.val = a[2];
  return r;
}

void apply_resid_dbcs(
    ParameterList const& params,
    int space,
    RCP<Disc> disc,
    RCP<VectorT> u,
    System& sys) {
  auto b_data = sys.b->get1dViewNonConst();
  auto u_data = u->get1dView();
  for (auto it = params.begin(); it != params.end(); ++it) {
    DBC const dbc = get_dbc(params, it);
    NodeSet const& nodes = disc->nodes(space, dbc.set);
    for (apf::Node const& node : nodes) {
      LO const row = disc->get_lid(space, node, dbc.eq);
      double const v = get_val(disc, dbc.val, node);
      double const sol = u_data[row];
      b_data[row] = sol - v;
    }
  }
}

using indices_type = typename MatrixT::nonconst_local_inds_host_view_type;
using entries_type = typename MatrixT::nonconst_values_host_view_type;

void apply_jacob_dbcs(
    ParameterList const& params,
    int space,
    RCP<Disc> disc,
    RCP<VectorT> u,
    System& sys,
    bool adjoint) {
  auto x_data = sys.x->get1dViewNonConst();
  auto b_data = sys.b->get1dViewNonConst();
  auto u_data = u->get1dView();
  Teuchos::Array<double> entries;
  Teuchos::Array<int> indices;
  for (auto it = params.begin(); it != params.end(); ++it) {
    DBC const dbc = get_dbc(params, it);
    NodeSet const& nodes = disc->nodes(space, dbc.set);
    for (apf::Node const& node : nodes) {
      LO const row = disc->get_lid(space, node, dbc.eq);
      double const sol = u_data[row];
      double const v = get_val(disc, dbc.val, node);
      size_t ncols = sys.A->getNumEntriesInLocalRow(row);
      indices.resize(ncols);
      entries.resize(ncols);
      indices_type indices_view(indices.getRawPtr(), ncols);
      entries_type entries_view(entries.getRawPtr(), ncols);
      sys.A->getLocalRowCopy(row, indices_view, entries_view, ncols);
      for (size_t col = 0; col < ncols; ++col) {
        if (indices[col] == row) entries[col] = 1.0;
        else entries[col] = 0.;
      }
      sys.A->replaceLocalValues(row, indices(), entries());
      if (!adjoint) b_data[row] = sol - v;
      else b_data[row] = 0.;
    }
  }
}

}
