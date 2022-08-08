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

void apply_jacob_dbcs(
    ParameterList const& params,
    int space,
    RCP<Disc> disc,
    System& sys,
    bool adjoint) {
  auto x_data = sys.x->get1dViewNonConst();
  auto b_data = sys.x->get1dViewNonConst();
  Teuchos::Array<double> entries;
  Teuchos::Array<int> indices;
  for (auto it = params.begin(); it != params.end(); ++it) {
    DBC const dbc = get_dbc(params, it);
    NodeSet const& nodes = disc->nodes(space, dbc.set);
    for (size_t node = 0; node < nodes.size(); ++node) {
      apf::Node const n = nodes[node];
      LO const row = disc->get_lid(space, n, dbc.eq);
      double const sol = x_data[row];
      double const v = get_val(disc, dbc.val, n);
      size_t ncols = sys.A->getNumEntriesInLocalRow(row);
      indices.resize(ncols);
      entries.resize(ncols);
      sys.A->getLocalRowCopy(row, indices(), entries(), ncols);
      double diag = 0.;
      for (size_t col = 0; col < ncols; ++col) {
        if (indices[col] == row) diag = entries[col];
        else entries[col] = 0.;
      }
      sys.A->replaceLocalValues(row, indices(), entries());
      if (!adjoint) b_data[row] = diag*(sol-v);
      else b_data[row] = 0.;
    }
  }
}

}
