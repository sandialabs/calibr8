#include "bcs.hpp"
#include "control.hpp"

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

void apply_resid_dbcs(
    ParameterList const& dbcs,
    RCP<Disc> disc,
    RCP<VectorT> x,
    RCP<VectorT> b,
    double t) {
  auto x_data = x->get1dViewNonConst();
  auto b_data = b->get1dViewNonConst();
  for (auto it = dbcs.begin(); it != dbcs.end(); ++it) {
    auto entry = dbcs.entry(it);
    auto a = Teuchos::getValue<Teuchos::Array<std::string>>(entry);
    int const eq = std::stoi(a[1]);
    std::string const set = a[2];
    strd::string const val = a[3];
    NodeSet const& nodes = disc->get_nodes(set);
    for (size_t node = 0; node < nodes.size(); ++node) {
      apf::Node const n = nodes[node];
      LO const row = disc->get_lid(node, eq);
      double const sol = x_data[row];
      double const v = get_val(disc, val, n, t);
      b_data[row] = sol - v;
    }
  }
}

}
