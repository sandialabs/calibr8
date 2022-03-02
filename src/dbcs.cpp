#include <apf.h>
#include <apfMesh2.h>
#include <apfNumbering.h>
#include "arrays.hpp"
#include "control.hpp"
#include "dbcs.hpp"
#include "disc.hpp"
#include "macros.hpp"

namespace calibr8 {

static double get_val(
    RCP<Disc> disc,
    std::string const& val,
    apf::Node const& node,
    double t) {
  apf::Vector3 x;
  apf::MeshEntity* e = node.entity;
  apf::Mesh* m = disc->apf_mesh();
  m->getPoint(e, 0, x);
  double v = eval(val, x[0], x[1], x[2], t);
  return v;
}

void apply_primal_dbcs(
    ParameterList const& dbcs,
    RCP<Disc> disc,
    RCP<MatrixT>& dR_dx,
    RCP<VectorT>& R,
    Array1D<apf::Field*>& x,
    double t,
    bool is_adjoint) {

  int const num_resids = disc->num_residuals();

  // sanity check
  DEBUG_ASSERT(x.size() == size_t(num_resids));

  // grab data from the blocked vectors
  Teuchos::ArrayRCP<double> R_data;
  R_data = R->get1dViewNonConst();

  // storage used below
  Teuchos::Array<double> entries;
  Teuchos::Array<LO> indices;
  Array1D<double> sol_comps(3);

  // loop through all the dirichlet boundary conditions
  for (auto it = dbcs.begin(); it != dbcs.end(); ++it) {

    // get the data for this specific dbc
    auto pentry = dbcs.entry(it);
    auto a = Teuchos::getValue<Teuchos::Array<std::string>>(pentry);
    int i = std::stoi(a[0]); // the residual index
    int eq = std::stoi(a[1]);
    std::string const set = a[2];
    std::string const val = a[3];
    NodeSet const& nodes = disc->nodes(set);

    // apply the dbcs to all nodes
    for (size_t node = 0; node < nodes.size(); ++node) {

      // get the current value of the solution for this resid/eq
      // and the intended specified DBC
      apf::Node n = nodes[node];
      LO const row = disc->get_lid(n, i, eq);
      apf::getComponents(x[i], n.entity, n.node, &(sol_comps[0]));
      double const sol = sol_comps[eq];
      double const v = get_val(disc, val, n, t);

      // get the local row in the matrix
      size_t num_cols = dR_dx->getNumEntriesInLocalRow(row);
      indices.resize(num_cols);
      entries.resize(num_cols);
      dR_dx->getLocalRowCopy(row, indices(), entries(), num_cols);

      double diag = 0.;
      for (size_t col = 0; col < num_cols; ++col) {
        if (indices[col] == row) {
          diag = entries[col];
        } else {
          entries[col] = 0.;
        }
      }

      dR_dx->replaceLocalValues(row, indices(), entries());

      if (!is_adjoint) {
        R_data[row] = diag * (sol - v);
      } else {
        R_data[row] = 0.;
      }

    }

  }

}

}
