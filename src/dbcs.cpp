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

void apply_expression_primal_dbcs(
    ParameterList const& dbcs,
    RCP<Disc> disc,
    Array2D<RCP<MatrixT>>& dR_dx,
    Array1D<RCP<VectorT>>& R,
    Array1D<apf::Field*>& x,
    double t,
    bool is_adjoint) {

  int const num_resids = disc->num_residuals();

  // sanity check
  DEBUG_ASSERT(dR_dx.size() == size_t(num_resids));
  DEBUG_ASSERT(R.size() == size_t(num_resids));
  DEBUG_ASSERT(x.size() == size_t(num_resids));

  // grab data from the blocked vectors
  Array1D<Teuchos::ArrayRCP<double>> R_data(num_resids);
  for (int i = 0; i < num_resids; ++i) {
    R_data[i] = R[i]->get1dViewNonConst();
  }

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

      // loop over the blocks in the column direction
      for (int j = 0; j < num_resids; ++j) {

        // get the local row in the matrix
        size_t num_cols = dR_dx[i][j]->getNumEntriesInLocalRow(row);
        indices.resize(num_cols);
        entries.resize(num_cols);
        dR_dx[i][j]->getLocalRowCopy(row, indices(), entries(), num_cols);

        // if we are at the diagonal block
        if (i == j) {
          double diag = 0.;
          for (size_t col = 0; col < num_cols; ++col) {
            if (indices[col] == row) {
              diag = entries[col];
            } else {
              entries[col] = 0.;
            }
          }
          dR_dx[i][j]->replaceLocalValues(row, indices(), entries());
          if (!is_adjoint) {
            R_data[i][row] = diag * (sol - v);
          } else {
            R_data[i][row] = 0.;
          }
        }

        // if we are on an off-diagonal block
        else {
          for (size_t col = 0; col < num_cols; ++col) {
            entries[col] = 0;
            dR_dx[i][j]->replaceLocalValues(row, indices(), entries());
          }
        }

      }

    }

  }

}

void apply_field_primal_dbcs(
    ParameterList const& dbcs,
    RCP<Disc> disc,
    Array2D<RCP<MatrixT>>& dR_dx,
    Array1D<RCP<VectorT>>& R,
    Array1D<apf::Field*>& x,
    double t,
    int step,
    std::string const& prefix,
    bool is_adjoint) {

  int const num_resids = disc->num_residuals();

  // sanity check
  DEBUG_ASSERT(dR_dx.size() == size_t(num_resids));
  DEBUG_ASSERT(R.size() == size_t(num_resids));
  DEBUG_ASSERT(x.size() == size_t(num_resids));

  // grab data from the blocked vectors
  Array1D<Teuchos::ArrayRCP<double>> R_data(num_resids);
  for (int i = 0; i < num_resids; ++i) {
    R_data[i] = R[i]->get1dViewNonConst();
  }

  // storage used below
  Teuchos::Array<double> entries;
  Teuchos::Array<LO> indices;
  Array1D<double> sol_comps(3);

  apf::Mesh* m = disc->apf_mesh();

  // loop through all the dirichlet boundary conditions
  for (auto it = dbcs.begin(); it != dbcs.end(); ++it) {

    // get the data for this specific dbc
    auto pentry = dbcs.entry(it);
    auto a = Teuchos::getValue<Teuchos::Array<std::string>>(pentry);
    int i = std::stoi(a[0]); // the residual index
    int eq = std::stoi(a[1]);
    std::string const set = a[2];
    NodeSet const& nodes = disc->nodes(set);

    std::string name = prefix + "_" + std::to_string(step);
    apf::Field* u_meas = m->findField(name.c_str());
    ALWAYS_ASSERT(u_meas);

    // apply the dbcs to all nodes
    for (size_t node = 0; node < nodes.size(); ++node) {

      // get the current value of the solution for this resid/eq
      // and the intended specified DBC
      apf::Node n = nodes[node];
      LO const row = disc->get_lid(n, i, eq);
      apf::getComponents(x[i], n.entity, n.node, &(sol_comps[0]));
      double const sol = sol_comps[eq];

      apf::Vector3 x;
      apf::MeshEntity* e = n.entity;
      m->getPoint(e, 0, x);

      apf::Vector3 uval;
      apf::getVector(u_meas, n.entity, n.node, uval);
      double const v = uval[eq];

      // loop over the blocks in the column direction
      for (int j = 0; j < num_resids; ++j) {

        // get the local row in the matrix
        size_t num_cols = dR_dx[i][j]->getNumEntriesInLocalRow(row);
        indices.resize(num_cols);
        entries.resize(num_cols);
        dR_dx[i][j]->getLocalRowCopy(row, indices(), entries(), num_cols);

        // if we are at the diagonal block
        if (i == j) {
          double diag = 0.;
          for (size_t col = 0; col < num_cols; ++col) {
            if (indices[col] == row) {
              diag = entries[col];
            } else {
              entries[col] = 0.;
            }
          }
          dR_dx[i][j]->replaceLocalValues(row, indices(), entries());
          if (!is_adjoint) {
            R_data[i][row] = diag * (sol - v);
          } else {
            R_data[i][row] = 0.;
          }
        }

        // if we are on an off-diagonal block
        else {
          for (size_t col = 0; col < num_cols; ++col) {
            entries[col] = 0;
            dR_dx[i][j]->replaceLocalValues(row, indices(), entries());
          }
        }

      }

    }

  }
}

void apply_primal_dbcs(
    ParameterList const& dbcs,
    RCP<Disc> disc,
    Array2D<RCP<MatrixT>>& dR_dx,
    Array1D<RCP<VectorT>>& R,
    Array1D<apf::Field*>& x,
    double t,
    int step,
    bool is_adjoint) {

  if (dbcs.isSublist("expression")) {
    auto expression_dbcs = dbcs.sublist("expression");
    apply_expression_primal_dbcs(expression_dbcs, disc, dR_dx, R, x, t,
        is_adjoint);
  }

  else if (dbcs.isSublist("field")) {
    auto field_dbcs = dbcs.sublist("field");
    apply_field_primal_dbcs(field_dbcs, disc, dR_dx, R, x, t, step, "measured",
        is_adjoint);
  }

  else {
    throw std::runtime_error("apply_primal dbcs: no dbcs specified!");
  }

}

void apply_adjoint_dbcs(
    ParameterList const& full_dbcs,
    RCP<Disc> disc,
    Array1D<apf::Field*>& z) {

  ParameterList dbcs;
  if (full_dbcs.isSublist("expression")) {
    dbcs = full_dbcs.sublist("expression");
  } else if (full_dbcs.isSublist("field")) {
    dbcs = full_dbcs.sublist("field");
  }

  int const num_resids = disc->num_residuals();

  // sanity check
  DEBUG_ASSERT(z.size() == size_t(num_resids));

  // storage used below
  Teuchos::Array<double> entries;
  Teuchos::Array<LO> indices;
  Array1D<double> sol_comps(3);

  apf::Mesh* m = disc->apf_mesh();

  // loop through all the dirichlet boundary conditions
  for (auto it = dbcs.begin(); it != dbcs.end(); ++it) {

    // get the data for this specific dbc
    auto pentry = dbcs.entry(it);
    auto a = Teuchos::getValue<Teuchos::Array<std::string>>(pentry);
    int i = std::stoi(a[0]); // the residual index
    int eq = std::stoi(a[1]);
    std::string const set = a[2];
    NodeSet const& nodes = disc->nodes(set);

    // apply the dbcs to all nodes
    for (size_t node = 0; node < nodes.size(); ++node) {

      // get the current value of the solution for this resid/eq
      // and the intended specified DBC and set it to zero
      apf::Node n = nodes[node];
      LO const row = disc->get_lid(n, i, eq);
      apf::getComponents(z[i], n.entity, n.node, &(sol_comps[0]));
      sol_comps[eq] = 0.;
      apf::setComponents(z[i], n.entity, n.node, &(sol_comps[0]));

    }
  }
}

}
