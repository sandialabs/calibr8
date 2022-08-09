#include "bcs.hpp"
#include "control.hpp"
#include "linalg.hpp"
#include "physics.hpp"
#include "weights.hpp"

namespace calibr8 {

// this assumes from/to is either p1/p2
// apf did not implement projection for packed fields, which
// is a bit whack, so we use this method which is valid for transfers
// p1->p2 and p2->p1
static void project_from_to(RCP<Disc> disc, apf::Field* from, apf::Field* to) {
  ASSERT(apf::countComponents(from) == apf::countComponents(to));
  ASSERT(apf::getShape(from) != apf::getShape(to));
  int const neqs = apf::countComponents(from);
  int const space = disc->get_space(apf::getShape(to));
  apf::MeshEntity* vtx[2];
  std::vector<double> from_vals0(neqs, 0.);
  std::vector<double> from_vals1(neqs, 0);
  std::vector<double> to_vals(neqs, 0.);
  apf::DynamicArray<apf::Node> owned_nodes = disc->owned_nodes(space);
  for (apf::Node const& node : owned_nodes) {
    apf::MeshEntity* ent = node.entity;
    int const local_node = node.node;
    ASSERT(local_node == 0);
    int const ent_type = disc->apf_mesh()->getType(ent);
    if (ent_type == apf::Mesh::VERTEX) {
      apf::getComponents(from, ent, local_node, &(from_vals0[0]));
      apf::setComponents(to, ent, local_node, &(from_vals0[0]));
    }
    if (ent_type == apf::Mesh::EDGE) {
      disc->apf_mesh()->getDownward(ent, 0, vtx);
      apf::getComponents(from, vtx[0], local_node, &(from_vals0[0]));
      apf::getComponents(from, vtx[1], local_node, &(from_vals1[0]));
      for (int eq = 0; eq < neqs; ++eq) {
        to_vals[eq] = 0.5*(from_vals0[eq] + from_vals1[eq]);
      }
      apf::setComponents(to, ent, local_node, &(to_vals[0]));
    }
  }
  apf::synchronize(to);
}

void Fields::project_uH_onto_h(RCP<Disc> disc) {
  ASSERT(!uH_h);
  ASSERT(u[COARSE]);
  int const neqs = apf::countComponents(u[COARSE]);
  apf::Mesh* mesh = apf::getMesh(u[COARSE]);
  uH_h = apf::createPackedField(mesh, "uH_h", neqs);
  apf::zeroField(uH_h);
  project_from_to(disc, u[COARSE], uH_h);
}

void Fields::destroy() {
  for (int space = 0; space < NUM_SPACE; ++space) {
    if (u[space]) apf::destroyField(u[space]);
    if (z[space]) apf::destroyField(z[space]);
    u[space] = nullptr;
    z[space] = nullptr;
  }
}

template <typename T>
void assemble(
    int space,
    int mode,
    RCP<Disc> disc,
    RCP<Residual<T>> r,
    RCP<Weight> weight,
    RCP<VectorT> U,
    System& sys) {
  r->set_space(space);
  r->set_mode(mode);
  apf::Mesh2* mesh = disc->apf_mesh();
  int order = 6;
  for (int es = 0; es < disc->num_elem_sets(); ++es) {
    std::string es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);
    for (size_t elem = 0; elem < elems.size(); ++elem) {
      apf::MeshElement* me = apf::createMeshElement(mesh, elems[elem]);
      r->in_elem(me, disc);
      r->gather(disc, U);
      int const npts = apf::countIntPoints(me, order);
      for (int pt = 0; pt < npts; ++pt) {
        apf::Vector3 xi;
        apf::getIntPoint(me, order, pt, xi);
        double const w = apf::getIntWeight(me, order, pt);
        double const dv = apf::getDV(me, xi);
        r->at_point(xi, w, dv, weight, disc);
      }
      r->scatter(disc, sys);
      r->out_elem();
    }
  }
  // apply tbcs here
  r->set_space(-1);
  r->set_mode(-1);
}

static void fill_field(
    int space,
    RCP<Disc> disc,
    RCP<VectorT> x,
    apf::Field* f) {
  int const neqs = apf::countComponents(f);
  apf::Mesh2* mesh = disc->apf_mesh();
  auto x_data = x->get1dView();
  std::vector<double> vals(neqs, 0);
  apf::DynamicArray<apf::Node> nodes = disc->owned_nodes(space);
  for (auto& node : nodes) {
    apf::MeshEntity* ent = node.entity;
    int const local_node = node.node;
    for (int eq = 0; eq < neqs; ++eq) {
      LO const row = disc->get_lid(space, node, eq);
      vals[eq] = x_data[row];
    }
    apf::setComponents(f, ent, local_node, &(vals[0]));
  }
  apf::synchronize(f);
}

static void fill_vector(
    int space,
    RCP<Disc> disc,
    apf::Field* f,
    RCP<VectorT> x) {
  int const neqs = apf::countComponents(f);
  apf::Mesh2* meshh = disc->apf_mesh();
  auto x_data = x->get1dViewNonConst();
  std::vector<double> vals(neqs, 0);
  apf::DynamicArray<apf::Node> nodes = disc->owned_nodes(space);
  for (auto& node : nodes) {
    apf::MeshEntity* ent = node.entity;
    int const local_node = node.node;
    apf::getComponents(f, ent, local_node, &(vals[0]));
    for (int eq = 0; eq < neqs; ++eq) {
      LO const row = disc->get_lid(space, node, eq);
      x_data[row] = vals[eq];
    }
  }
}

apf::Field* solve_primal(
    int space,
    RCP<ParameterList> params,
    RCP<Disc> disc,
    RCP<Residual<double>> residual,
    RCP<Residual<FADT>> jacobian) {

  print("---------");
  print("primal %s", disc->space_name(space).c_str());
  print("---------");

  apf::Mesh2* mesh = disc->apf_mesh();
  apf::FieldShape* shape = disc->shape(space);
  mesh->changeShape(shape, true);

  Vector U(space, disc);
  Vector dU(space, disc);
  Vector R(space, disc);
  Matrix dRdU(space, disc);
  System ghost_sys(GHOST, dRdU, dU, R);
  System owned_sys(OWNED, dRdU, dU, R);

  RCP<Weight> weight = rcp(new Weight(disc->shape(space)));
  ParameterList const dbcs = params->sublist("dbcs");
  ParameterList& lin_alg = params->sublist("linear algebra");
  ParameterList& newton = params->sublist("newton solve");

  U.zero();

  int iter = 1;
  bool converged = false;
  int const max_iters = newton.get<int>("max iters");
  double const tolerance = newton.get<double>("tolerance");

  while ((iter <= max_iters) && (!converged)) {

    print(" > (%d) Newton iteration", iter);

    dRdU.begin_fill();
    dU.zero();
    R.zero();
    dRdU.zero();

    assemble(space, JACOBIAN, disc, jacobian, weight, U.val[GHOST], ghost_sys);
    dRdU.gather(Tpetra::ADD);
    R.gather(Tpetra::ADD);
    R.val[OWNED]->scale(-1.0);
    apply_jacob_dbcs(dbcs, space, disc, U.val[OWNED], owned_sys, false);
    dRdU.end_fill();

    solve(lin_alg, space, disc, owned_sys);
    U.val[OWNED]->update(1.0, *(dU.val[OWNED]), 1.0);
    U.scatter(Tpetra::INSERT);

    R.zero();
    assemble(space, RESIDUAL, disc, residual, weight, U.val[GHOST], ghost_sys);
    R.gather(Tpetra::ADD);
    apply_resid_dbcs(dbcs, space, disc, U.val[OWNED], owned_sys);
    double const R_norm = R.val[OWNED]->norm2();
    print(" ||R|| = %e", R_norm);
    if (R_norm < tolerance) converged = true;

    iter++;

  }

  std::string const name = "u" + disc->space_name(space);
  int const neqs = residual->num_eqs();
  apf::Field* f = apf::createPackedField(mesh, name.c_str(), neqs);
  apf::zeroField(f);
  fill_field(space, disc, U.val[OWNED], f);
  return f;

}

}
