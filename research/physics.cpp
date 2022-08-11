#include <PCU.h>
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

apf::Field* project(RCP<Disc> disc, apf::Field* from, std::string const& name) {
  ASSERT(apf::getMesh(from) == disc->apf_mesh());
  apf::FieldShape* from_shape = apf::getShape(from);
  int const from_space = disc->get_space(from_shape);
  int const to_space = (from_space == COARSE) ? FINE : COARSE;
  int const neqs = apf::countComponents(from);
  apf::Mesh* mesh = apf::getMesh(from);
  apf::FieldShape* to_shape = disc->shape(to_space);
  apf::Field* to = apf::createPackedField(mesh, name.c_str(), neqs, to_shape);
  apf::zeroField(to);
  project_from_to(disc, from, to);
  return to;
}

apf::Field* subtract(RCP<Disc> disc, apf::Field* a, apf::Field* b, std::string const& name) {
  ASSERT(apf::getMesh(a) == apf::getMesh(b));
  ASSERT(apf::getMesh(b) == disc->apf_mesh());
  ASSERT(apf::getShape(a) == apf::getShape(b));
  ASSERT(apf::countComponents(a) == apf::countComponents(b));
  apf::Mesh* mesh = apf::getMesh(a);
  apf::FieldShape* shape = apf::getShape(a);
  int const space = disc->get_space(shape);
  int const neqs = apf::countComponents(a);
  apf::Field* diff = apf::createPackedField(mesh, name.c_str(), neqs, shape);
  apf::DynamicArray<apf::Node> nodes = disc->owned_nodes(space);
  std::vector<double> a_vals(neqs, 0.);
  std::vector<double> b_vals(neqs, 0.);
  std::vector<double> diff_vals(neqs, 0.);
  for (auto& node : nodes) {
    apf::MeshEntity* ent = node.entity;
    int const local_node = node.node;
    apf::getComponents(a, ent, local_node, &(a_vals[0]));
    apf::getComponents(b, ent, local_node, &(b_vals[0]));
    for (int eq = 0; eq < neqs; ++eq) {
      diff_vals[eq] = a_vals[eq] - b_vals[eq];
    }
    apf::setComponents(diff, ent, local_node, &(diff_vals[0]));
  }
  apf::synchronize(diff);
  return diff;
}

void Fields::destroy() {
  for (int space = 0; space < NUM_SPACE; ++space) {
    if (u[space]) apf::destroyField(u[space]);
    if (z[space]) apf::destroyField(z[space]);
    u[space] = nullptr;
    z[space] = nullptr;
  }
  if (uH_h) apf::destroyField(uH_h);
  if (zH_h) apf::destroyField(zH_h);
  if (uh_minus_uH_h) apf::destroyField(uh_minus_uH_h);
  if (zh_minus_zH_h) apf::destroyField(zh_minus_zH_h);
  if (E_L) apf::destroyField(E_L);
  uH_h = nullptr;
  zH_h = nullptr;
  uh_minus_uH_h = nullptr;
  zh_minus_zH_h = nullptr;
  E_L = nullptr;
}

template <typename T>
void assemble_residual(
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

template <typename T>
void assemble_qoi(
    int space,
    RCP<Disc> disc,
    RCP<Residual<T>> r,
    RCP<QoI<T>> qoi,
    RCP<VectorT> U,
    System* sys) {
  r->set_space(space);
  qoi->set_space(space);
  apf::Mesh2* mesh = disc->apf_mesh();
  int order = 6;
  for (int es = 0; es < disc->num_elem_sets(); ++es) {
    std::string es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);
    for (size_t elem = 0; elem < elems.size(); ++elem) {
      apf::MeshElement* me = apf::createMeshElement(mesh, elems[elem]);
      r->in_elem(me, disc);
      qoi->in_elem(me, r, disc);
      r->gather(disc, U);
      int const npts = apf::countIntPoints(me, order);
      for (int pt = 0; pt < npts; ++pt) {
        apf::Vector3 xi;
        apf::getIntPoint(me, order, pt, xi);
        double const w = apf::getIntWeight(me, order, pt);
        double const dv = apf::getDV(me, xi);
        qoi->at_point(xi, w, dv, r, disc);
      }
      qoi->scatter(disc, sys);
      qoi->out_elem();
    }
  }
  qoi->post(disc, sys);
  r->set_space(-1);
  qoi->set_space(-1);
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
    Vector& x) {
  int const neqs = apf::countComponents(f);
  apf::Mesh2* meshh = disc->apf_mesh();
  auto x_data = x.val[OWNED]->get1dViewNonConst();
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
  x.scatter(Tpetra::INSERT);
}

apf::Field* solve_primal(
    int space,
    RCP<ParameterList> params,
    RCP<Disc> disc,
    RCP<Residual<double>> residual,
    RCP<Residual<FADT>> jacobian) {

  apf::Mesh2* mesh = disc->apf_mesh();
  apf::FieldShape* shape = disc->shape(space);
  mesh->changeShape(shape, true);

  Vector U(space, disc);
  Vector dU(space, disc);
  Vector R(space, disc);
  Matrix dRdU(space, disc);
  System ghost_sys(GHOST, dRdU, dU, R);
  System owned_sys(OWNED, dRdU, dU, R);

  RCP<Weight> weight = rcp(new Weight(shape));
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

    assemble_residual(space, JACOBIAN, disc, jacobian, weight, U.val[GHOST], ghost_sys);
    dRdU.gather(Tpetra::ADD);
    R.gather(Tpetra::ADD);
    R.val[OWNED]->scale(-1.0);
    apply_jacob_dbcs(dbcs, space, disc, U.val[OWNED], owned_sys, false);
    dRdU.end_fill();

    solve(lin_alg, space, disc, owned_sys);
    U.val[OWNED]->update(1.0, *(dU.val[OWNED]), 1.0);
    U.scatter(Tpetra::INSERT);

    R.zero();
    assemble_residual(space, RESIDUAL, disc, residual, weight, U.val[GHOST], ghost_sys);
    R.gather(Tpetra::ADD);
    apply_resid_dbcs(dbcs, space, disc, U.val[OWNED], owned_sys);
    double const R_norm = R.val[OWNED]->norm2();
    print(" ||R|| = %e", R_norm);
    if (R_norm < tolerance) converged = true;

    iter++;

  }

  std::string const name = "u" + disc->space_name(space);
  int const neqs = residual->num_eqs();
  apf::Field* f = apf::createPackedField(mesh, name.c_str(), neqs, shape);
  apf::zeroField(f);
  fill_field(space, disc, U.val[OWNED], f);
  return f;

}

double compute_qoi(
    int space,
    RCP<ParameterList> params,
    RCP<Disc> disc,
    RCP<Residual<double>> resid,
    RCP<QoI<double>> qoi,
    apf::Field* u_space) {

  apf::Mesh2* mesh = disc->apf_mesh();
  apf::FieldShape* shape = disc->shape(space);
  mesh->changeShape(shape, true);

  qoi->reset();

  Vector U(space, disc);
  fill_vector(space, disc, u_space, U);
  assemble_qoi(space, disc, resid, qoi, U.val[GHOST], nullptr);

  double J = qoi->value();
  J = PCU_Add_Double(J);
  std::string const name = "J" + disc->space_name(space);
  print("%s = %.15e", name.c_str(), J);

  return J;

}

apf::Field* compute_linearization_error(
    RCP<ParameterList> params,
    RCP<Disc> disc,
    RCP<Residual<double>> resid,
    RCP<Residual<FADT>> jacobian,
    apf::Field* uH_h,
    apf::Field* uh_minus_uH_h) {

  apf::Mesh2* mesh = disc->apf_mesh();
  apf::FieldShape* shape = disc->shape(FINE);
  mesh->changeShape(shape, true);

  Vector U(FINE, disc);
  Vector R(FINE, disc);
  Vector E(FINE, disc);
  Vector U_diff(FINE, disc);
  Matrix dRdU(FINE, disc);
  System ghost_sys(GHOST, dRdU, R, R);
  System owned_sys(OWNED, dRdU, R, R);

  RCP<Weight> weight = rcp(new Weight(shape));
  ParameterList const dbcs = params->sublist("dbcs");

  dRdU.begin_fill();
  R.zero();
  dRdU.zero();
  fill_vector(FINE, disc, uH_h, U);
  fill_vector(FINE, disc, uh_minus_uH_h, U_diff);

  assemble_residual(FINE, JACOBIAN, disc, jacobian, weight, U.val[GHOST], ghost_sys);
  R.gather(Tpetra::ADD);
  dRdU.gather(Tpetra::ADD);
  apply_jacob_dbcs(dbcs, FINE, disc, U.val[OWNED], owned_sys, false);
  dRdU.end_fill();

  dRdU.val[OWNED]->apply(*(U_diff.val[OWNED]), *(E.val[OWNED]));
  E.val[OWNED]->update(-1.0, *(R.val[OWNED]), -1.0);

  double const R_norm = R.val[OWNED]->norm2();
  double const E_norm = E.val[OWNED]->norm2();
  print(" > ||R|| = %.15e", R_norm);
  print(" > ||E_L|| = %.15e", E_norm);

  int const neqs = jacobian->num_eqs();
  apf::Field* f = apf::createPackedField(mesh, "E_L", neqs, shape);
  apf::zeroField(f);
  fill_field(FINE, disc, E.val[OWNED], f);
  return f;

}

}
