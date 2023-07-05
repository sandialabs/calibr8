#include <PCU.h>
#include <apfDynamicMatrix.h>
#include "bcs.hpp"
#include "control.hpp"
#include "cspr.hpp"
#include "disc.hpp"
#include "linalg.hpp"
#include "residual.hpp"
#include "physics.hpp"

namespace calibr8 {

static apf::Field* interpolate_to_ips(apf::Field* nodal, std::string const& name) {
  apf::Mesh* m = apf::getMesh(nodal);
  int const neqs = apf::countComponents(nodal);
  int const dim = m->getDimension();
  int const q_order = apf::getShape(nodal)->getOrder() + 1;
  apf::FieldShape* ip_shape = apf::getIPShape(dim, q_order);
  apf::Field* ip = apf::createPackedField(m, name.c_str(), neqs, ip_shape);
  apf::zeroField(ip);
  apf::MeshEntity* elem;
  apf::MeshIterator* elems = m->begin(dim);
  Array1D<double> field_comps(9);
  while ((elem = m->iterate(elems))) {
    apf::MeshElement* me = apf::createMeshElement(m, elem);
    apf::Element* nodal_elem = apf::createElement(nodal, me);
    int const npts = apf::countIntPoints(me, q_order);
    for (int pt = 0; pt < npts; ++pt) {
      apf::Vector3 iota;
      apf::getIntPoint(me, q_order, pt, iota);
      apf::getComponents(nodal_elem, iota, &(field_comps[0]));
      apf::setComponents(ip, elem, pt, &(field_comps[0]));
    }
    apf::destroyElement(nodal_elem);
    apf::destroyMeshElement(me);
  }
  m->end(elems);
  return ip;
}

static void project_from_to(RCP<Disc> disc, apf::Field* from, apf::Field* to) {
  ASSERT(apf::countComponents(from) == apf::countComponents(to));
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

static double add(double a, double b) { return a+b; }
static double subtract(double a, double b) { return a-b; }
static double negate_multiply(double a, double b) { return -a*b; }

static void sum_into(double& a, double b) { a += b; }
static void abs_sum_into(double& a, double b) { a += std::abs(b); }

apf::Field* op(
    std::function<double(double,double)> f,
    RCP<Disc> disc,
    apf::Field* a,
    apf::Field* b,
    std::string const& name) {
  ASSERT(apf::getMesh(a) == apf::getMesh(b));
  ASSERT(apf::getMesh(b) == disc->apf_mesh());
  ASSERT(apf::getShape(a) == apf::getShape(b));
  ASSERT(apf::countComponents(a) == apf::countComponents(b));
  apf::Mesh* mesh = apf::getMesh(a);
  apf::FieldShape* shape = apf::getShape(a);
  int const space = disc->get_space(shape);
  int const neqs = apf::countComponents(a);
  apf::Field* result = apf::createPackedField(mesh, name.c_str(), neqs, shape);
  apf::DynamicArray<apf::Node> nodes = disc->owned_nodes(space);
  std::vector<double> a_vals(neqs, 0.);
  std::vector<double> b_vals(neqs, 0.);
  std::vector<double> result_vals(neqs, 0.);
  for (auto& node : nodes) {
    apf::MeshEntity* ent = node.entity;
    int const local_node = node.node;
    apf::getComponents(a, ent, local_node, &(a_vals[0]));
    apf::getComponents(b, ent, local_node, &(b_vals[0]));
    for (int eq = 0; eq < neqs; ++eq) {
      result_vals[eq] = f(a_vals[eq], b_vals[eq]);
    }
    apf::setComponents(result, ent, local_node, &(result_vals[0]));
  }
  apf::synchronize(result);
  return result;
}

double op(
    std::function<void(double&,double)> f1,
    std::function<void(double&,double)> f2,
    RCP<Disc> disc,
    apf::Field* a) {
  double result = 0.;
  apf::Mesh* mesh = apf::getMesh(a);
  apf::FieldShape* shape = apf::getShape(a);
  int const space = disc->get_space(shape);
  int const neqs = apf::countComponents(a);
  apf::DynamicArray<apf::Node> nodes = disc->owned_nodes(space);
  std::vector<double> a_vals(neqs, 0.);
  for (auto& node : nodes) {
    apf::MeshEntity* ent = node.entity;
    int const local_node = node.node;
    apf::getComponents(a, ent, local_node, &(a_vals[0]));
    double node_val = 0.;
    for (int eq = 0; eq < neqs; ++eq) {
      f1(node_val, a_vals[eq]);
    }
    f2(result, node_val);
  }
  result = PCU_Add_Double(result);
  return result;
}

void zero_boundary_nodes(
    ParameterList const& dbcs,
    RCP<Disc> disc,
    apf::Field* f) {
  double const vals[3] = {0.,0.,0.};
  int space = disc->get_space(apf::getShape(f));
  for (auto it = dbcs.begin(); it != dbcs.end(); ++it) {
    auto const entry = dbcs.entry(it);
    auto const a = Teuchos::getValue<Teuchos::Array<std::string>>(entry);
    auto const set = a[1];
    NodeSet const& nodes = disc->nodes(space, set);
    for (apf::Node const& node : nodes) {
      apf::setComponents(f, node.entity, node.node, &(vals[0]));
    }
  }
}
 
template <typename T>
void assemble_residual(
    int space,
    int mode,
    RCP<Disc> disc,
    RCP<Residual<T>> r,
    RCP<VectorT> U,
    RCP<Weight> W,
    System& sys) {
  r->set_space(space, disc);
  r->set_mode(mode);
  r->set_weight(W);
  apf::Mesh2* mesh = disc->apf_mesh();
  int q_order = 2;
  for (int es = 0; es < disc->num_elem_sets(); ++es) {
    std::string es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);
    r->before_elems(es, disc);
    for (size_t elem = 0; elem < elems.size(); ++elem) {
      apf::MeshElement* me = apf::createMeshElement(mesh, elems[elem]);
      r->in_elem(me, disc);
      r->gather(disc, U);
      int const npts = apf::countIntPoints(me, q_order);
      for (int pt = 0; pt < npts; ++pt) {
        apf::Vector3 xi;
        apf::getIntPoint(me, q_order, pt, xi);
        double const w = apf::getIntWeight(me, q_order, pt);
        double const dv = apf::getDV(me, xi);
        W->at_point(me, xi);
        r->at_point(xi, w, dv, disc);
      }
      r->scatter(disc, sys);
      r->out_elem();
      apf::destroyMeshElement(me);
    }
  }
  // apply tbcs here
  r->set_space(-1, disc);
  r->set_mode(-1);
  r->set_weight(Teuchos::null);
}

template <typename T>
void assemble_qoi(
    int space,
    RCP<Disc> disc,
    RCP<Residual<T>> r,
    RCP<QoI<T>> qoi,
    RCP<VectorT> U,
    System* sys) {
  r->set_space(space, disc);
  qoi->set_space(space, disc);
  apf::Mesh2* mesh = disc->apf_mesh();
  int q_order = 2;
  for (int es = 0; es < disc->num_elem_sets(); ++es) {
    std::string es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);
    r->before_elems(es, disc);
    qoi->before_elems(es, disc);
    for (size_t elem = 0; elem < elems.size(); ++elem) {
      apf::MeshElement* me = apf::createMeshElement(mesh, elems[elem]);
      r->in_elem(me, disc);
      qoi->in_elem(me, r, disc);
      r->gather(disc, U);
      int const npts = apf::countIntPoints(me, q_order);
      for (int pt = 0; pt < npts; ++pt) {
        apf::Vector3 xi;
        apf::getIntPoint(me, q_order, pt, xi);
        double const w = apf::getIntWeight(me, q_order, pt);
        double const dv = apf::getDV(me, xi);
        qoi->at_point(xi, w, dv, r, disc);
      }
      qoi->scatter(disc, sys);
      qoi->out_elem();
    }
  }
  qoi->post(space, disc, U, sys);
  r->set_space(-1, disc);
  qoi->set_space(-1, disc);
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

void fill_vector(
    int space,
    RCP<Disc> disc,
    apf::Field* f,
    Vector& x) {
  int const neqs = apf::countComponents(f);
  apf::Mesh2* mesh = disc->apf_mesh();
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

static apf::Field* solve_primal(
    int space,
    RCP<ParameterList> params,
    RCP<Disc> disc,
    RCP<Residual<double>> residual,
    RCP<Residual<FADT>> jacobian) {
  apf::Mesh2* mesh = disc->apf_mesh();
  disc->change_shape(space);
  Vector U(space, disc);
  Vector dU(space, disc);
  Vector R(space, disc);
  Matrix dRdU(space, disc);
  System ghost_sys(GHOST, dRdU, dU, R);
  System owned_sys(OWNED, dRdU, dU, R);
  std::string const linear_algebra_name =
    "primal linear algebra " + disc->space_name(space);
  ParameterList const dbcs = params->sublist("dbcs");
  ParameterList& lin_alg = params->sublist(linear_algebra_name);
  ParameterList& newton = params->sublist("newton solve");
  U.zero();
  int iter = 1;
  bool converged = false;
  int const max_iters = newton.get<int>("max iters");
  double const tolerance = newton.get<double>("tolerance");
  RCP<Weight> W = rcp(new Weight(disc->shape(space)));
  while ((iter <= max_iters) && (!converged)) {
    print(" > (%d) Newton iteration", iter);
    dRdU.begin_fill();
    dU.zero();
    R.zero();
    dRdU.zero();
    assemble_residual(space, JACOBIAN, disc, jacobian, U.val[GHOST], W, ghost_sys);
    dRdU.gather(Tpetra::ADD);
    R.gather(Tpetra::ADD);
    apply_jacob_dbcs(dbcs, space, disc, U.val[OWNED], owned_sys, false);
    dRdU.end_fill();
    R.val[OWNED]->scale(-1.0);
    solve(lin_alg, space, disc, owned_sys);
    U.val[OWNED]->update(1.0, *(dU.val[OWNED]), 1.0);
    U.scatter(Tpetra::INSERT);
    R.zero();
    assemble_residual(space, RESIDUAL, disc, residual, U.val[GHOST], W, ghost_sys);
    R.gather(Tpetra::ADD);
    apply_resid_dbcs(dbcs, space, disc, U.val[OWNED], owned_sys);
    double const R_norm = R.val[OWNED]->norm2();
    print(" > ||R|| = %e", R_norm);
    if (R_norm < tolerance) converged = true;
    iter++;
  }
  std::string const name = "u" + disc->space_name(space);
  int const neqs = residual->num_eqs();
  apf::FieldShape* shape = disc->shape(space);
  apf::Field* f = apf::createPackedField(mesh, name.c_str(), neqs, shape);
  apf::zeroField(f);
  fill_field(space, disc, U.val[OWNED], f);
  return f;
}

static double compute_qoi(
    int space,
    RCP<ParameterList> params,
    RCP<Disc> disc,
    RCP<Residual<double>> resid,
    RCP<QoI<double>> qoi,
    apf::Field* u_space) {
  apf::Mesh2* mesh = disc->apf_mesh();
  disc->change_shape(space);
  qoi->reset();
  Vector U(space, disc);
  fill_vector(space, disc, u_space, U);
  assemble_qoi(space, disc, resid, qoi, U.val[GHOST], nullptr);
  double J = qoi->value();
  J = PCU_Add_Double(J);
  return J;
}

static apf::Field* solve_adjoint(
    int space,
    RCP<ParameterList> params,
    RCP<Disc> disc,
    RCP<Residual<FADT>> jacobian,
    RCP<QoI<FADT>> qoi,
    apf::Field* u_space,
    std::string const& post="",
    apf::Field* u_star = nullptr) {
  apf::Mesh2* mesh = disc->apf_mesh();
  disc->change_shape(space);
  Vector U(space, disc);
  Vector Z(space, disc);
  Vector dJdUT(space, disc);
  Matrix dRdUT(space, disc);
  Vector U_star(space, disc);
  System ghost_sys(GHOST, dRdUT, Z, dJdUT);
  System owned_sys(OWNED, dRdUT, Z, dJdUT);
  ParameterList const dbcs = params->sublist("dbcs");
  ParameterList& lin_alg = params->sublist("adjoint linear algebra");
  fill_vector(space, disc, u_space, U);
  if (u_star) {
    fill_vector(space, disc, u_star, U_star);
  } else {
    fill_vector(space, disc, u_space, U_star);
  }
  dRdUT.begin_fill();
  dJdUT.zero();
  Z.zero();
  RCP<Weight> W = rcp(new Weight(disc->shape(space)));
  assemble_residual(space, ADJOINT, disc, jacobian, U.val[GHOST], W, ghost_sys);
  assemble_qoi(space, disc, jacobian, qoi, U_star.val[GHOST], &ghost_sys);
  dRdUT.gather(Tpetra::ADD);
  dJdUT.gather(Tpetra::ADD);
  apply_jacob_dbcs(dbcs, space, disc, U.val[OWNED], owned_sys, true);
  dRdUT.end_fill();
  solve(lin_alg, space, disc, owned_sys);
  std::string const name = "z" + disc->space_name(space) + post;
  int const neqs = jacobian->num_eqs();
  apf::FieldShape* shape = disc->shape(space);
  apf::Field* f = apf::createPackedField(mesh, name.c_str(), neqs, shape);
  apf::zeroField(f);
  fill_field(space, disc, Z.val[OWNED], f);
  return f;
}

static apf::Field* solve_linearized_error(
    RCP<ParameterList> params,
    RCP<Disc> disc,
    RCP<Residual<FADT>> jacobian,
    apf::Field* u,
    std::string const& n) {
  apf::Mesh2* mesh = disc->apf_mesh();
  disc->change_shape(FINE);
  Vector U(FINE, disc);
  Vector EL(FINE, disc);
  Vector R(FINE, disc);
  Matrix dRdU(FINE, disc);
  System ghost_sys(GHOST, dRdU, EL, R);
  System owned_sys(OWNED, dRdU, EL, R);
  ParameterList const dbcs = params->sublist("dbcs");
  ParameterList& lin_alg = params->sublist("adjoint linear algebra");
  fill_vector(FINE, disc, u, U);
  dRdU.begin_fill();
  R.zero();
  EL.zero();
  RCP<Weight> W = rcp(new Weight(disc->shape(FINE)));
  assemble_residual(FINE, JACOBIAN, disc, jacobian, U.val[GHOST], W, ghost_sys);
  R.gather(Tpetra::ADD);
  dRdU.gather(Tpetra::ADD);
  apply_jacob_dbcs(dbcs, FINE, disc, U.val[OWNED], owned_sys, false);
  dRdU.end_fill();
  R.val[OWNED]->scale(-1.0);
  solve(lin_alg, FINE, disc, owned_sys);
  int const neqs = jacobian->num_eqs();
  apf::FieldShape* shape = disc->shape(FINE);
  apf::Field* f = apf::createPackedField(mesh, n.c_str(), neqs, shape);
  apf::zeroField(f);
  fill_field(FINE, disc, EL.val[OWNED], f);
  return f;
}

static apf::Field* solve_2nd_adjoint(
    RCP<ParameterList> params,
    RCP<Disc> disc,
    RCP<Residual<FADT>> jacobian,
    RCP<Residual<FAD2T>> hessian,
    RCP<QoI<FAD2T>> qoi_hessian,
    std::string const& n,
    apf::Field* u,
    apf::Field* el) {
  disc->change_shape(FINE);
  Matrix dRdUT(FINE, disc);
  Vector U(FINE, disc);
  Vector Y(FINE, disc);
  Vector EL(FINE, disc);
  Vector RHS(FINE, disc);
  fill_vector(FINE, disc, u, U);
  fill_vector(FINE, disc, el, EL);
  System ghost_sys(GHOST, dRdUT, Y, RHS);
  System owned_sys(OWNED, dRdUT, Y, RHS);
  ParameterList const dbcs = params->sublist("dbcs");
  ParameterList& lin_alg = params->sublist("adjoint linear algebra");
  {
    RHS.zero();
    Matrix H(FINE, disc);
    Vector dummy(FINE, disc);
    System ghost(GHOST, H, dummy, dummy);
    H.begin_fill();
    assemble_qoi(FINE, disc, hessian, qoi_hessian, U.val[GHOST], &ghost);
    H.gather(Tpetra::ADD);
    H.end_fill();
    H.val[OWNED]->apply(*(EL.val[OWNED]), *(RHS.val[OWNED]));
  }
  dRdUT.begin_fill();
  Y.zero();
  RCP<Weight> W = rcp(new Weight(disc->shape(FINE)));
  assemble_residual(FINE, ADJOINT, disc, jacobian, U.val[GHOST], W, ghost_sys);
  dRdUT.gather(Tpetra::ADD);
  apply_jacob_dbcs(dbcs, FINE, disc, U.val[OWNED], owned_sys, true);
  dRdUT.end_fill();
  solve(lin_alg, FINE, disc, owned_sys);
  (Y.val[OWNED])->scale(0.5);
  int const neqs = jacobian->num_eqs();
  apf::Mesh2* mesh = disc->apf_mesh();
  apf::FieldShape* shape = disc->shape(FINE);
  apf::Field* f = apf::createPackedField(mesh, n.c_str(), neqs, shape);
  apf::zeroField(f);
  fill_field(FINE, disc, Y.val[OWNED], f);
  return f;
}

static apf::Field* solve_ERL(
    RCP<ParameterList> params,
    RCP<Disc> disc,
    RCP<Residual<double>> resid,
    RCP<Residual<FADT>> jacobian,
    apf::Field* uH_h,
    apf::Field* uh_minus_uH_h,
    std::string const& n) {
  apf::Mesh2* mesh = disc->apf_mesh();
  disc->change_shape(FINE);
  Vector U(FINE, disc);
  Vector R(FINE, disc);
  Vector E(FINE, disc);
  Vector U_diff(FINE, disc);
  Matrix dRdU(FINE, disc);
  System ghost_sys(GHOST, dRdU, R, R);
  System owned_sys(OWNED, dRdU, R, R);
  ParameterList const dbcs = params->sublist("dbcs");
  dRdU.begin_fill();
  R.zero();
  dRdU.zero();
  fill_vector(FINE, disc, uH_h, U);
  fill_vector(FINE, disc, uh_minus_uH_h, U_diff);
  RCP<Weight> W = rcp(new Weight(disc->shape(FINE)));
  assemble_residual(FINE, JACOBIAN, disc, jacobian, U.val[GHOST], W, ghost_sys);
  R.gather(Tpetra::ADD);
  dRdU.gather(Tpetra::ADD);
  apply_jacob_dbcs(dbcs, FINE, disc, U.val[OWNED], owned_sys, false);
  dRdU.end_fill();
  dRdU.val[OWNED]->apply(*(U_diff.val[OWNED]), *(E.val[OWNED]));
  E.val[OWNED]->update(-1.0, *(R.val[OWNED]), -1.0);
  int const neqs = jacobian->num_eqs();
  apf::FieldShape* shape = disc->shape(FINE);
  apf::Field* f = apf::createPackedField(mesh, n.c_str(), neqs, shape);
  apf::zeroField(f);
  fill_field(FINE, disc, E.val[OWNED], f);
  return f;
}

static double compute_f(
    RCP<ParameterList> params,
    RCP<Disc> disc,
    RCP<Residual<FADT>> jacobian,
    RCP<QoI<FADT>> qoi_deriv,
    Vector const& E,
    apf::Field* u_eval,
    double Jeh) {
  Vector dummy_vec(FINE, disc);
  Matrix dummy_mat(FINE, disc);
  Vector U(FINE, disc);
  Vector dJdUT(FINE, disc);
  fill_vector(FINE, disc, u_eval, U);
  dJdUT.zero();
  System ghost_sys(GHOST, dummy_mat, dummy_vec, dJdUT);
  System owned_sys(OWNED, dummy_mat, dummy_vec, dJdUT);
  assemble_qoi(FINE, disc, jacobian, qoi_deriv, U.val[GHOST], &ghost_sys);
  dJdUT.gather(Tpetra::ADD);
  double const JL = (dJdUT.val[OWNED])->dot(*(E.val[OWNED]));
  return Jeh - JL;
}

static apf::Field* find_u_star_newton(
    RCP<ParameterList> params,
    RCP<Disc> disc,
    RCP<Residual<FADT>> jacobian,
    RCP<Residual<FAD2T>> hessian,
    RCP<QoI<FADT>> qoi_deriv,
    RCP<QoI<FAD2T>> qoi_hessian,
    nonlinear_in in) {
  ParameterList const dbcs = params->sublist("dbcs");
  ParameterList const newton = params->sublist("newton solve");
  double const Jeh = in.J_fine - in.J_coarse;
  int iter = 1;
  bool converged = false;
  int const max_iters = newton.get<int>("max iters");
  double const tolerance = newton.get<double>("tolerance");
  Vector E(FINE, disc);
  Vector U(FINE, disc);
  Matrix H(FINE, disc);
  Vector He(FINE, disc);
  Vector dummy(FINE, disc);
  System ghost_sys(GHOST, H, dummy, dummy);
  fill_vector(FINE, disc, in.ue, E);
  apf::Field* u_star = nullptr;
  double theta = 0.5;
  while ((iter <= max_iters) && (!converged)) {
    print("> (%d) Newton iteration", iter);
    auto star_op = [&] (double a, double b) { return (1.-theta)*a + theta*b; };
    std::string const name = "u_star" + in.name_append;
    u_star = op(star_op, disc, in.u_coarse, in.u_fine, name);
    double f = compute_f(params, disc, jacobian, qoi_deriv, E, u_star, Jeh); 
    print("> theta = %.15e", theta);
    print("> |f| = %.15e", std::abs(f));
    if (std::abs(f) < tolerance) {
      converged = true;
      break;
    }
    H.zero();
    He.zero();
    H.begin_fill();
    fill_vector(FINE, disc, u_star, U);
    assemble_qoi(FINE, disc, hessian, qoi_hessian, U.val[GHOST], &ghost_sys);
    H.gather(Tpetra::ADD);
    H.end_fill();
    H.val[OWNED]->apply(*(E.val[OWNED]), *(He.val[OWNED]));
    double const df = -(E.val[OWNED])->dot(*(He.val[OWNED]));
    theta = theta - f/df;
    apf::destroyField(u_star);
    iter++;
  }
  return u_star;
}

static apf::Field* find_u_star_bisection(
    RCP<ParameterList> params,
    RCP<Disc> disc,
    RCP<Residual<double>> residual,
    RCP<Residual<FADT>> jacobian,
    RCP<QoI<double>> qoi,
    RCP<QoI<FADT>> qoi_deriv,
    nonlinear_in in) {
  Vector E(FINE, disc);
  Vector dummy_vec(FINE, disc);
  Matrix dummy_mat(FINE, disc);
  fill_vector(FINE, disc, in.ue, E);
  double const Jeh = in.J_fine - in.J_coarse;
  int iter = 1;
  double theta_left = 0.0;
  double theta_right = 1.0;
  apf::Field* u_star = nullptr;
  apf::Field* u_left = nullptr;
  double f_left = compute_f(params, disc, jacobian, qoi_deriv, E, in.u_coarse, Jeh);
  double f_right = compute_f(params, disc, jacobian, qoi_deriv, E, in.u_fine, Jeh); 

  std::cout << "f_left: " << f_left << "\n";
  std::cout << "f_right: " << f_right << "\n";

  if ((f_left * f_right) > 1.e-8) {
    throw std::runtime_error("invalid qoi bisection starting points");
  }


  { // print some values of f to visualize it
    int const n = 10;
    double const dx = 1./double(n);
    std::cout << "theta, f\n";
    for (int i = 0; i < n+1; ++i) {
      double const theta = i*dx;
      auto tmp_op = [&] (double a, double b) { return (1.-theta)*a + theta*b; };
      auto u_tmp = op(tmp_op, disc, in.u_coarse, in.u_fine, "u_tmp");
      double const f = compute_f(params, disc, jacobian, qoi_deriv, E, u_tmp, Jeh);
      std::cout << theta << ", " << f << "\n";
      apf::destroyField(u_tmp);
    }
    std::cout << "---\n";
  }


  while (true) {
    double const theta_mid = 0.5*(theta_right + theta_left);
    auto left = [&] (double a, double b) { return (1.-theta_left)*a + theta_left*b; };
    auto mid = [&] (double a, double b) { return (1.-theta_mid)*a + theta_mid*b; };
    u_star = op(mid, disc, in.u_coarse, in.u_fine, "u_star");
    u_left = op(left, disc, in.u_coarse, in.u_fine, "u_left");
    double f_mid = compute_f(params, disc, jacobian, qoi_deriv, E, u_star, Jeh);
    double f_left = compute_f(params, disc, jacobian, qoi_deriv, E, u_left, Jeh);
    apf::destroyField(u_left);
    print("> (%d) qoi bisesction iteration", iter);
    print("> theta = %.15e", theta_mid);
    print("> |f| = %.15e", f_mid);
    if (std::abs(f_mid) < 1.e-10) {
      print("> converged");
      break;
    } else if ((f_mid * f_left) < 0.) {
      theta_right = theta_mid;
      apf::destroyField(u_star);
    } else {
      theta_left = theta_mid;
      apf::destroyField(u_star);
    }
    iter++;
  }
  return u_star;
}

static nonlinear_out solve_nonlinear_adjoint(
    RCP<ParameterList> params,
    RCP<Disc> disc,
    RCP<Residual<double>> residual,
    RCP<Residual<FADT>> jacobian,
    RCP<Residual<FAD2T>> hessian,
    RCP<QoI<double>> qoi,
    RCP<QoI<FADT>> qoi_deriv,
    RCP<QoI<FAD2T>> qoi_hessian,
    nonlinear_in in) {
  auto const& error_params = params->sublist("error");
  bool const bisection = error_params.get<bool>("bisection");
  nonlinear_out out;
  if (bisection) {
    out.u_star = find_u_star_bisection(
        params, disc, residual, jacobian, qoi, qoi_deriv, in);
  } else {
    out.u_star = find_u_star_newton(
      params, disc, jacobian, hessian, qoi_deriv, qoi_hessian, in);
  }
  std::string const append = "_star" + in.name_append;
  out.z_star = solve_adjoint(
      FINE, params, disc, jacobian, qoi_deriv, in.u_coarse, append, out.u_star);
  return out;
}

static apf::Field* evaluate_residual(
    int space,
    RCP<ParameterList> params,
    RCP<Disc> disc,
    RCP<Residual<double>> residual,
    apf::Field* u_space) {
  apf::Mesh2* mesh = disc->apf_mesh();
  disc->change_shape(space);
  ParameterList const dbcs = params->sublist("dbcs");
  Vector U(space, disc);
  Vector R(space, disc);
  System ghost_sys; ghost_sys.b = R.val[GHOST];
  System owned_sys; owned_sys.b = R.val[OWNED];
  fill_vector(space, disc, u_space, U);
  R.zero();
  RCP<Weight> W = rcp(new Weight(disc->shape(space)));
  assemble_residual(FINE, RESIDUAL, disc, residual, U.val[GHOST], W, ghost_sys);
  R.gather(Tpetra::ADD);
  apply_resid_dbcs(dbcs, FINE, disc, U.val[OWNED], owned_sys);
  std::string const name = "R" + disc->space_name(space);
  int const neqs = residual->num_eqs();
  apf::FieldShape* shape = disc->shape(space);
  apf::Field* f = apf::createPackedField(mesh, name.c_str(), neqs, shape);
  apf::zeroField(f);
  fill_field(space, disc, R.val[OWNED], f);
  return f;
}

Physics::Physics(RCP<ParameterList> params) {
  m_params = params;
  ParameterList const resid_params = m_params->sublist("residual");
  ParameterList const disc_params = m_params->sublist("discretization");
  ParameterList const qoi_params = m_params->sublist("quantity of interest");
  m_disc = rcp(new Disc(disc_params));
  m_residual = create_residual<double>(resid_params, m_disc->num_dims());
  m_jacobian = create_residual<FADT>(resid_params, m_disc->num_dims());
  m_hessian = create_residual<FAD2T>(resid_params, m_disc->num_dims());
  m_qoi = create_QoI<double>(qoi_params);
  m_qoi_deriv = create_QoI<FADT>(qoi_params);
  m_qoi_hessian = create_QoI<FAD2T>(qoi_params);
}

void Physics::build_disc() {
  int const neqs = m_residual->num_eqs();
  m_disc->build_data(neqs);
}

void Physics::destroy_disc() {
  m_disc->destroy_data();
}

void Physics::destroy_residual_data() {
  m_residual->destroy_data();
  m_jacobian->destroy_data();
}

apf::Field* Physics::solve_primal(int space) {
  return calibr8::solve_primal(
      space,
      m_params,
      m_disc,
      m_residual,
      m_jacobian);
}

apf::Field* Physics::solve_adjoint(int space, apf::Field* u) {
  ASSERT(apf::getShape(u) == m_disc->shape(space));
  return calibr8::solve_adjoint(
      space,
      m_params,
      m_disc,
      m_jacobian,
      m_qoi_deriv,
      u);
}

apf::Field* Physics::solve_linearized_error(apf::Field* u, std::string const& n) {
  ASSERT(apf::getShape(u) == m_disc->shape(FINE));
  return calibr8::solve_linearized_error(
      m_params,
      m_disc,
      m_jacobian,
      u,
      n);
}

apf::Field* Physics::solve_2nd_adjoint(apf::Field* u, apf::Field* ue, std::string const& n) {
  ASSERT(apf::getShape(u) == m_disc->shape(FINE));
  ASSERT(apf::getShape(ue) == m_disc->shape(FINE));
  return calibr8::solve_2nd_adjoint(
      m_params,
      m_disc,
      m_jacobian,
      m_hessian,
      m_qoi_hessian,
      n,
      u,
      ue);
}

apf::Field* Physics::solve_ERL(apf::Field* u, apf::Field* ue, std::string const& n) {
  ASSERT(apf::getShape(u) == m_disc->shape(FINE));
  ASSERT(apf::getShape(ue) == m_disc->shape(FINE));
  return calibr8::solve_ERL(
      m_params,
      m_disc,
      m_residual,
      m_jacobian,
      u,
      ue,
      n);
}

nonlinear_out Physics::solve_nonlinear_adjoint(nonlinear_in in) {
  ASSERT(apf::getShape(in.u_coarse) == m_disc->shape(FINE));
  ASSERT(apf::getShape(in.u_fine) == m_disc->shape(FINE));
  ASSERT(apf::getShape(in.ue) == m_disc->shape(FINE));
  ASSERT(in.J_coarse != 0.);
  ASSERT(in.J_fine != 0.);
  return calibr8::solve_nonlinear_adjoint(
      m_params,
      m_disc,
      m_residual,
      m_jacobian,
      m_hessian,
      m_qoi,
      m_qoi_deriv,
      m_qoi_hessian,
      in);
}

apf::Field* Physics::evaluate_residual(int space, apf::Field* u) {
  ASSERT(apf::getShape(u) == m_disc->shape(space));
  return calibr8::evaluate_residual(
      space,
      m_params,
      m_disc,
      m_residual,
      u);
}

apf::Field* Physics::subtract(apf::Field* f, apf::Field* g, std::string const& n) {
  ASSERT(apf::getShape(f) == apf::getShape(g));
  return op(calibr8::subtract, m_disc, f, g, n.c_str());
}

apf::Field* Physics::prolong(apf::Field* f, std::string const& n) {
  ASSERT(apf::getShape(f) == m_disc->shape(COARSE));
  return calibr8::project(m_disc, f, n.c_str());
}

apf::Field* Physics::restrict(apf::Field* f, std::string const& n) {
  ASSERT(apf::getShape(f) == m_disc->shape(FINE));
  return calibr8::project(m_disc, f, n.c_str());
}

apf::Field* Physics::recover(apf::Field* f, std::string const& n) {
  ASSERT(apf::getShape(f) == m_disc->shape(COARSE));
  std::string const name = std::string(apf::getName(f)) + "_ip";
  apf::Field* f_ips = interpolate_to_ips(f, name);
  m_disc->change_shape(FINE);
  apf::Field* f_spr = spr_recovery(f_ips);
  ParameterList const dbcs = m_params->sublist("dbcs");
  zero_boundary_nodes(dbcs, m_disc, f_spr);
  apf::renameField(f_spr, n.c_str());
  apf::destroyField(f_ips);
  return f_spr;
}

apf::Field* Physics::modify_star(
    apf::Field* z, apf::Field* R, apf::Field* E, std::string const& n) {
  ASSERT(apf::getShape(z) == m_disc->shape(FINE));
  ASSERT(apf::getShape(R) == m_disc->shape(FINE));
  ASSERT(apf::getShape(E) == m_disc->shape(FINE));
  double const num = this->dot(z,E);
  double const den = this->dot(R,R);
  double const gamma = num/den;
  auto add_scaled = [&] (double a, double b) {
    return a + gamma*b;
  };
  return op(add_scaled, m_disc, z, R, n);
}

apf::Field* Physics::diff(apf::Field* z, std::string const& n) {
  ASSERT(apf::getShape(z) == m_disc->shape(FINE));
  apf::Field* z1 = this->restrict(z, "tmp1");
  apf::Field* z2 = this->prolong(z1, "tmp2");
  apf::Field* diff = this->subtract(z, z2, n);
  apf::destroyField(z1);
  apf::destroyField(z2);
  return diff;
}

apf::Field* Physics::localize(apf::Field* u, apf::Field* z, std::string const& n) {
  ASSERT(apf::getShape(u) == m_disc->shape(FINE));
  ASSERT(apf::getShape(z) == m_disc->shape(FINE));
  return m_residual->assemble(u, z, n);
}

apf::Field* Physics::localize(
    apf::Field* R,
    apf::Field* z, apf::Field* z_diff,
    apf::Field* y, apf::Field* y_diff,
    apf::Field* E, std::string const& n) {
  ASSERT(apf::getShape(R) == m_disc->shape(FINE));
  ASSERT(apf::getShape(z) == m_disc->shape(FINE));
  ASSERT(apf::getShape(z_diff) == m_disc->shape(FINE));
  ASSERT(apf::getShape(y) == m_disc->shape(FINE));
  ASSERT(apf::getShape(y_diff) == m_disc->shape(FINE));
  ASSERT(apf::getShape(E) == m_disc->shape(FINE));
  apf::Field* eta1 = op(negate_multiply, m_disc, R, z_diff, "tmp1");
  apf::Field* eta2 = op(negate_multiply, m_disc, R, y_diff, "tmp2");
  apf::Field* eta3 = op(negate_multiply, m_disc, E, z, "tmp3");
  apf::Field* eta4 = op(negate_multiply, m_disc, E, y, "tmp4");
  apf::Field* eta12 = op(add, m_disc, eta1, eta2, "tmp5");
  apf::Field* eta34 = op(add, m_disc, eta3, eta4, "tmp6");
  apf::Field* result = op(add, m_disc, eta12, eta34, n);
  apf::destroyField(eta1);
  apf::destroyField(eta2);
  apf::destroyField(eta3);
  apf::destroyField(eta4);
  apf::destroyField(eta12);
  apf::destroyField(eta34);
  return result;
}

double Physics::compute_qoi(int space, apf::Field* u) {
  ASSERT(apf::getShape(u) == m_disc->shape(space));
  return calibr8::compute_qoi(
    space,
    m_params,
    m_disc,
    m_residual,
    m_qoi,
    u);
}

double Physics::dot(apf::Field* a, apf::Field* b) {
  ASSERT(apf::getShape(a) == m_disc->shape(FINE));
  ASSERT(apf::getShape(b) == m_disc->shape(FINE));
  Vector A(FINE, m_disc);
  Vector B(FINE, m_disc);
  fill_vector(FINE, m_disc, a, A);
  fill_vector(FINE, m_disc, b, B);
  return (A.val[OWNED])->dot(*(B.val[OWNED]));
}

double Physics::compute_sum(apf::Field* e) {
  return op(sum_into, sum_into, m_disc, e);
}

double Physics::compute_bound(apf::Field* e) {
  return op(sum_into, abs_sum_into, m_disc, e);
}

}
