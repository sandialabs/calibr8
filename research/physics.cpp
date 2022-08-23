#include <PCU.h>
#include "bcs.hpp"
#include "control.hpp"
#include "disc.hpp"
#include "linalg.hpp"
#include "residual.hpp"
#include "physics.hpp"

namespace calibr8 {

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

static double subtract(double a, double b) { return a-b; }
static double negate_multiply(double a, double b) { return -a*b; }

static void sum_into (double& a, double b) { a += b; }
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

template <typename T>
void assemble_residual(
    int space,
    int mode,
    RCP<Disc> disc,
    RCP<Residual<T>> r,
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
        r->at_point(xi, w, dv, disc);
      }
      r->scatter(disc, sys);
      r->out_elem();
      apf::destroyMeshElement(me);
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
    assemble_residual(space, JACOBIAN, disc, jacobian, U.val[GHOST], ghost_sys);
    dRdU.gather(Tpetra::ADD);
    R.gather(Tpetra::ADD);
    apply_jacob_dbcs(dbcs, space, disc, U.val[OWNED], owned_sys, false);
    dRdU.end_fill();
    R.val[OWNED]->scale(-1.0);
    solve(lin_alg, space, disc, owned_sys);
    U.val[OWNED]->update(1.0, *(dU.val[OWNED]), 1.0);
    U.scatter(Tpetra::INSERT);
    R.zero();
    assemble_residual(space, RESIDUAL, disc, residual, U.val[GHOST], ghost_sys);
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
  std::string const name = "J" + disc->space_name(space);
  print(" > %s = %.15e", name.c_str(), J);
  return J;
}

static apf::Field* solve_adjoint(
    int space,
    RCP<ParameterList> params,
    RCP<Disc> disc,
    RCP<Residual<FADT>> jacobian,
    RCP<QoI<FADT>> qoi,
    apf::Field* u_space) {
  apf::Mesh2* mesh = disc->apf_mesh();
  disc->change_shape(space);
  Vector U(space, disc);
  Vector Z(space, disc);
  Vector dJdUT(space, disc);
  Matrix dRdUT(space, disc);
  System ghost_sys(GHOST, dRdUT, Z, dJdUT);
  System owned_sys(OWNED, dRdUT, Z, dJdUT);
  ParameterList const dbcs = params->sublist("dbcs");
  ParameterList& lin_alg = params->sublist("linear algebra");
  fill_vector(space, disc, u_space, U);
  dRdUT.begin_fill();
  dJdUT.zero();
  Z.zero();
  assemble_residual(space, ADJOINT, disc, jacobian, U.val[GHOST], ghost_sys);
  assemble_qoi(space, disc, jacobian, qoi, U.val[GHOST], &ghost_sys);
  dRdUT.gather(Tpetra::ADD);
  dJdUT.gather(Tpetra::ADD);
  apply_jacob_dbcs(dbcs, space, disc, U.val[OWNED], owned_sys, true);
  dRdUT.end_fill();
  solve(lin_alg, space, disc, owned_sys);
  std::string const name = "z" + disc->space_name(space);
  int const neqs = jacobian->num_eqs();
  apf::FieldShape* shape = disc->shape(space);
  apf::Field* f = apf::createPackedField(mesh, name.c_str(), neqs, shape);
  apf::zeroField(f);
  fill_field(space, disc, Z.val[OWNED], f);
  return f;
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
  fill_vector(FINE, disc, u_space, U);
  R.zero();
  assemble_residual(FINE, RESIDUAL, disc, residual, U.val[GHOST], ghost_sys);
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

// work on below

static apf::Field* compute_linearization_error(
    RCP<ParameterList> params,
    RCP<Disc> disc,
    RCP<Residual<double>> resid,
    RCP<Residual<FADT>> jacobian,
    apf::Field* uH_h,
    apf::Field* uh_minus_uH_h,
    double& norm_R,
    double& norm_E) {
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
  assemble_residual(FINE, JACOBIAN, disc, jacobian, U.val[GHOST], ghost_sys);
  R.gather(Tpetra::ADD);
  dRdU.gather(Tpetra::ADD);
  apply_jacob_dbcs(dbcs, FINE, disc, U.val[OWNED], owned_sys, false);
  dRdU.end_fill();
  dRdU.val[OWNED]->apply(*(U_diff.val[OWNED]), *(E.val[OWNED]));
  E.val[OWNED]->update(-1.0, *(R.val[OWNED]), -1.0);
  norm_R = R.val[OWNED]->norm2();
  norm_E = E.val[OWNED]->norm2();
  print(" > ||R|| = %.15e", norm_R);
  print(" > ||E_L|| = %.15e", norm_E);
  int const neqs = jacobian->num_eqs();
  apf::FieldShape* shape = disc->shape(FINE);
  apf::Field* f = apf::createPackedField(mesh, "E_L", neqs, shape);
  apf::zeroField(f);
  fill_field(FINE, disc, E.val[OWNED], f);
  return f;
}

static double compute_eta_L(
    RCP<Disc> disc,
    apf::Field* z,
    apf::Field* E_L) {
  Vector Z(FINE, disc);
  Vector E(FINE, disc);
  fill_vector(FINE, disc, z, Z);
  fill_vector(FINE, disc, E_L, E);
  double eta_L = -(Z.val[OWNED])->dot(*E.val[OWNED]);
  print(" > eta_L = %.15e", eta_L);
  return eta_L;
}

Physics::Physics(RCP<ParameterList> params) {
  m_params = params;
  ParameterList const resid_params = m_params->sublist("residual");
  ParameterList const disc_params = m_params->sublist("discretization");
  ParameterList const qoi_params = m_params->sublist("quantity of interest");
  m_disc = rcp(new Disc(disc_params));
  m_residual = create_residual<double>(resid_params, m_disc->num_dims());
  m_jacobian = create_residual<FADT>(resid_params, m_disc->num_dims());
  m_qoi = create_QoI<double>(qoi_params);
  m_qoi_deriv = create_QoI<FADT>(qoi_params);
}

void Physics::build_disc() {
  int const neqs = m_residual->num_eqs();
  m_disc->build_data(neqs);
}

void Physics::destroy_disc() {
  m_disc->destroy_data();
}

apf::Field* Physics::solve_primal(int space) {
  print("primal %s", m_disc->space_name(space).c_str());
  return calibr8::solve_primal(
      space,
      m_params,
      m_disc,
      m_residual,
      m_jacobian);
}

double Physics::compute_qoi(int space, apf::Field* u) {
  ASSERT(apf::getShape(u) == m_disc->shape(space));
  print("qoi %s", m_disc->space_name(space).c_str());
  return calibr8::compute_qoi(
    space,
    m_params,
    m_disc,
    m_residual,
    m_qoi,
    u);
}

apf::Field* Physics::solve_adjoint(int space, apf::Field* u) {
  print("adjoint %s", m_disc->space_name(space).c_str());
  ASSERT(apf::getShape(u) == m_disc->shape(space));
  return calibr8::solve_adjoint(
      space,
      m_params,
      m_disc,
      m_jacobian,
      m_qoi_deriv,
      u);
}

apf::Field* Physics::prolong_u_coarse_onto_fine(apf::Field* u) {
  print("prolonging uH onto h");
  ASSERT(apf::getShape(u) == m_disc->shape(COARSE));
  return calibr8::project(m_disc, u, "uH_h");
}

apf::Field* Physics::restrict_z_fine_onto_fine(apf::Field* z) {
  print("restricting zh onto H on h");
  ASSERT(apf::getShape(z) == m_disc->shape(FINE));
  apf::Field* tmp = calibr8::project(m_disc, z, "tmp");
  apf::Field* zh_H = calibr8::project(m_disc, tmp, "zh_H");
  apf::destroyField(tmp);
  return zh_H;
}

apf::Field* Physics::subtract_z_coarse_from_z_fine(apf::Field* zh, apf::Field* zH) {
  print("subtracting zH from zh");
  ASSERT(apf::getShape(zh) == m_disc->shape(FINE));
  ASSERT(apf::getShape(zH) == m_disc->shape(FINE));
  return op(subtract, m_disc, zh, zH, "zh_minus_zH");
}

apf::Field* Physics::evaluate_residual(int space, apf::Field* u) {
  print("evaluating residual");
  return calibr8::evaluate_residual(
      space,
      m_params,
      m_disc,
      m_residual,
      u);
}

apf::Field* Physics::localize_error(apf::Field* R, apf::Field* z) {
  print("localizing error");
  ASSERT(apf::getShape(R) == m_disc->shape(FINE));
  ASSERT(apf::getShape(z) == m_disc->shape(FINE));
  return op(negate_multiply, m_disc, R, z, "eta");
}

double Physics::estimate_error(apf::Field* eta) {
  print("estimating error");
  double const estimate = op(sum_into, sum_into, m_disc, eta);
  print(" > eta = %.15e", estimate);
  return estimate;
}

double Physics::estimate_error_bound(apf::Field* eta) {
  print("estimating error bound");
  double const estimate = op(sum_into, abs_sum_into, m_disc, eta);
  print(" > |eta| < %.15e", estimate);
  return estimate;
}

// work on the below

apf::Field* Physics::compute_linearization_error(
    apf::Field* uH_h,
    apf::Field* uh_minus_uH_h,
    double& norm_R,
    double& norm_E) {
  print("computing linearization error");
  ASSERT(apf::getShape(uH_h) == m_disc->shape(FINE));
  ASSERT(apf::getShape(uh_minus_uH_h) == m_disc->shape(FINE));
  return calibr8::compute_linearization_error(
    m_params,
    m_disc,
    m_residual,
    m_jacobian,
    uH_h,
    uh_minus_uH_h,
    norm_R,
    norm_E);
}

double Physics::compute_eta_L(apf::Field* z, apf::Field* E_L) {
  ASSERT(apf::getShape(z) == m_disc->shape(FINE));
  ASSERT(apf::getShape(E_L) == m_disc->shape(FINE));
  print("computing linearization error estimate");
  return calibr8::compute_eta_L(m_disc, z, E_L);
}

}
