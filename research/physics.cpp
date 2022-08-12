#include <PCU.h>
#include "bcs.hpp"
#include "control.hpp"
#include "disc.hpp"
#include "linalg.hpp"
#include "residual.hpp"
#include "physics.hpp"
#include "weights.hpp"

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

template <typename T>
void assemble_residual(
    int space,
    int mode,
    RCP<Disc> disc,
    RCP<Residual<T>> r,
    RCP<Weight> weight,
    RCP<VectorT> U,
    RCP<VectorT> Z,
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
      weight->in_elem(me, disc);
      weight->gather(disc, Z);
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
      weight->out_elem();
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
    assemble_residual(space, JACOBIAN, disc, jacobian, weight, U.val[GHOST], Teuchos::null, ghost_sys);
    dRdU.gather(Tpetra::ADD);
    R.gather(Tpetra::ADD);
    R.val[OWNED]->scale(-1.0);
    apply_jacob_dbcs(dbcs, space, disc, U.val[OWNED], owned_sys, false);
    dRdU.end_fill();
    solve(lin_alg, space, disc, owned_sys);
    U.val[OWNED]->update(1.0, *(dU.val[OWNED]), 1.0);
    U.scatter(Tpetra::INSERT);
    R.zero();
    assemble_residual(space, RESIDUAL, disc, residual, weight, U.val[GHOST], Teuchos::null, ghost_sys);
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

static double compute_qoi(
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
  print(" %s = %.15e", name.c_str(), J);
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
  apf::FieldShape* shape = disc->shape(space);
  mesh->changeShape(shape, true);
  Vector U(space, disc);
  Vector Z(space, disc);
  Vector dJdUT(space, disc);
  Matrix dRdUT(space, disc);
  System ghost_sys(GHOST, dRdUT, Z, dJdUT);
  System owned_sys(OWNED, dRdUT, Z, dJdUT);
  RCP<Weight> weight = rcp(new Weight(shape));
  ParameterList const dbcs = params->sublist("dbcs");
  ParameterList& lin_alg = params->sublist("linear algebra");
  fill_vector(space, disc, u_space, U);
  dRdUT.begin_fill();
  dJdUT.zero();
  Z.zero();
  assemble_residual(space, ADJOINT, disc, jacobian, weight, U.val[GHOST], Teuchos::null, ghost_sys);
  assemble_qoi(space, disc, jacobian, qoi, U.val[GHOST], &ghost_sys);
  dRdUT.gather(Tpetra::ADD);
  dJdUT.gather(Tpetra::ADD);
  apply_jacob_dbcs(dbcs, space, disc, U.val[OWNED], owned_sys, true);
  dRdUT.end_fill();
  solve(lin_alg, space, disc, owned_sys);
  std::string const name = "z" + disc->space_name(space);
  int const neqs = jacobian->num_eqs();
  apf::Field* f = apf::createPackedField(mesh, name.c_str(), neqs, shape);
  apf::zeroField(f);
  fill_field(space, disc, Z.val[OWNED], f);
  return f;
}

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
  assemble_residual(FINE, JACOBIAN, disc, jacobian, weight, U.val[GHOST], Teuchos::null, ghost_sys);
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
  apf::Field* f = apf::createPackedField(mesh, "E_L", neqs, shape);
  apf::zeroField(f);
  fill_field(FINE, disc, E.val[OWNED], f);
  return f;
}

static void do_stuff(
    RCP<Disc> disc,
    RCP<Residual<double>> residual,
    apf::Field* zh,
    apf::Field* uH_h) {
  Vector Z(FINE, disc);
  Vector R(FINE, disc);
  Vector U(FINE, disc);
  Matrix dRdU(FINE, disc);
  System ghost_sys(GHOST, dRdU, R, R);
  System owned_sys(OWNED, dRdU, R, R);
  fill_vector(FINE, disc, zh, Z);
  fill_vector(FINE, disc, uH_h, U);
  R.zero();
  apf::FieldShape* shape = disc->shape(FINE);
  RCP<Weight> weight = rcp(new Weight(shape));
  assemble_residual(FINE, RESIDUAL, disc, residual, weight, U.val[GHOST], Teuchos::null, ghost_sys);
  R.gather(Tpetra::ADD);
  double eta = (Z.val[OWNED])->dot(*(R.val[OWNED]));
  print("eta = %.15e", eta);
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

apf::Field* Physics::solve_primal(int space) {
  print("primal %s", m_disc->space_name(space).c_str());
  return calibr8::solve_primal(
      space,
      m_params,
      m_disc,
      m_residual,
      m_jacobian);
}

apf::Field* Physics::solve_adjoint(int space, apf::Field* u) {
  ASSERT(apf::getShape(u) == m_disc->shape(space));
  print("adjoint %s", m_disc->space_name(space).c_str());
  return calibr8::solve_adjoint(
      space,
      m_params,
      m_disc,
      m_jacobian,
      m_qoi_deriv,
      u);
}

apf::Field* Physics::prolong_u_coarse_onto_fine(apf::Field* u) {
  ASSERT(apf::getShape(u) == m_disc->shape(COARSE));
  print("prolonging uH onto h");
  return calibr8::project(m_disc, u, "uH_h");
}

apf::Field* Physics::restrict_z_fine_onto_fine(apf::Field* z) {
  ASSERT(apf::getShape(z) == m_disc->shape(FINE));
  print("restricting zh onto H on h");
  apf::Field* tmp = calibr8::project(m_disc, z, "tmp");
  apf::Field* zh_H = calibr8::project(m_disc, tmp, "zh_H");
  apf::destroyField(tmp);
  return zh_H;
}

apf::Field* Physics::compute_linearization_error(
    apf::Field* uH_h,
    apf::Field* uh_minus_uH_h,
    double& norm_R,
    double& norm_E) {
  ASSERT(apf::getShape(uH_h) == m_disc->shape(FINE));
  ASSERT(apf::getShape(uh_minus_uH_h) == m_disc->shape(FINE));
  print("computing linearization error");
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

}
