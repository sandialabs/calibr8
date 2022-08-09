#include <Teuchos_YamlParameterListHelpers.hpp>
#include <lionPrint.h>
#include "bcs.hpp"
#include "control.hpp"
#include "disc.hpp"
#include "linalg.hpp"
#include "residual.hpp"
#include "weights.hpp"

using namespace calibr8;

class Driver {
  public:
    Driver(std::string const& input_file);
    ~Driver();
    void drive();
  private:
    void solve_primal(int space);
    apf::Field* fill(int space, std::string const& name, RCP<VectorT> x);
  private:
    RCP<ParameterList> m_params;
    RCP<Disc> m_disc;
    RCP<Residual<double>> m_residual;
    RCP<Residual<FADT>> m_jacobian;
    apf::Field* m_u[NUM_SPACE] = {nullptr};
    apf::Field* m_z[NUM_SPACE] = {nullptr};
};

Driver::Driver(std::string const& in) {
  print("reading input: %s", in.c_str());
  m_params = rcp(new ParameterList);
  Teuchos::updateParametersFromYamlFile(in, m_params.ptr());
  ParameterList const resid_params = m_params->sublist("residual");
  ParameterList const disc_params = m_params->sublist("discretization");
  m_disc = rcp(new Disc(disc_params));
  m_residual = create_residual<double>(resid_params, m_disc->num_dims());
  m_jacobian = create_residual<FADT>(resid_params, m_disc->num_dims());
}

Driver::~Driver() {
  m_disc->destroy_data();
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
  int order = disc->order(space);
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

void Driver::solve_primal(int space) {

  apf::Mesh2* mesh = m_disc->apf_mesh();
  apf::FieldShape* shape = m_disc->shape(space);
  mesh->changeShape(shape, true);

  Vector U(space, m_disc);
  Vector dU(space, m_disc);
  Vector R(space, m_disc);
  Matrix dRdU(space, m_disc);
  System ghost_sys(GHOST, dRdU, dU, R);
  System owned_sys(OWNED, dRdU, dU, R);

  RCP<Weight> weight = rcp(new Weight(m_disc->shape(space)));
  ParameterList const dbcs = m_params->sublist("dbcs");
  ParameterList& lin_alg = m_params->sublist("linear algebra");
  ParameterList& newton = m_params->sublist("newton solve");

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

    assemble(space, JACOBIAN, m_disc, m_jacobian, weight, U.val[GHOST], ghost_sys);
    dRdU.gather(Tpetra::ADD);
    R.gather(Tpetra::ADD);
    R.val[OWNED]->scale(-1.0);
    apply_jacob_dbcs(dbcs, space, m_disc, U.val[OWNED], owned_sys, false);
    dRdU.end_fill();

    solve(lin_alg, space, m_disc, owned_sys);
    U.val[OWNED]->update(1.0, *(dU.val[OWNED]), 1.0);
    U.scatter(Tpetra::INSERT);

    R.zero();
    assemble(space, RESIDUAL, m_disc, m_residual, weight, U.val[GHOST], ghost_sys);
    R.gather(Tpetra::ADD);
    apply_resid_dbcs(dbcs, space, m_disc, U.val[OWNED], owned_sys);
    double const R_norm = R.val[OWNED]->norm2();
    print(" ||R|| = %e", R_norm);
    if (R_norm < tolerance) converged = true;

    iter++;

  }

  std::string const name = "u_" + m_disc->space_name(space);
  m_u[space] = fill(space, name, U.val[OWNED]);

}

apf::Field* Driver::fill(
    int space,
    std::string const& name,
    RCP<VectorT> x) {
  int const neqs = m_residual->num_eqs();
  apf::Mesh2* mesh = m_disc->apf_mesh();
  apf::Field* f = apf::createPackedField(mesh, name.c_str(), neqs);
  apf::zeroField(f);
  auto x_data = x->get1dView();
  std::vector<double> vals(neqs, 0);
  apf::DynamicArray<apf::Node> nodes = m_disc->owned_nodes(space);
  for (auto& node : nodes) {
    apf::MeshEntity* ent = node.entity;
    int const local_node = node.node;
    for (int eq = 0; eq < neqs; ++eq) {
      LO const row = m_disc->get_lid(space, node, eq);
      vals[eq] = x_data[row];
    }
    apf::setComponents(f, ent, local_node, &(vals[0]));
  }
  return f;
}

void Driver::drive() {
  int const neqs = m_residual->num_eqs();
  m_disc->build_data(neqs);
  solve_primal(COARSE);

  apf::writeVtkFiles("debug", m_disc->apf_mesh());

}

int main(int argc, char** argv) {
  initialize();
  ALWAYS_ASSERT(argc == 2);
  {
    lion_set_verbosity(1);
    Driver driver(argv[1]);
    driver.drive();
  }
  finalize();
}
