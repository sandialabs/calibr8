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
  private:
    RCP<ParameterList> m_params;
    RCP<Disc> m_disc;
    RCP<Residual<double>> m_residual;
    RCP<Residual<FADT>> m_jacobian;
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
      r->gather(disc, sys.x);
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
  r->set_space(-1);
  r->set_mode(-1);
}

void Driver::solve_primal(int space) {

  Vector U(space, m_disc);
  Vector R(space, m_disc);
  Matrix dR_dU(space, m_disc);
  System ghost_sys(GHOST, dR_dU, U, R);
  System owned_sys(OWNED, dR_dU, U, R);

  RCP<Weight> weight = rcp(new Weight(m_disc->shape(space)));
  ParameterList const dbcs = m_params->sublist("dbcs");
  ParameterList& linalg = m_params->sublist("linear algebra");

  dR_dU.begin_fill();
  U.zero();
  R.zero();
  dR_dU.zero();

  assemble(space, JACOBIAN, m_disc, m_jacobian, weight, ghost_sys);
  dR_dU.gather(Tpetra::ADD);
  R.gather(Tpetra::ADD);
  apply_jacob_dbcs(dbcs, space, m_disc, owned_sys, false);
  dR_dU.end_fill();

  solve(linalg, space, m_disc, owned_sys);

}

void Driver::drive() {
  int const neqs = m_residual->num_eqs();
  m_disc->build_data(neqs);
  solve_primal(COARSE);
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
