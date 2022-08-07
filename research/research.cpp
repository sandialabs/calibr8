#include <Teuchos_YamlParameterListHelpers.hpp>
#include <lionPrint.h>
#include "control.hpp"
#include "disc.hpp"
#include "residual.hpp"
#include "system.hpp"

using namespace calibr8;

class Driver {
  public:
    Driver(std::string const& input_file);
    ~Driver();
    void drive();
  private:
    RCP<ParameterList> m_params;
    RCP<Disc> m_disc;
    RCP<Residual<double>> m_residual;
    RCP<Residual<FADT>> m_jacobian;
    RCP<System> m_system;
};

Driver::Driver(std::string const& in) {
  print("reading input: %s", in.c_str());
  m_params = rcp(new ParameterList);
  Teuchos::updateParametersFromYamlFile(in, m_params.ptr());
  ParameterList const resid_params = m_params->sublist("residual");
  ParameterList const disc_params = m_params->sublist("discretization");
  m_disc = rcp(new Disc(disc_params));
  m_system = rcp(new System());
  m_residual = create_residual<double>(resid_params, m_disc->num_dims());
  m_jacobian = create_residual<FADT>(resid_params, m_disc->num_dims());
}

Driver::~Driver() {
  m_system->destroy_data();
  m_disc->destroy_data();
}

template <typename T>
void assemble(
    int space,
    int mode,
    RCP<Disc> disc,
    RCP<Residual<T>> r,
    RCP<System> sys) {
  apf::Mesh2* mesh = disc->apf_mesh();
  int order = disc->order(space);
  for (int es = 0; es < disc->num_elem_sets(); ++es) {
    std::string es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);
    for (size_t elem = 0; elem < elems.size(); ++elem) {
      apf::MeshElement* me = apf::createMeshElement(mesh, elems[elem]);
      r->gather(me, disc, sys->x[space][GHOST]);
      int const npts = apf::countIntPoints(me, order);
      for (int pt = 0; pt < npts; ++pt) {
        apf::Vector3 xi;
        apf::getIntPoint(me, order, pt, xi);
        double const w = apf::getIntWeight(me, order, pt);
        double const dv = apf::getDV(me, xi);
        r->interpolate(xi);
        r->at_point(xi, w, dv);
      }
      r->scatter(me, disc, sys);
    }
  }
}

void Driver::drive() {
  m_disc->build_data(/*neqs=*/1);
  m_system->build_data(m_disc);


  assemble(COARSE, RESIDUAL, m_disc, m_residual, m_system);
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
