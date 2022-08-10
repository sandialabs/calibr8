#include <Teuchos_YamlParameterListHelpers.hpp>
#include <lionPrint.h>
#include "bcs.hpp"
#include "control.hpp"
#include "disc.hpp"
#include "linalg.hpp"
#include "residual.hpp"
#include "physics.hpp"
#include "weights.hpp"

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
    Fields m_fields;
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

static void print_banner(std::string const& banner) {
  print("****************");
  print("%s", banner.c_str());
  print("****************");
}

void Driver::drive() {
  int const neqs = m_residual->num_eqs();
  m_disc->build_data(neqs);
  print_banner("primal H");
  m_fields.u[COARSE] = solve_primal(COARSE, m_params, m_disc, m_residual, m_jacobian);
  print_banner("primal h");
  m_fields.u[FINE] = solve_primal(FINE, m_params, m_disc, m_residual, m_jacobian);
  print("* projecting uH onto h");
  m_fields.uH_h = project(m_disc, m_fields.u[COARSE], "uH_h");
  print("* computing uh-uH_h");
  print("----");
  m_fields.uh_minus_uH_h = subtract(m_disc, m_fields.u[FINE], m_fields.uH_h, "uh-uH_h");
  print_banner("linearization error");
  m_fields.E_L = compute_linearization_error(
      m_params, m_disc, m_residual, m_jacobian, m_fields.uH_h, m_fields.uh_minus_uH_h);


  // ---debug below---
  apf::writeVtkFiles("debug", m_disc->apf_mesh());

}

int main(int argc, char** argv) {
  initialize();
  ASSERT(argc == 2);
  {
    lion_set_verbosity(1);
    Driver driver(argv[1]);
    driver.drive();
  }
  finalize();
}
