#include <Teuchos_YamlParameterListHelpers.hpp>
#include <lionPrint.h>
#include "control.hpp"
#include "physics.hpp"

using namespace calibr8;

class Driver {
  public:
    Driver(std::string const& input_file);
    void drive();
  private:
    RCP<ParameterList> m_params;
    RCP<Physics> m_physics;
};

Driver::Driver(std::string const& in) {
  print("reading input: %s", in.c_str());
  m_params = rcp(new ParameterList);
  Teuchos::updateParametersFromYamlFile(in, m_params.ptr());
  m_physics = rcp(new Physics(m_params));
}

void Driver::drive() {
  m_physics->build_disc();
  apf::Field* uH = m_physics->solve_primal(COARSE);
  apf::Field* uh = m_physics->solve_primal(FINE);
  double const JH = m_physics->compute_qoi(COARSE, uH);
  double const Jh = m_physics->compute_qoi(FINE, uh);
  apf::Field* uH_h = m_physics->prolong_u_coarse_onto_fine(uH);
  apf::Field* zh = m_physics->solve_adjoint(FINE, uH_h);
  apf::Field* uh_minus_uH_h = subtract(m_physics->disc(), uh, uH_h, "uh-uH_h");
  apf::Field* zh_H = m_physics->restrict_z_fine_onto_fine(zh);
  apf::Field* zh_minus_zh_H = subtract(m_physics->disc(), zh, zh_H, "zh-zh_H");
  double norm_R, norm_E;
  apf::Field* E_L = m_physics->compute_linearization_error(
      uH_h, uh_minus_uH_h, norm_R, norm_E);



  apf::writeVtkFiles("debug", m_physics->disc()->apf_mesh());

  apf::destroyField(uh_minus_uH_h);
  apf::destroyField(zh);
  apf::destroyField(uH_h);
  apf::destroyField(uh);
  apf::destroyField(uH);

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
