#include <Teuchos_YamlParameterListHelpers.hpp>
#include <lionPrint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "control.hpp"
#include "error.hpp"
#include "physics.hpp"

using namespace calibr8;

class Driver {
  public:
    Driver(std::string const& input_file);
    void drive();
  private:
    RCP<ParameterList> m_params;
    RCP<Physics> m_physics;
    RCP<Error> m_error;
};

Driver::Driver(std::string const& in) {
  print("reading input: %s", in.c_str());
  m_params = rcp(new ParameterList);
  Teuchos::updateParametersFromYamlFile(in, m_params.ptr());
  m_physics = rcp(new Physics(m_params));
  m_error = create_error(m_params->sublist("error"));
}

void Driver::drive() {
  ParameterList const error_params = m_params->sublist("error");
  std::string const output = error_params.get<std::string>("output");
  int check = mkdir(output.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  { // loop over adaptive cycles here
    m_physics->build_disc();
    apf::Field* eta = m_error->compute_error(m_physics);
    m_error->write_mesh(m_physics, output, 0);
    m_error->destroy_intermediate_fields();
    m_physics->destroy_disc();
    //adapt mesh here if ncycles is bigger than 1
  }
  m_error->write_pvd(output, 1);
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
