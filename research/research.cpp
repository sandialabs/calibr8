#include <Teuchos_YamlParameterListHelpers.hpp>
#include <lionPrint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "adapt.hpp"
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
    RCP<Adapt> m_adapt;
};

Driver::Driver(std::string const& in) {
  print("reading input: %s", in.c_str());
  m_params = rcp(new ParameterList);
  Teuchos::updateParametersFromYamlFile(in, m_params.ptr());
  m_physics = rcp(new Physics(m_params));
  m_error = create_error(m_params->sublist("error"));
  m_adapt = create_adapt(m_params->sublist("adapt"));
}

void Driver::drive() {
  ParameterList const error_params = m_params->sublist("error");
  ParameterList const adapt_params = m_params->sublist("adapt");
  std::string const output = error_params.get<std::string>("output");
  int const nadapt = adapt_params.get<int>("num iterations");
  mkdir(output.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  for (int adapt_ctr = 1;  adapt_ctr <=  nadapt; ++adapt_ctr) {
    m_physics->build_disc();
    apf::Field* eta = m_error->compute_error(m_physics);
    m_error->write_mesh(m_physics, output, adapt_ctr);
    m_error->destroy_intermediate_fields();
    m_physics->destroy_disc();
    if (adapt_ctr != nadapt) {
      m_adapt->adapt(adapt_params, m_physics);
    }
  }
  m_error->write_pvd(output, nadapt);
  m_error->write_history(output, 0.);
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
