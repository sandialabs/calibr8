#include <Teuchos_YamlParameterListHelpers.hpp>
#include <lionPrint.h>
#include "control.hpp"
#include "disc.hpp"

using namespace calibr8;

int main(int argc, char** argv) {
  initialize();
  ALWAYS_ASSERT(argc == 2);
  {
    lion_set_verbosity(1);
    std::string const yaml_input = argv[1];
    print("reading input: %s", yaml_input.c_str());
    RCP<ParameterList> params = rcp(new ParameterList);
    Teuchos::updateParametersFromYamlFile(yaml_input, params.ptr());
    ParameterList const disc_params = params->sublist("discretization");
    RCP<Disc> disc = rcp(new Disc(disc_params));
    disc->build_data(1);
    disc->destroy_data();
  }
  finalize();
}
