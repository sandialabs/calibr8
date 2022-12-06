#include <lionPrint.h>
//#include <ROL_Algorithm.hpp>
//#include <ROL_Bounds.hpp>
//#include <ROL_LineSearchStep.hpp>
//#include <ROL_Objective.hpp>
//#include <ROL_ParameterList.hpp>
//#include <ROL_Stream.hpp>
#include <PCU.h>
#include <Teuchos_YamlParameterListHelpers.hpp>
#include "arrays.hpp"
#include "control.hpp"
#include "defines.hpp"
#include "evaluations.hpp"
#include "global_residual.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "state.hpp"
#include "virtual_power.hpp"
//#include "vfm_objective.hpp"

using namespace calibr8;

static ParameterList get_valid_params() {
  ParameterList p;
  p.sublist("discretization");
  p.sublist("residuals");
  p.sublist("problem");
  p.sublist("linear algebra");
  p.sublist("quantity of interest");
  p.sublist("virtual fields");
  return p;
}

class Solver {
  public:
    Solver(std::string const& input_file);
    void solve();
    void write_at_end();
    void write_at_step(int step, bool has_adjoint);
  private:
    std::string base_name();
    void write_pvd();
    void write_native();
  private:
    RCP<ParameterList> m_params;
    RCP<State> m_state;
    RCP<VirtualPower> m_virtual_power;
    bool m_eval_qoi = false;
    bool m_eval_regression = false;
    bool m_write_synthetic = true;
    bool m_write_measured = false;
};

Solver::Solver(std::string const& input_file) {
  print("reading input: %s", input_file.c_str());
  m_params = rcp(new ParameterList);
  Teuchos::updateParametersFromYamlFile(input_file, m_params.ptr());
  m_params->validateParameters(get_valid_params(), 0);
  m_state = rcp(new State(*m_params));
  m_virtual_power = rcp(new VirtualPower(m_params, m_state, m_state->disc));
}

void Solver::solve() {
  ParameterList problem_params = m_params->sublist("problem", true);
  std::string const name = problem_params.get<std::string>("name");
  int const nsteps = problem_params.get<int>("num steps");
  double const dt = problem_params.get<double>("step size");
  double t = 0.;
  double J = 0.;
  for (int step = 1; step <= nsteps; ++step) {
    t += dt;
    J += m_virtual_power->compute_at_step(step, t, dt);
  }
  J = PCU_Add_Double(J);
  print("Virtual Power Squared Mismatch: %.16e\n", J);
}

int main(int argc, char** argv) {
  initialize();
  ALWAYS_ASSERT(argc == 2);
  {
    lion_set_verbosity(1);
    std::string const yaml_input = argv[1];
    Solver solver(yaml_input);
    solver.solve();
  }
  finalize();
}
