#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <Teuchos_YamlParameterListHelpers.hpp>
#include <lionPrint.h>
#include <PCU.h>
#include "arrays.hpp"
#include "control.hpp"
#include "defines.hpp"
#include "evaluations.hpp"
#include "global_residual.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "primal.hpp"
#include "state.hpp"
#include "synthetic.hpp"

using namespace calibr8;

class Objective
{
  public:
    int num_problems = 0;
    std::vector<double> QoIs;
    std::vector<ParameterList> params;
    std::vector<RCP<State>> states;
    std::vector<RCP<Primal>> primals;
};

void solve_primal(RCP<State> state, RCP<Primal> primal) {
  int const nsteps = state->disc->num_time_steps();
  print("nsteps = %d", nsteps);
  double J = 0.;
  for (int step = 1; step <= nsteps; ++step) {
    primal->solve_at_step(step);
    J += eval_qoi(state, state->disc, step);
  }
  J = PCU_Add_Double(J);
  print("J: %.16e\n", J);
}

int main(int argc, char** argv)
{
  initialize();
  ALWAYS_ASSERT(argc == 2);
  {
    lion_set_verbosity(1);
    std::string const yaml_input = argv[1];
    auto params_list = rcp(new ParameterList);
    auto objective = rcp(new Objective);
    Teuchos::updateParametersFromYamlFile(yaml_input, params_list.ptr());

    // set up the 'objective' function from the inputs
    auto problems_list = params_list->sublist("problems");
    for (auto problem_entry : problems_list) {
      auto problem_list = Teuchos::getValue<ParameterList>(problem_entry.second);
      auto rcp_prob_list = RCP(new ParameterList);
      *rcp_prob_list = problem_list;
      auto state = rcp(new State(problem_list));
      auto primal = rcp(new Primal(rcp_prob_list, state, state->disc));
      objective->states.push_back(state);
      objective->primals.push_back(primal);
      objective->params.push_back(problem_list);
    }
    objective->num_problems = objective->states.size();
    objective->QoIs.resize(objective->num_problems);

    // go through a forward pass and solve the primal
    // problems as built by the 'objective'
    for (int i = 0; i < objective->num_problems; ++i) {
      auto state = objective->states[i];
      auto primal = objective->primals[i];
      solve_primal(state, primal);
    }

  }
  finalize();
}
