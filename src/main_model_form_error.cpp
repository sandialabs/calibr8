#include <PCU.h>
#include <ma.h>
#include <lionPrint.h>
#include <Teuchos_YamlParameterListHelpers.hpp>
#include "adjoint.hpp"
#include "arrays.hpp"
#include "control.hpp"
#include "defines.hpp"
#include "dbcs.hpp"
#include "evaluations.hpp"
#include "fields.hpp"
#include "global_residual.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "mesh_size.hpp"
#include "nested.hpp"
#include "primal.hpp"
#include "state.hpp"
#include "tbcs.hpp"

using namespace calibr8;

class Driver {
  public:
    Driver(std::string const& input_file);
    void drive();
  private:
    double solve_primal();
    void prepare_fine_space();
    void solve_adjoint();
    void compute_local_errors();
    double sum_local_errors();
  private:
    int m_ncycles = 1;
    RCP<ParameterList> m_params;
    RCP<State> m_state;
    RCP<Primal> m_primal;
    RCP<Adjoint> m_adjoint;
};

Driver::Driver(std::string const& input_file) {
  print("reading input: %s", input_file.c_str());
  m_params = rcp(new ParameterList);
  Teuchos::updateParametersFromYamlFile(input_file, m_params.ptr());
  m_state = rcp(new State(*m_params));
  m_primal = rcp(new Primal(m_params, m_state, m_state->disc));
  ALWAYS_ASSERT(m_state->qoi != Teuchos::null);
}

double Driver::solve_primal() {
  ParameterList problem_params = m_params->sublist("problem", true);
  int const nsteps = problem_params.get<int>("num steps");
  double const dt = problem_params.get<double>("step size");
  double t = 0;
  double J = 0;
  for (int step = 1; step <= nsteps; ++step) {
    t += dt;
    m_primal->solve_at_step(step, t, dt);
    J += eval_qoi(m_state, m_state->disc, step);
  }
  J = PCU_Add_Double(J);
  print("J^H: %.16e\n", J);
  return J;
}

void Driver::prepare_fine_space() {
  print("PREPARING FINE MODEL\n");
  m_state->model_form = FINE_MODEL;
  auto disc = m_state->disc;
  auto residuals = m_state->residuals;
  auto d_residuals = m_state->d_residuals;
  residuals->local[FINE_MODEL]->init_variables(m_state, false);
  d_residuals->local[FINE_MODEL]->init_variables(m_state, false);
  int const nsteps = disc->primal().size();
  // prolong local[BASE_MODEL] to local[FINE_MODEL]
  disc->create_primal_fine_model(residuals, nsteps);
}

void Driver::solve_adjoint() {
  auto disc = m_state->disc;
  m_adjoint = rcp(new Adjoint(m_params, m_state, disc));
  int const nsteps = disc->primal().size() - 1;
  auto residuals = m_state->residuals;
  disc->create_adjoint(residuals, nsteps, FINE_MODEL);
  for (int step = nsteps; step > 0; --step) {
    m_adjoint->solve_at_step(step);
  }
  m_adjoint = Teuchos::null;
}

void Driver::compute_local_errors() {
  print("COMPUTING LOCAL ERRORS");
  auto disc = m_state->disc;
  apf::Mesh* m = disc->apf_mesh();
  apf::Field* R_error = apf::createStepField(m, "R_error", apf::SCALAR);
  apf::Field* C_error = apf::createStepField(m, "C_error", apf::SCALAR);
  apf::zeroField(R_error);
  apf::zeroField(C_error);
  int const nsteps = disc->primal().size();
  for (int step = 1; step < nsteps; ++step) {
    Array1D<apf::Field*> zfields = disc->adjoint(step).global;
    eval_error_contributions(m_state, disc, R_error, C_error, step);
  }
}

double Driver::sum_local_errors() {
  auto disc = m_state->disc;
  apf::Mesh* m = disc->apf_mesh();
  apf::Field* R_error = m->findField("R_error");
  apf::Field* C_error = m->findField("C_error");
  double eta = 0;
  double eta_bound = 0.;
  apf::MeshEntity* elem;
  apf::MeshIterator* elems = m->begin(m->getDimension());
  while ((elem = m->iterate(elems))) {
    double const R_val = apf::getScalar(R_error, elem, 0);
    double const C_val = apf::getScalar(C_error, elem, 0);
    double const val = R_val + C_val;
    eta += val;
    eta_bound += std::abs(val);
  }
  eta = PCU_Add_Double(eta);
  eta_bound = PCU_Add_Double(eta_bound);
  m->end(elems);
  print("eta ~ %.16e", eta);
  print("|eta| < %.16e", eta_bound);
  return eta;
}

void Driver::drive() {
  double const J = solve_primal();
  prepare_fine_space();
  solve_adjoint();
  compute_local_errors();
  double const eta = sum_local_errors();
}

int main(int argc, char** argv) {
  initialize();
  ALWAYS_ASSERT(argc == 2);
  {
    lion_set_verbosity(1);
    std::string const yaml_input = argv[1];
    Driver driver(yaml_input);
    driver.drive();
  }
  finalize();
}
