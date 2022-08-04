#include <PCU.h>
#include <lionPrint.h>
#include <Teuchos_YamlParameterListHelpers.hpp>
#include "adjoint.hpp"
#include "control.hpp"
#include "dbcs.hpp"
#include "defines.hpp"
#include "evaluations.hpp"
#include "fields.hpp"
#include "global_residual.hpp"
#include "macros.hpp"
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
    void solve_primal_coarse();
    void prepare_fine_space();
    void solve_primal_fine();
    void solve_adjoint_fine();
    void compute_local_errors();
    void sum_local_errors();
  private:
    RCP<ParameterList> m_params;
    RCP<State> m_state;
    RCP<NestedDisc> m_nested;
    RCP<Primal> m_primal;
    RCP<Adjoint> m_adjoint;
    double m_J_H = 0.;
    double m_J_h = 0.;
};

Driver::Driver(std::string const& input_file) {
  print("reading input: %s", input_file.c_str());
  m_params = rcp(new ParameterList);
  Teuchos::updateParametersFromYamlFile(input_file, m_params.ptr());
  m_state = rcp(new State(*m_params));
  ALWAYS_ASSERT(m_state->qoi != Teuchos::null);
}

void Driver::solve_primal_coarse() {
  print("SOLVING PRIMAL COARSE");
  m_primal = rcp(new Primal(m_params, m_state, m_state->disc));
  ParameterList problem_params = m_params->sublist("problem", true);
  int const nsteps = problem_params.get<int>("num steps");
  double const dt = problem_params.get<double>("step size");
  double t = 0;
  m_J_H = 0.;
  for (int step = 1; step <= nsteps; ++step) {
    t += dt;
    m_primal->solve_at_step(step, t, dt);
    m_J_H += eval_qoi(m_state, m_state->disc, step);
  }
  m_J_H = PCU_Add_Double(m_J_H);
  print("J^H: %.16e\n", m_J_H);
}

void Driver::prepare_fine_space() {
  print("PREPARING FINE SPACE");
  RCP<Disc> disc = m_state->disc;
  disc->destroy_data();
  m_nested = rcp(new NestedDisc(disc, VERIFICATION));
  auto global = m_state->residuals->global;
  int const nr = global->num_residuals();
  Array1D<int> const neq = global->num_eqs();
  m_nested->build_data(nr, neq);
  m_nested->create_verification_data();
  m_state->la->destroy_data();
  m_state->la->build_data(m_nested);
}

void Driver::solve_primal_fine() {
  print("SOLVING PRIMAL FINE");
  m_primal = rcp(new Primal(m_params, m_state, m_state->disc));
  m_primal = rcp(new Primal(m_params, m_state, m_nested));
  ParameterList problem_params = m_params->sublist("problem", true);
  int const nsteps = problem_params.get<int>("num steps");
  double const dt = problem_params.get<double>("step size");
  double t = 0;
  m_J_h = 0.;
  for (int step = 1; step <= nsteps; ++step) {
    t += dt;
    m_primal->solve_at_step(step, t, dt);
    m_J_h += eval_qoi(m_state, m_nested, step);
  }
  m_J_h = PCU_Add_Double(m_J_h);
  print("J^h: %.16e\n", m_J_h);
}

void Driver::solve_adjoint_fine() {
  print("SOLVING ADJOINT FINE");
  m_adjoint = rcp(new Adjoint(m_params, m_state, m_nested));
  int const nsteps = m_nested->primal().size() - 1;
  auto residuals = m_state->residuals;
  m_nested->create_adjoint(residuals, nsteps);
  for (int step = nsteps; step > 0; --step) {
    m_adjoint->solve_at_step(step);
  }
  m_adjoint = Teuchos::null;
}

void Driver::compute_local_errors() {
  print("COMPUTING LOCAL ERRORS");
  apf::Field* R_error = apf::createStepField(m, "R_error", apf::SCALAR);
  apf::Field* C_error = apf::createStepField(m, "C_error", apf::SCALAR);
  apf::zeroField(R_error);
  apf::zeroField(C_error);
  for (int step = 1; step < nsteps; ++step) {
    Array1D<apf::Field*> zfields = m_nested->adjoint(step).global;
    eval_exact_errors(m_state, m_nested, R_error, C_error, step);
    // might need some tbc stuff here
  }
}
