#include <Teuchos_YamlParameterListHelpers.hpp>
#include "adjoint.hpp"
#include "control.hpp"
#include "defines.hpp"
#include "evaluations.hpp"
#include "fields.hpp"
#include "global_residual.hpp"
#include "macros.hpp"
#include "nested.hpp"
#include "primal.hpp"
#include "state.hpp"

using namespace calibr8;

class Driver {
  public:
    Driver(std::string const& input_file);
    void drive();
  private:
    void solve_primal();
    void prepare_fine_space();
    void solve_adjoint();
    void estimate_error();
    void print_error_estimate();
    void clean_up_fine_space();
  private:
    double m_J;
    RCP<ParameterList> m_params;
    RCP<State> m_state;
    RCP<NestedDisc> m_nested;
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

void Driver::solve_primal() {
  ParameterList problem_params = m_params->sublist("problem", true);
  int const nsteps = problem_params.get<int>("num steps");
  double const dt = problem_params.get<double>("step size");
  double t = 0;
  m_J = 0;
  for (int step = 1; step <= nsteps; ++step) {
    t += dt;
    m_primal->solve_at_step(step, t, dt);
    m_J += eval_qoi(m_state, m_state->disc, step);
  }
  print("J^H: %.16e\n", m_J);
}

void Driver::prepare_fine_space() {
  print("PREPARING FINE SPACE");
  RCP<Disc> disc = m_state->disc;
  disc->destroy_data();
  m_nested = rcp(new NestedDisc(disc));
  auto global = m_state->residuals->global;
  int const nr = global->num_residuals();
  Array1D<int> const neq = global->num_eqs();
  m_nested->build_data(nr, neq);
  m_state->la->destroy_data();
  m_state->la->build_data(m_nested);
}

void Driver::solve_adjoint() {
  m_adjoint = rcp(new Adjoint(m_params, m_state, m_nested));
  int const nsteps = m_nested->primal().size() - 1;
  auto residuals = m_state->residuals;
  m_nested->create_adjoint(residuals, nsteps);
  for (int step = nsteps; step > 0; --step) {
    m_adjoint->solve_at_step(step);
  }
  m_adjoint = Teuchos::null;
}

void Driver::estimate_error() {
  print("ESTIMATING ERROR");
  apf::Mesh* m = m_nested->apf_mesh();
  apf::Field* R_error = apf::createStepField(m, "R_error", apf::SCALAR);
  apf::Field* C_error = apf::createStepField(m, "C_error", apf::SCALAR);
  apf::zeroField(R_error);
  apf::zeroField(C_error);
  int const nsteps = m_nested->primal().size();
  for (int step = 1; step < nsteps; ++step) {
    eval_error_contributions(m_state, m_nested, R_error, C_error, step);
  }
}

void Driver::print_error_estimate() {
  apf::Mesh* m = m_nested->apf_mesh();
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
  m->end(elems);
  print("eta ~ %.16e", eta);
  print("|eta| < %.16e", eta_bound);
  ParameterList qoi_params = m_params->sublist("quantity of interest", true);
  if (qoi_params.isParameter("exact")) {
    double const J_exact = qoi_params.get<double>("exact");
    double const E = J_exact - m_J;
    double const I = eta / E;
    print("E_exact: %.16e", E);
    print("I: %.16e", I);
  }
}

void Driver::clean_up_fine_space() {
  apf::writeVtkFiles("debug", m_nested->apf_mesh());
  m_state->disc->destroy_primal();
  m_state->disc->destroy_adjoint();
  m_nested = Teuchos::null;
  // TODO: maybe rebuild coarse disc data
  // and la containers here in anticipation of an
  // adaptive loop
}

void Driver::drive() {
  solve_primal();
  prepare_fine_space();
  solve_adjoint();
  estimate_error();
  print_error_estimate();
  clean_up_fine_space();
}

int main(int argc, char** argv) {
  initialize();
  ALWAYS_ASSERT(argc == 2);
  {
    std::string const yaml_input = argv[1];
    Driver driver(yaml_input);
    driver.drive();
  }
  finalize();
}
