#include <PCU.h>
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
    void estimate_error();
    void print_error_estimate();
    void evaluate_linearization_error();
  private:
    RCP<ParameterList> m_params;
    RCP<State> m_state;
    RCP<NestedDisc> m_nested;
    RCP<Primal> m_primal;
    RCP<Adjoint> m_adjoint;
    double m_J_H = 0.;
    double m_J_h = 0.;
    double m_E_lin_R = 0.;
    double m_E_lin_C = 0.;
};

Driver::Driver(std::string const& input_file) {
  print("reading input: %s", input_file.c_str());
  m_params = rcp(new ParameterList);
  Teuchos::updateParametersFromYamlFile(input_file, m_params.ptr());
  m_state = rcp(new State(*m_params));
  ALWAYS_ASSERT(m_state->qoi != Teuchos::null);
}

void Driver::solve_primal_coarse() {
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
  ParameterList problem_params = m_params->sublist("problem", true);
  ParameterList& tbcs = m_params->sublist("traction bcs");
  double const dt = problem_params.get<double>("step size");
  double t = dt;
  for (int step = 1; step < nsteps; ++step) {
    print(" > at error step: %d", step);
    Array1D<apf::Field*> zfields = m_nested->adjoint(step).global;
    eval_error_contributions(m_state, m_nested, R_error, C_error, step);
    eval_tbcs_error_contributions(tbcs, m_nested, zfields, R_error, t);
  }
}

void Driver::evaluate_linearization_error() {
  print("EVALUATING LINEARIZATION ERROR");
  m_E_lin_R = 0.;
  m_E_lin_C = 0.;
  int const nsteps = m_nested->primal().size();
  apf::Mesh* m = m_nested->apf_mesh();
  ParameterList problem_params = m_params->sublist("problem", true);
  ParameterList& tbcs = m_params->sublist("traction bcs");
  double const dt = problem_params.get<double>("step size");
  double t = dt;
  for (int step = 1; step < nsteps; ++step) {
    print(" > at error linearization error: %d", step);
    Array1D<apf::Field*> zfields = m_nested->adjoint(step).global;
    eval_linearization_errors(m_state, m_nested, step, m_E_lin_R, m_E_lin_C);
    m_E_lin_R += sum_tbcs_error_contributions(tbcs, m_nested, zfields, t);
    t += dt;
  }
  m_E_lin_R = PCU_Add_Double(m_E_lin_R);
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
  double const E_exact = m_J_h - m_J_H;
  double const I = eta / E_exact;
  print("E_exact: %.16e", E_exact);
  print("I: %.16e", I);
  print("E_lin_R: %.16e", m_E_lin_R);
  print("E_lin_C: %.16e", m_E_lin_C);
  double const E_computed = eta + m_E_lin_R + m_E_lin_C;
  print("E_computed - E_exact: %.16e", E_computed - E_exact);
  print("E_computed / E_exact: %.16e", E_computed / E_exact);

  ParameterList problem_params = m_params->sublist("problem", true);
  bool check = problem_params.get<bool>("do regression", false);
  if (check) {
    std::cout << "------ regression summary -----\n";
    if (std::abs((E_computed / E_exact) - 1.0) < 1.e-8) {
      std::cout << " PASS\n";
    } else {
      std::cout << " FAIL\n";
      abort();
    }
    std::cout << "-------------------------------\n";
  }
}

void Driver::drive() {
  solve_primal_coarse();
  prepare_fine_space();
  solve_primal_fine();
  solve_adjoint_fine();
  estimate_error();
  evaluate_linearization_error();
  print_error_estimate();
  apf::writeVtkFiles("verify", m_nested->apf_mesh());
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
