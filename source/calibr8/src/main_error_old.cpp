#include <Teuchos_YamlParameterListHelpers.hpp>
#include <spr.h>
#include "adjoint.hpp"
#include "control.hpp"
#include "defines.hpp"
#include "evaluations.hpp"
#include "fields.hpp"
#include "macros.hpp"
#include "primal.hpp"
#include "state.hpp"

using namespace calibr8;

class Driver {
  public:
    Driver(std::string const& input_file);
    void drive();
  private:
    void solve_primal();
    void solve_adjoint();
    void prepare_error_estimation();
    void estimate_error();
    void print_error_estimate();
  private:
    double m_J;
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
  m_primal = rcp(new Primal(m_params, m_state));
  m_adjoint = rcp(new Adjoint(m_params, m_state));
  ALWAYS_ASSERT(m_state->qoi != Teuchos::null);
}

void Driver::solve_primal() {
  ParameterList problem_params = m_params->sublist("problem", true);
  std::string const name = problem_params.get<std::string>("name");
  int const nsteps = problem_params.get<int>("num steps");
  double const dt = problem_params.get<double>("step size");
  double t = 0;
  m_J = 0;
  for (int step = 1; step <= nsteps; ++step) {
    t += dt;
    m_primal->solve_at_step(step, t, dt);
    m_J += eval_qoi(m_state, step);
  }
  print("J: %.16e\n", m_J);
}

void Driver::solve_adjoint() {
  ParameterList problem_params = m_params->sublist("problem", true);
  int const nsteps = problem_params.get<int>("num steps");
  m_state->create_all_adjoint_H(nsteps);
  for (int step = nsteps; step > 0; --step) {
    m_adjoint->solve_at_step(step);
  }
}

void Driver::prepare_error_estimation() {
  print("PREPARING ERROR ESTIMATION");
  RCP<Disc> disc = m_state->disc;
  apf::Mesh* m = disc->apf_mesh();
  apf::FieldShape* coarse_shape = disc->gv_shape(COARSE);
  apf::FieldShape* fine_shape = disc->gv_shape(FINE);
  print(" > prolonging primal fields");
  m_state->prolong_primal_H();
  //m_state->destroy_primal_H();
  print(" > enriching adjoint fields");
  m->changeShape(fine_shape);
  m_state->enrich_adjoint_H();
  //m_state->destroy_adjoint_H();
  m->changeShape(coarse_shape);
}

void Driver::estimate_error() {
  print("ESTIMATING ERROR");
  apf::Mesh* m = m_state->disc->apf_mesh();
  apf::Field* R_error = apf::createStepField(m, "R_error", apf::SCALAR);
  apf::Field* C_error = apf::createStepField(m, "C_error", apf::SCALAR);
  apf::zeroField(R_error);
  apf::zeroField(C_error);
  int const nsteps = m_state->primal_h.size();
  for (int step = 1; step < nsteps; ++step) {
    eval_error_contributions(m_state, R_error, C_error, step);
  }
//  m_state->destroy_primal_h();
//  m_state->destroy_adjoint_h();
}

void Driver::print_error_estimate() {
  apf::Mesh* m = m_state->disc->apf_mesh();
  apf::Field* R_error = m->findField("R_error");
  apf::Field* C_error = m->findField("C_error");
  double eta = 0.;
  double eta_bound = 0.;
  apf::MeshEntity* elem;
  apf::MeshIterator* elems = m->begin(3);
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

void Driver::drive() {
  solve_primal();
  solve_adjoint();
  prepare_error_estimation();
  estimate_error();
  print_error_estimate();


  // estimate error
  // debug viz:
  // need to change 'mesh shape' to quadratic to viz higher order
  // nodal fields, unfortunately
  RCP<Disc> disc = m_state->disc;
  apf::Mesh* m = disc->apf_mesh();
  apf::FieldShape* coarse_shape = disc->gv_shape(COARSE);
  apf::FieldShape* fine_shape = disc->gv_shape(FINE);
  apf::writeVtkFiles("debug_coarse", m_state->disc->apf_mesh());
  m->changeShape(fine_shape);
  apf::writeVtkFiles("debug_fine", m_state->disc->apf_mesh());
  m->changeShape(coarse_shape);

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
