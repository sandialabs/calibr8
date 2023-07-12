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
    void prepare_fine_space(int cycle);
    void solve_adjoint();
    void compute_local_errors();
    double sum_local_errors();
    void adapt_base_model(double const reduction_amount);
    void clean_up_fields();
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
  print("SOLVING PRIMAL WITH BASE MODEL\n");
  m_state->model_form = BASE_MODEL;
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

void Driver::prepare_fine_space(int cycle) {
  print("PREPARING FINE MODEL\n");
  m_state->model_form = FINE_MODEL;
  auto disc = m_state->disc;
  auto residuals = m_state->residuals;
  auto d_residuals = m_state->d_residuals;
  residuals->local[FINE_MODEL]->init_variables(m_state, false);
  d_residuals->local[FINE_MODEL]->init_variables(m_state, false);
  int const nsteps = disc->primal().size();
  // prolong local[BASE_MODEL] to local[FINE_MODEL]
  bool has_fine_ic = false;
  if (cycle > 0) {
    has_fine_ic = true;
  }
  disc->create_primal_fine_model(residuals, nsteps, has_fine_ic);
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
  apf::Field* abs_error = apf::createStepField(m, "abs_error", apf::SCALAR);
  apf::zeroField(abs_error);
  double eta = 0;
  double eta_combined_bound = 0.;
  double eta_C_bound = 0.;
  apf::MeshEntity* elem;
  apf::MeshIterator* elems = m->begin(m->getDimension());
  while ((elem = m->iterate(elems))) {
    double const R_val = apf::getScalar(R_error, elem, 0);
    double const C_val = apf::getScalar(C_error, elem, 0);
    double const val = R_val + C_val;
    eta += val;
    eta_combined_bound += std::abs(val);
    eta_C_bound += std::abs(C_val);
    apf::setScalar(abs_error, elem, 0, std::abs(C_val));
  }
  m->end(elems);
  eta = PCU_Add_Double(eta);
  eta_combined_bound = PCU_Add_Double(eta_combined_bound);
  eta_C_bound = PCU_Add_Double(eta_C_bound);
  print("eta ~ %.16e", eta);
  print("combined |eta| < %.16e", eta_combined_bound);
  print("local residual |eta| < %.16e", eta_C_bound);
  return eta_C_bound;
}

// this will only work in serial
static apf::Field* get_top_elems_by_percent(
    apf::Mesh* m,
    apf::Field* error,
    int percent) {
  std::vector<std::pair<double, apf::MeshEntity*>> errors;
  apf::MeshEntity* elem;
  auto elem_iterator = m->begin(m->getDimension());
  while ((elem = m->iterate(elem_iterator))) {
    double const value = apf::getScalar(error, elem, 0);
    errors.push_back(std::make_pair(value, elem));
  }
  m->end(elem_iterator);
  std::sort(errors.begin(), errors.end());
  double const nelems = m->count(m->getDimension());
  double const factor = percent/100.;
  int const nrefine_elems = int(factor*nelems);
  apf::Field* top = apf::createStepField(m, "top", apf::SCALAR);
  apf::zeroField(top);
  for (int i = 1; i <= nrefine_elems; ++i) {
    apf::setScalar(top, errors[nelems-i].second, 0, 1.);
  }
  return top;
}

// this will only work in serial
static apf::Field* get_top_elems_by_bound_reduction(
    apf::Mesh* m,
    apf::Field* error,
    double bound) {
  std::vector<std::pair<double, apf::MeshEntity*>> errors;
  apf::MeshEntity* elem;
  auto elem_iterator = m->begin(m->getDimension());
  while ((elem = m->iterate(elem_iterator))) {
    double const value = apf::getScalar(error, elem, 0);
    errors.push_back(std::make_pair(value, elem));
  }
  m->end(elem_iterator);
  std::sort(errors.begin(), errors.end());
  double const nelems = m->count(m->getDimension());
  apf::Field* top = apf::createStepField(m, "top", apf::SCALAR);
  apf::zeroField(top);
  int i = 1;
  while (bound > 0.) {
    double const elem_error_bound = apf::getScalar(error, errors[nelems-i].second, 0);
    apf::setScalar(top, errors[nelems-i].second, 0, 1.);
    bound -= elem_error_bound;
    ++i;
  }
  print("%d ELEMENTS MARKED FOR REFINEMENT", i - 1);
  return top;
}

// this will only work for a single element set
void Driver::adapt_base_model(double const reduction_amount) {
  auto disc = m_state->disc;
  apf::Mesh* m = disc->apf_mesh();
  apf::Field* abs_error = m->findField("abs_error");
  auto& base_models = disc->base_local_residuals();
  if (base_models.size() == 0) {
    disc->initialize_base_local_residuals();
  }
  //double const percent_refine = 10.;
  //apf::Field* top = get_top_elems_by_percent(m, abs_error, percent_refine);
  apf::Field* top = get_top_elems_by_bound_reduction(m, abs_error, reduction_amount);
  int const es = 0;
  apf::MeshEntity* elem;
  auto elem_iterator = m->begin(m->getDimension());
  int i = 0;
  while ((elem = m->iterate(elem_iterator))) {
    if (apf::getScalar(top, elem, 0) > 0. ) {
      base_models[es][i++] = FINE_MODEL;
    } else {
      ++i;
    }
  }
  m->end(elem_iterator);
}

void Driver::clean_up_fields() {
  m_state->disc->destroy_primal(true);
  m_state->disc->destroy_adjoint();
  auto m = m_state->disc->apf_mesh();
  apf::Field* R_error = m->findField("R_error");
  apf::Field* C_error = m->findField("C_error");
  apf::Field* abs_error = m->findField("abs_error");
  apf::Field* top = m->findField("top");
  apf::destroyField(R_error);
  apf::destroyField(C_error);
  apf::destroyField(abs_error);
  apf::destroyField(top);
}

void Driver::drive() {
  double const reduction_factor = 0.5;
  for (int i = 0; i < m_ncycles; ++i) {
    double const J = solve_primal();
    prepare_fine_space(i);
    solve_adjoint();
    compute_local_errors();
    double const eta_bound = sum_local_errors();
    adapt_base_model(eta_bound * reduction_factor);
    auto disc = m_state->disc;
    std::string e_name = "error_";
    e_name += std::to_string(i);
    apf::writeVtkFiles(e_name.c_str(), disc->apf_mesh());
    clean_up_fields();
  }
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
