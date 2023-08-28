#include <PCU.h>
#include <ma.h>
#include <lionPrint.h>
#include <Teuchos_YamlParameterListHelpers.hpp>
#include "adjoint.hpp"
#include "arrays.hpp"
#include "control.hpp"
#include "dbcs.hpp"
#include "defines.hpp"
#include "evaluations.hpp"
#include "fields.hpp"
#include "global_residual.hpp"
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
    double solve_primal_coarse();
    void prepare_fine_space();
    double solve_primal_fine();
    void solve_adjoint_fine();
    void compute_local_errors();
    double sum_local_errors();

    void set_error();
    void clean_up_fine_space();
    void adapt_mesh(int cycle);
    void rebuild_coarse_space();
    void print_final_summary();

    RCP<ParameterList> m_params;
    RCP<State> m_state;
    RCP<NestedDisc> m_nested;
    RCP<Primal> m_primal_coarse;
    RCP<Primal> m_primal_fine;
    RCP<Adjoint> m_adjoint;

    int m_ncycles = 1;
    Array1D<double> m_eta;
    Array1D<double> m_J_H;
    Array1D<double> m_J_h;
    Array1D<double> m_nnodes_H;
};

Driver::Driver(std::string const& input_file) {
  print("reading input: %s", input_file.c_str());
  m_params = rcp(new ParameterList);
  Teuchos::updateParametersFromYamlFile(input_file, m_params.ptr());
  m_state = rcp(new State(*m_params));
  m_primal_coarse = rcp(new Primal(m_params, m_state, m_state->disc));
  ALWAYS_ASSERT(m_state->qoi != Teuchos::null);
  if (m_params->isSublist("adaptivity")) {
    auto adapt_params = m_params->sublist("adaptivity", true);
    m_ncycles = adapt_params.get<int>("solve cycles");
  }
}

double Driver::solve_primal_coarse() {
  print("SOLVING PRIMAL COARSE");
  ParameterList problem_params = m_params->sublist("problem", true);
  int const nsteps = m_state->disc->num_time_steps();
  double J = 0.;
  for (int step = 1; step <= nsteps; ++step) {
    m_primal_coarse->solve_at_step(step);
    J += eval_qoi(m_state, m_state->disc, step);
  }
  J = PCU_Add_Double(J);
  return J;
}

void Driver::prepare_fine_space() {
  print("PREPARING FINE SPACE");
  RCP<Disc> disc = m_state->disc;
  disc->destroy_data();
  m_nested = rcp(new NestedDisc(disc, VERIFICATION));
  m_primal_fine = rcp(new Primal(m_params, m_state, m_nested));
  auto global = m_state->residuals->global;
  int const nr = global->num_residuals();
  Array1D<int> const neq = global->num_eqs();
  m_nested->build_data(nr, neq);
  m_nested->create_verification_data();
  m_state->la->destroy_data();
  m_state->la->build_data(m_nested);
}

double Driver::solve_primal_fine() {
  print("SOLVING PRIMAL FINE");
  ParameterList problem_params = m_params->sublist("problem", true);
  int const nsteps = m_state->disc->num_time_steps();
  double J = 0.;
  for (int step = 1; step <= nsteps; ++step) {
    m_primal_fine->solve_at_step(step);
    J += eval_qoi(m_state, m_nested, step);
  }
  J = PCU_Add_Double(J);
  return J;
}

void Driver::solve_adjoint_fine() {
  print("SOLVING ADJOINT");
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
  apf::Mesh* m = m_nested->apf_mesh();
  apf::Field* R_error = apf::createStepField(m, "R_error", apf::SCALAR);
  apf::Field* C_error = apf::createStepField(m, "C_error", apf::SCALAR);
  apf::zeroField(R_error);
  apf::zeroField(C_error);
  int const nsteps = m_nested->primal().size();
  for (int step = 1; step < nsteps; ++step) {
    Array1D<apf::Field*> zfields = m_nested->adjoint(step).global;
    eval_exact_errors(m_state, m_nested, R_error, C_error, step);
  }
}

double Driver::sum_local_errors() {
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
  eta = PCU_Add_Double(eta);
  eta_bound = PCU_Add_Double(eta_bound);
  m->end(elems);
  print("eta ~ %.16e", eta);
  print("|eta| < %.16e", eta_bound);
  return eta;
}

void Driver::set_error() {
  apf::Mesh* m = m_nested->apf_mesh();
  apf::Field* R_error = m->findField("R_error");
  apf::Field* C_error = m->findField("C_error");
  m_nested->set_error(R_error, C_error);
  apf::destroyField(R_error);
  apf::destroyField(C_error);
}

static apf::Field* get_size_field(apf::Field* error, int cycle, ParameterList& p) {
  double const scale = p.get<double>("target growth", 1.0);
  int target = p.get<int>("target elems");
  target = int(target * std::pow(scale, cycle));
  return get_iso_target_size(error, target);
}

static void configure_ma(ma::Input* in, ParameterList& p) {
  int const adapt_iters = p.get<int>("adapt iters", 1);
  bool const should_coarsen = p.get<bool>("should coarsen", false);
  bool const fix_shape = p.get<bool>("fix shape", false);
  double const good_quality = p.get<double>("good quality", 0.5);
  print("ma inputs----");
  print(" > adapt iters: %d", adapt_iters);
  print(" > should coarsen: %d", should_coarsen);
  print(" > fix shape: %d", fix_shape);
  print(" > good quality: %f", good_quality);
  print("----");
  in->maximumIterations = adapt_iters;
  in->shouldCoarsen = should_coarsen;
  in->shouldFixShape = fix_shape;
  in->goodQuality = good_quality;
  in->shouldRunPreParma = true;
  in->shouldRunMidParma = true;
  in->shouldRunPostParma = true;
}

void Driver::adapt_mesh(int cycle) {
  print("ADAPTING MESH");
  ParameterList adapt_params = m_params->sublist("adaptivity", true);
  apf::Mesh2* m = m_state->disc()->apf_mesh();
  apf::Field* error = m->findField("error");
  apf::Field* size = get_size_field(error, cycle, adapt_params);
  auto in = ma::makeAdvanced(ma::configure(m, size));
  configure_ma(in, adapt_params);
  ma::adapt(in);
  apf::destroyField(size);
}

void Driver::clean_up_fine_space() {
  m_state->disc->destroy_primal();
  m_state->disc->destroy_adjoint();
  m_state->la->destroy_data();
  m_primal_fine = Teuchos::null;
  m_nested = Teuchos::null;
}

void Driver::rebuild_coarse_space() {
  RCP<Disc> disc = m_state->disc;
  int const ngr = m_state->residuals->global->num_residuals();
  Array1D<int> const neqs = m_state->residuals->global->num_eqs();
  disc->build_data(ngr, neqs);
  m_state->la->build_data(disc);
}

static void write_primal_files(RCP<State> state, int cycle, RCP<ParameterList> p) {
  auto problem_params = p->sublist("problem", true);
  std::string name = problem_params.get<std::string>("name");
  name += "_primal_cycle_" + std::to_string(cycle);
  apf::writeVtkFiles(name.c_str(), state->disc->apf_mesh());
}

static void write_nested_files(RCP<NestedDisc> disc, int cycle, RCP<ParameterList> p) {
  auto problem_params = p->sublist("problem", true);
  std::string name = problem_params.get<std::string>("name");
  name += "_nested_cycle_" + std::to_string(cycle);
  apf::writeVtkFiles(name.c_str(), disc->apf_mesh());
}

void Driver::print_final_summary() {
  ALWAYS_ASSERT(m_J_H.size() == m_eta.size());
  print("*******************************************");
  print(" FINAL SUMMARY\n");
  print("*******************************************");
  print("step | nodes_H | J_H | J_h | eta");
  print("--------------------------------");
  for (size_t step = 0; step < m_J_H.size(); ++step) {
    int const nnodes = m_nnodes_H[step];
    double const JH = m_J_H[step];
    double const Jh = m_J_h[step];
    double const eta = m_eta[step];
    print("%d | %d | %.15e | %.15e | %.15e", step, nnodes, JH, Jh, eta);
  }
}

void Driver::drive() {
  for (int cycle = 0; cycle < m_ncycles; ++cycle) {
    print("****** solve-adapt cycle: %d", cycle);
    double const J_H = solve_primal_coarse();
    double nnodes = apf::countOwned(m_state->disc->apf_mesh(), 0);
    nnodes = PCU_Add_Double(nnodes);
    m_J_H.push_back(J_H);
    m_nnodes_H.push_back(nnodes);
    prepare_fine_space();
    double const J_h = solve_primal_fine();
    m_J_h.push_back(J_h);
    solve_adjoint_fine();
    compute_local_errors();
    double const eta = sum_local_errors();
    m_eta.push_back(eta);
    write_nested_files(m_nested, cycle, m_params);
    set_error();
    write_primal_files(m_state, cycle, m_params);
    clean_up_fine_space();
    if (cycle < (m_ncycles - 1)) {
      adapt_mesh(cycle);
      rebuild_coarse_space();
    }
  }
  print_final_summary();
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
