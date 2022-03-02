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
    void prepare_fine_space(bool truth);
    void solve_adjoint();
    void estimate_error();
    double sum_error_estimate();
    void set_error();
    void clean_up_fine_space();
    void adapt_mesh(int cycle);
    void rebuild_coarse_space();
    double solve_primal_fine();
    void print_final_summary();
  private:
    int m_ncycles = 1;
    RCP<ParameterList> m_params;
    RCP<State> m_state;
    RCP<NestedDisc> m_nested;
    RCP<Primal> m_primal;
    RCP<Adjoint> m_adjoint;
  private:
    bool m_solve_exact = false;
    Array1D<double> m_eta;
    Array1D<double> m_J_H;
    Array1D<double> m_nnodes;
    double m_J_exact;
};

Driver::Driver(std::string const& input_file) {
  print("reading input: %s", input_file.c_str());
  m_params = rcp(new ParameterList);
  Teuchos::updateParametersFromYamlFile(input_file, m_params.ptr());
  m_state = rcp(new State(*m_params));
  m_primal = rcp(new Primal(m_params, m_state, m_state->disc));
  ALWAYS_ASSERT(m_state->qoi != Teuchos::null);
  if (m_params->isSublist("adaptivity")) {
    auto adapt_params = m_params->sublist("adaptivity", true);
    m_ncycles = adapt_params.get<int>("solve cycles");
    m_solve_exact = adapt_params.get<bool>("solve exact", false);
  }
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

void Driver::prepare_fine_space(bool truth = false) {
  print("PREPARING FINE SPACE");
  RCP<Disc> disc = m_state->disc;
  disc->destroy_data();
  if (!truth) m_nested = rcp(new NestedDisc(disc));
  else m_nested = rcp(new NestedDisc(disc, TRUTH));
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
  RCP<VectorT>& z = m_state->la->x[OWNED];
  RCP<VectorT>& R = m_state->la->b[OWNED];
  RCP<VectorT>& R_ghost = m_state->la->b[GHOST];
  RCP<MatrixT>& dR_dx = m_state->la->A[OWNED];
  apf::Field* R_error = apf::createStepField(m, "R_error", apf::SCALAR);
  apf::Field* C_error = apf::createStepField(m, "C_error", apf::SCALAR);
  apf::zeroField(R_error);
  apf::zeroField(C_error);
  int const nsteps = m_nested->primal().size();
  ParameterList problem_params = m_params->sublist("problem", true);
  double const dt = problem_params.get<double>("step size");
  ParameterList& tbcs = m_params->sublist("traction bcs");
  ParameterList& dbcs = m_params->sublist("dirichlet bcs", true);
  double t = dt;
  double e = 0.;
  for (int step = 1; step < nsteps; ++step) {
    Array1D<apf::Field*> zfields = m_nested->adjoint(step).global;
    m_state->la->resume_fill_A();
    m_state->la->zero_all();
    eval_error_contributions(m_state, m_nested, R_error, C_error, step);
    eval_tbcs_error_contributions(tbcs, m_nested, zfields, R_error, t);
    apply_primal_tbcs(tbcs, m_nested, R_ghost, t);
    m_state->la->gather_b();
    m_state->la->gather_x();
    apply_primal_dbcs(dbcs, m_nested, dR_dx, R, zfields, t, /*is_adjoint=*/true);
    t += dt;
    e += (R->dot(*(z)));
  }
  e = PCU_Add_Double(e);
  print("eta ~ %.16e", e);
}

double Driver::sum_error_estimate() {
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

double Driver::solve_primal_fine() {
  RCP<Primal> primal = rcp(new Primal(m_params, m_state, m_nested));
  ParameterList problem_params = m_params->sublist("problem", true);
  int const nsteps = problem_params.get<int>("num steps");
  double const dt = problem_params.get<double>("step size");
  double t = 0;
  double J = 0.;
  for (int step = 1; step <= nsteps; ++step) {
    t += dt;
    primal->solve_at_step(step, t, dt);
    J += eval_qoi(m_state, m_nested, step);
  }
  J = PCU_Add_Double(J);
  print("J^h: %.16e\n", J);
  return J;
}

void Driver::print_final_summary() {
  ALWAYS_ASSERT(m_J_H.size() == m_eta.size());
  if (m_solve_exact) {
    print("*******************************************");
    print(" FINAL SUMMARY\n");
    print("*******************************************");
    print("step | nodes | J_ex  | J_H  | eta  | I");
    print("--------------------------------");
    for (size_t step = 0; step < m_J_H.size(); ++step) {
      int const nnodes = m_nnodes[step];
      double const JH = m_J_H[step];
      double const J_ex = m_J_exact;
      double const eta = m_eta[step];
      double const I = eta / (J_ex - JH);
      print("%d | %d | %.15e | %.15e | %.15e | %.15e", step, nnodes, J_ex, JH, eta, I);
    }
  } else {
    print("*******************************************");
    print(" FINAL SUMMARY\n");
    print("*******************************************");
    print("step | nodes | J_H  | eta");
    print("--------------------------------");
    for (size_t step = 0; step < m_J_H.size(); ++step) {
      int const nnodes = m_nnodes[step];
      double const JH = m_J_H[step];
      double const eta = m_eta[step];
      print("%d | %d | %.15e | %.15e", step, nnodes, JH, eta);
    }
  }
}

void Driver::drive() {
  for (int cycle = 0; cycle < m_ncycles; ++cycle) {
    print("****** solve-adapt cycle: %d", cycle);
    double const J = solve_primal();
    double nnodes = m_state->disc->apf_mesh()->count(0);
    nnodes = PCU_Add_Double(nnodes);
    m_J_H.push_back(J);
    m_nnodes.push_back(nnodes);
    write_primal_files(m_state, cycle, m_params);
    prepare_fine_space();
    solve_adjoint();
    estimate_error();
    double const eta = sum_error_estimate();
    m_eta.push_back(eta);
    write_nested_files(m_nested, cycle, m_params);
    set_error();
    clean_up_fine_space();
    if (cycle < (m_ncycles - 1)) {
      adapt_mesh(cycle);
      rebuild_coarse_space();
    }
  }
  if (m_solve_exact) {
    prepare_fine_space(true);
    m_J_exact = solve_primal_fine();
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
