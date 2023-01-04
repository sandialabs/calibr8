#include <PCU.h>
#include <ma.h>
#include <lionPrint.h>
#include <Teuchos_YamlParameterListHelpers.hpp>
#include "adjoint.hpp"
#include "arrays.hpp"
#include "control.hpp"
#include "cspr.hpp"
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
    void solve_adjoint();
    void prepare_fine_space();
    void perform_SPR();
    void estimate_error();
  private:
    int m_ncycles = 1;
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
  if (m_params->isSublist("adaptivity")) {
    auto adapt_params = m_params->sublist("adaptivity", true);
    m_ncycles = adapt_params.get<int>("solve cycles");
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

void Driver::solve_adjoint() {
  auto disc = m_state->disc;
  m_adjoint = rcp(new Adjoint(m_params, m_state, disc));
  int const nsteps = disc->primal().size() - 1;
  auto residuals = m_state->residuals;
  disc->create_adjoint(residuals, nsteps);
  for (int step = nsteps; step > 0; --step) {
    m_adjoint->solve_at_step(step);
  }
  m_adjoint = Teuchos::null;
}

static void write_base_files(RCP<State> state, int cycle, RCP<ParameterList> p) {
  auto problem_params = p->sublist("problem", true);
  std::string name = problem_params.get<std::string>("name");
  name += "_base_cycle_" + std::to_string(cycle);
  apf::writeVtkFiles(name.c_str(), state->disc->apf_mesh());
}

static void write_nested_files(RCP<NestedDisc> disc, int cycle, RCP<ParameterList> p) {
  auto problem_params = p->sublist("problem", true);
  std::string name = problem_params.get<std::string>("name");
  name += "_nested_cycle_" + std::to_string(cycle);
  apf::writeVtkFiles(name.c_str(), disc->apf_mesh());
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

static apf::Field* interpolate_to_cell_center(
    apf::Field* nodal,
    std::string const& name) {
  apf::Mesh* m = apf::getMesh(nodal);
  int const neqs = apf::countComponents(nodal);
  int const dim = m->getDimension();
  int const q_order = 1;
  apf::FieldShape* ip_shape = apf::getIPShape(dim, q_order);
  apf::Field* ip = apf::createPackedField(m, name.c_str(), neqs, ip_shape);
  apf::zeroField(ip);
  apf::MeshEntity* elem;
  apf::MeshIterator* elems = m->begin(dim);
  Array1D<double> field_comps(9);
  while ((elem = m->iterate(elems))) {
    apf::MeshElement* me = apf::createMeshElement(m, elem);
    apf::Element* nodal_elem = apf::createElement(nodal, me);
    int const npts = apf::countIntPoints(me, q_order);
    for (int pt = 0; pt < npts; ++pt) {
      apf::Vector3 iota;
      apf::getIntPoint(me, q_order, pt, iota);
      apf::getComponents(nodal_elem, iota, &(field_comps[0]));
      apf::setComponents(ip, elem, pt, &(field_comps[0]));
    }
    apf::destroyElement(nodal_elem);
    apf::destroyMeshElement(me);
  }
  m->end(elems);
  return ip;
}

static double mysubtract(double a, double b) { return a-b; }

apf::Field* op(
    std::function<double(double,double)> f,
    RCP<Disc> disc,
    apf::Field* a,
    apf::Field* b,
    std::string const& name) {
  ALWAYS_ASSERT(apf::getMesh(a) == apf::getMesh(b));
  ALWAYS_ASSERT(apf::getMesh(b) == disc->apf_mesh());
  ALWAYS_ASSERT(apf::getShape(a) == apf::getShape(b));
  ALWAYS_ASSERT(apf::countComponents(a) == apf::countComponents(b));
  apf::Mesh* mesh = apf::getMesh(a);
  apf::FieldShape* shape = apf::getShape(a);
  int const neqs = apf::countComponents(a);
  apf::Field* result = apf::createPackedField(mesh, name.c_str(), neqs, shape);
  apf::DynamicArray<apf::Node> nodes = disc->get_owned_nodes();
  std::vector<double> a_vals(neqs, 0.);
  std::vector<double> b_vals(neqs, 0.);
  std::vector<double> result_vals(neqs, 0.);
  for (auto& node : nodes) {
    apf::MeshEntity* ent = node.entity;
    int const local_node = node.node;
    apf::getComponents(a, ent, local_node, &(a_vals[0]));
    apf::getComponents(b, ent, local_node, &(b_vals[0]));
    for (int eq = 0; eq < neqs; ++eq) {
      result_vals[eq] = f(a_vals[eq], b_vals[eq]);
    }
    apf::setComponents(result, ent, local_node, &(result_vals[0]));
  }
  apf::synchronize(result);
  return result;
}

void Driver::perform_SPR() {
  print("IT'S SPR TIME");
  std::vector<apf::Field*> diffs;
  auto& adjoint = m_nested->adjoint();
  int const nsteps = adjoint.size();
  for (int step = 0; step < nsteps; ++step) {
    auto& fields = adjoint[step].global;
    int const num_gr = fields.size();
    for (int i = 0; i < num_gr; ++i) {
      auto f = fields[i];
      std::string const f_cell_name = std::string(apf::getName(f)) + "_e";
      std::string const f_diff_name = std::string(apf::getName(f)) + "_diff";
      auto f_cell = interpolate_to_cell_center(f, f_cell_name);
      auto f_spr = spr_recovery(f_cell); // f_e is deleted here
      auto f_diff = op(mysubtract, m_nested, f_spr, f, f_diff_name);
//      apf::destroyField(f_spr);
//      apf::destroyField(f);
//      fields[i] = f_diff;
      fields[i] = f_spr;
    }
  }
}

void Driver::estimate_error() {
  print("ESTIMATING THE ERROR");
  int const nsteps = m_nested->adjoint().size();

  Array2D<RCP<MatrixT>>& dR_dx = m_state->la->A[OWNED];
  Array1D<RCP<VectorT>>& R = m_state->la->b[OWNED];
  Array1D<RCP<VectorT>>& R_ghost = m_state->la->b[GHOST];
  Array1D<RCP<VectorT>>& Z = m_state->la->x[OWNED];

  ParameterList& dbcs = m_params->sublist("dirichlet bcs", true);
  ParameterList& tbcs = m_params->sublist("traction bcs");
  ParameterList& prob = m_params->sublist("problem", true);
  double const dt = prob.get<double>("step size");

  double t = 0.;
  for (int step = 1; step < nsteps; ++step) {
    Array1D<apf::Field*> x = m_nested->primal(step).global;
    t += dt;
    m_state->la->resume_fill_A();
    m_state->la->zero_all();
    eval_forward_jacobian(m_state, m_nested, step);
    apply_primal_tbcs(tbcs, m_nested, R_ghost, t);
    m_state->la->gather_A();
    m_state->la->gather_b();
    apply_primal_dbcs(dbcs, m_nested, dR_dx, R, x, t, step);
    Array1D<apf::Field*> Z_fields = m_nested->adjoint(step).global;
    m_nested->populate_vector(Z_fields, Z);
    double step_error = 0.;
    for (int i = 0; i < Z_fields.size(); ++i) {
      step_error += R[i]->dot(*(Z[i]));
    }

    std::cout << step_error << "\n";

  }
}

void Driver::drive() {
  int cycle = 0;
  print("*** solve-adapt cycle: %d", cycle);
  double const J = solve_primal();
  solve_adjoint();
  write_base_files(m_state, cycle, m_params);
  prepare_fine_space();
  perform_SPR();
  estimate_error();
  write_nested_files(m_nested, cycle, m_params);
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
