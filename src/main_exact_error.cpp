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
    double solve_primal_fine();
    void solve_adjoint_fine();
    void prepare_fine_space();
    Array1D<apf::Field*> estimate_error();
    void localize_error(Array1D<apf::Field*> const& eta);

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
    Array1D<double> m_eta_bound;
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
  global->set_stabilization_h(BASE);
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


Array1D<apf::Field*> Driver::estimate_error() {
  print("ESTIMATING THE ERROR");
  int const nsteps = m_nested->adjoint().size();

  Array1D<RCP<VectorT>>& R = m_state->la->b[OWNED];
  Array1D<RCP<VectorT>>& R_ghost = m_state->la->b[GHOST];
  auto Z = m_state->la->x;
  auto& Z_owned = Z[OWNED];

  auto global = m_state->residuals->global;
  int const num_global_residuals = global->num_residuals();
  Array1D<apf::Field*> eta_field(num_global_residuals);

  auto gv_shape = m_nested->gv_shape();
  auto m = m_nested->apf_mesh();
  int const num_dims = m->getDimension();

  Array1D<RCP<VectorT>> ELR_ghost(num_global_residuals);
  Array1D<RCP<VectorT>> ELR_owned(num_global_residuals);
  for (int i = 0; i < num_global_residuals; ++i) {
    std::string name = global->resid_name(i);
    name = "eta_" + name;
    int const vtype = m_nested->get_value_type(global->num_eqs(i), num_dims);
    apf::Field* error = apf::createField(m, name.c_str(), vtype, gv_shape);
    apf::zeroField(error);
    eta_field[i] = error;

    auto ghost_map = m_nested->map(GHOST, i);
    ELR_ghost[i] = rcp(new VectorT(ghost_map));
    auto owned_map = m_nested->map(OWNED, i);
    ELR_owned[i] = rcp(new VectorT(owned_map));
  }

  apf::Field* C_error = apf::createStepField(m, "C_error", apf::SCALAR);
  apf::zeroField(C_error);

  // compute Z_fine* and C_error
  for (int step = 1; step < nsteps; ++step) {

    for (int i = 0; i < num_global_residuals; ++i) {
      ELR_ghost[i]->putScalar(0.);
      ELR_owned[i]->putScalar(0.);
     }

    m_state->la->zero_all();
    eval_global_residual(m_state, m_nested, step);
    m_state->la->gather_b();

    eval_linearization_error_terms(m_state, m_nested, step, ELR_ghost, C_error);
    Array1D<apf::Field*> Z_fields = m_nested->adjoint(step).global;
    m_nested->populate_vector(Z_fields, Z);

    for (int i = 0; i < num_global_residuals; ++i) {
      RCP<const ExportT> exporter = m_nested->exporter(i);
      ELR_owned[i]->doExport(*ELR_ghost[i], *exporter, Tpetra::ADD);
      double const ELR_dot_z = ELR_owned[i]->dot(*Z_owned[i]);
      double const R_dot_R = R[i]->dot(*R[i]);
      print("ELR_dot_z = %e", ELR_dot_z);
      print("R_dot_R = %e", R_dot_R);
      R[i]->scale(ELR_dot_z / R_dot_R);
    }

    m_nested->add_to_soln(Z_fields, R, 1.);

  }

  ParameterList& tbcs = m_params->sublist("traction bcs");
  ParameterList& prob = m_params->sublist("problem", true);
  double t = 0.;

  for (int step = 1; step < nsteps; ++step) {
    t = m_state->disc->time(step);

    Array1D<apf::Field*> Z_fields = m_nested->adjoint(step).global;

    // form the coarse space interpolant of z^h
    /*
    Array1D<apf::Field*> Z_interp_fields(num_global_residuals);
    for (int i = 0; i < num_global_residuals; ++i) {
      auto f = Z_fields[i];
      auto interp = m_nested->get_coarse(f);
      Z_interp_fields[i] = interp;
    }
    */

    /*
    m_state->la->zero_all();
    eval_global_residual(m_state, m_nested, step, true, Z_interp_fields);
    eval_tbcs_error_contributions(tbcs, m_nested, Z_interp_fields, R_ghost, t);
    m_state->la->gather_b();
    m_nested->add_to_soln(eta_field, R, -1.);
    */

    m_state->la->zero_all();
    eval_global_residual(m_state, m_nested, step, true, Z_fields);
    eval_tbcs_error_contributions(tbcs, m_nested, Z_fields, R_ghost, t);
    m_state->la->gather_b();
    m_nested->add_to_soln(eta_field, R, 1.);
  }

  double eta_C = 0.;
  double eta_R = 0.;

  double error = 0.;
  double error_bound = 0.;
  // get the nodes associated with the nodes in the mesh
  apf::DynamicArray<apf::Node> nodes;
  auto owned_numbering = m_nested->owned_numbering();
  apf::getNodes(owned_numbering, nodes);

  // loop over all the nodes in the discretization
  for (size_t n = 0; n < nodes.size(); ++n) {

    // get information about the current node
    apf::Node node = nodes[n];
    apf::MeshEntity* ent = node.entity;
    int const ent_node = node.node;

    double eta_node = 0.;
    for (int i = 0; i < num_global_residuals; ++i) {
      apf::Field* f = eta_field[i];
      int const neqs = apf::countComponents(f);
      Array1D<double> sol_comps(9);
      apf::getComponents(f, ent, ent_node, &(sol_comps[0]));
      for (int eq = 0; eq < neqs; ++eq) {
        eta_node += sol_comps[eq];
      }
    }

    error += eta_node;
    eta_R += eta_node;
    error_bound += std::abs(eta_node);
  }

  apf::MeshEntity* elem;
  apf::MeshIterator* it = m->begin(num_dims);
  while ((elem = m->iterate(it))) {
    double const elem_error = apf::getScalar(C_error, elem, 0);
    eta_C += elem_error;
    error += elem_error;
    error_bound += std::abs(elem_error);
  }
  m->end(it);

  error = PCU_Add_Double(error);
  error_bound = PCU_Add_Double(error_bound);

  m_eta.push_back(error);
  m_eta_bound.push_back(error_bound);

  print("eta_R ~ %.15e", eta_R);
  print("eta_C ~ %.15e", eta_C);

  print("total estimate ~ %.15e", error);
  print("error bound ~ %.15e", error_bound);
  return eta_field;
}

void Driver::localize_error(Array1D<apf::Field*> const& eta) {
  print("LOCALIZING THE ERROR");

  apf::Mesh* m = m_nested->apf_mesh();
  apf::Field* R_error = apf::createStepField(m, "R_error", apf::SCALAR);
  apf::zeroField(R_error);
  apf::Field* C_error = m->findField("C_error");

  auto global = m_state->residuals->global;
  int const num_global_residuals = global->num_residuals();
  int const dim = m->getDimension();
  int const q_order = 1;

  for (int i = 0; i < num_global_residuals; ++i) {
    auto f = eta[i];
    int const neqs = apf::countComponents(f);
    Array1D<double> field_comps(neqs);

    apf::MeshEntity* elem;
    apf::MeshIterator* elems = m->begin(dim);
    while ((elem = m->iterate(elems))) {
      apf::MeshElement* me = apf::createMeshElement(m, elem);
      apf::Element* nodal_elem = apf::createElement(f, me);
      int const npts = apf::countIntPoints(me, q_order);
      double error_pt = apf::getScalar(R_error, elem, 0);
      for (int pt = 0; pt < npts; ++pt) {
        apf::Vector3 iota;
        apf::getIntPoint(me, q_order, pt, iota);
        apf::getComponents(nodal_elem, iota, &(field_comps[0]));
        for (int eq = 0; eq < neqs; ++eq) {
          error_pt += field_comps[eq];
        }
      }
      apf::setScalar(R_error, elem, 0, error_pt);
      apf::destroyElement(nodal_elem);
      apf::destroyMeshElement(me);
    }
    m->end(elems);
  }
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
  print("step | nodes_H | J_H | J_h | eta | eta_bound");
  print("--------------------------------");
  for (size_t step = 0; step < m_J_H.size(); ++step) {
    int const nnodes = m_nnodes_H[step];
    double const JH = m_J_H[step];
    double const Jh = m_J_h[step];
    double const eta = m_eta[step];
    double const eta_bound = m_eta_bound[step];
    print("%d | %d | %.15e | %.15e | %.15e | %.15e", step, nnodes, JH, Jh, eta, eta_bound);
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
    auto eta = estimate_error();
    localize_error(eta);
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
