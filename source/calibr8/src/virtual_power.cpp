#include "control.hpp"
#include "dbcs.hpp"
#include "evaluations.hpp"
#include "global_residual.hpp"
#include "linear_solve.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "nested.hpp"
#include "state.hpp"
#include "tbcs.hpp"
#include "virtual_power.hpp"

namespace calibr8 {

static int get_num_global_dofs(RCP<State> state) {
  RCP<GlobalResidual<double>> global = state->residuals->global;
  int const num_nodes = state->disc->num_gv_nodes_per_elem();
  int ndofs = 0;
  for (int i = 0; i < global->num_residuals(); ++i) {
    ndofs += global->num_eqs(i) * num_nodes;
  }
  return ndofs;
}

static int get_num_local_dofs(RCP<State> state) {
  int const model_form = state->model_form;
  RCP<LocalResidual<double>> local = state->residuals->local[model_form];
  int ndofs = 0;
  for (int i = 0; i < local->num_residuals(); ++i) {
    ndofs += local->num_eqs(i);
  }
  return ndofs;
}

void VirtualPower::populate_vf_vector() {
  m_disc->populate_vector(m_disc->virtual_fields(0).virtual_field, m_vf_vec);
}

void VirtualPower::initialize_sens_matrices() {
  int const nsets = m_disc->num_elem_sets();
  int const nlocal_dofs = get_num_local_dofs(m_state);
  int const num_pts = m_disc->num_lv_nodes_per_elem();
  m_local_sens.resize(nsets);
  for (int set = 0; set < nsets; ++set) {
    std::string const& set_name = m_disc->elem_set_name(set);
    int const nelems = m_disc->elems(set_name).size();
    m_local_sens[set].resize(nelems);
    for (int elem = 0; elem < nelems; ++elem) {
      m_local_sens[set][elem].resize(num_pts);
      for (int pt = 0; pt < num_pts; ++pt) {
        m_local_sens[set][elem][pt] = EMatrix::Zero(nlocal_dofs, m_num_params);
      }
    }
  }
}

void VirtualPower::initialize_adjoint_history_vectors() {
  int const nsets = m_disc->num_elem_sets();
  int const nglobal_dofs = get_num_global_dofs(m_state);
  int const nlocal_dofs = get_num_local_dofs(m_state);
  int const num_pts = m_disc->num_lv_nodes_per_elem();
  m_local_history_vectors.resize(nsets);
  for (int set = 0; set < nsets; ++set) {
    std::string const& set_name = m_disc->elem_set_name(set);
    int const nelems = m_disc->elems(set_name).size();
    m_local_history_vectors[set].resize(nelems);
    for (int elem = 0; elem < nelems; ++elem) {
      m_local_history_vectors[set][elem].resize(num_pts);
      for (int pt = 0; pt < num_pts; ++pt) {
        m_local_history_vectors[set][elem][pt] = EVector::Zero(nlocal_dofs);
      }
    }
  }
}

VirtualPower::VirtualPower(
    RCP<ParameterList> params_in,
    RCP<State> state_in,
    RCP<Disc> disc_in,
    int num_params) {

  m_params = params_in;
  m_state = state_in;
  m_disc = disc_in;
  ParameterList& vf_list = m_params->sublist("virtual fields", true);
  m_disc->create_virtual(m_state->residuals, vf_list);
  resize(m_vf_vec[OWNED], 1);
  resize(m_vf_vec[GHOST], 1);
  RCP<const MapT> owned_map = m_disc->map(0, 0);
  RCP<const MapT> ghost_map = m_disc->map(1, 0);
  m_vf_vec[OWNED][0] = rcp(new VectorT(owned_map));
  m_vf_vec[GHOST][0] = rcp(new VectorT(ghost_map));
  populate_vf_vector();
  m_num_params = num_params;

  if (m_num_params > 0) {
    resize(m_mvec[OWNED], 1);
    resize(m_mvec[GHOST], 1);
    RCP<const MapT> owned_map = m_disc->map(0, 0);
    RCP<const MapT> ghost_map = m_disc->map(1, 0);
    m_mvec[OWNED][0] = rcp(new MultiVectorT(owned_map, m_num_params));
    m_mvec[GHOST][0] = rcp(new MultiVectorT(ghost_map, m_num_params));
  }
}

void VirtualPower::compute_residual_at_step(int step) {

  // gather data needed to solve the problem
  Array1D<RCP<VectorT>>& R = m_state->la->b[OWNED];
  ALWAYS_ASSERT(R.size() == 1);
  ParameterList& resids = m_params->sublist("residuals", true);
  ParameterList& global = resids.sublist("global residual", true);
  bool const do_print = global.get<bool>("print step", false);
  bool const use_measured = true;

  // print the step information
  if (do_print) print("ON RESIDUAL CALCULATION STEP (%d)", step);

  // fill in the measured field
  m_disc->create_primal(m_state->residuals, step, use_measured);

  // evaluate the residual
  m_state->la->zero_b();                         // zero the residual
  eval_measured_residual(m_state, m_disc, step); // fill in the residual

  // gather the parallel objects to their OWNED state
  m_state->la->gather_b();  // gather the residual R
}

double VirtualPower::compute_internal_virtual_power() {
  Array1D<RCP<VectorT>>& R = m_state->la->b[OWNED];
  return R[0]->dot(*m_vf_vec[OWNED][0]);
}

double VirtualPower::compute_at_step(int step) {
  compute_residual_at_step(step);
  Array1D<RCP<VectorT>>& R = m_state->la->b[OWNED];
  return R[0]->dot(*m_vf_vec[OWNED][0]);
}

void VirtualPower::compute_at_step_forward_sens(
    int step,
    double& internal_virtual_power,
    Array1D<double>& grad) {

  // gather data needed to solve the problem
  Array1D<RCP<VectorT>>& R = m_state->la->b[OWNED];
  ALWAYS_ASSERT(R.size() == 1);
  ParameterList& resids = m_params->sublist("residuals", true);
  ParameterList& global = resids.sublist("global residual", true);
  bool const do_print = global.get<bool>("print step", false);
  bool const use_measured = true;

  if (step == 1) {
    initialize_sens_matrices();
  }

  // print the step information
  if (do_print) print("ON VIRTUAL POWER STEP (%d)", step);

  // fill in the measured field
  m_disc->create_primal(m_state->residuals, step, use_measured);

  // zero the residual and its dervatives
  m_state->la->zero_b();
  for (int distrib = 0; distrib < NUM_DISTRIB; ++distrib) {
    m_mvec[distrib][0]->putScalar(0.);
  }

  // evaluate the residual and its derivatives
  eval_measured_residual_and_grad(m_state, m_disc, m_mvec[GHOST],
      m_local_sens, step);

  // gather the parallel objects to their OWNED state
  m_state->la->gather_b();
  RCP<MultiVectorT> mvec_owned = m_mvec[OWNED][0];
  RCP<MultiVectorT> mvec_ghost = m_mvec[GHOST][0];
  RCP<const ExportT> exporter = m_disc->exporter(0);
  mvec_owned->doExport(*mvec_ghost, *exporter, Tpetra::ADD);

  internal_virtual_power = R[0]->dot(*m_vf_vec[OWNED][0]);
  for (int p = 0; p < m_num_params; ++p) {
    auto mvec_p = mvec_owned->getVector(p);
    grad[p] = mvec_p->dot(*m_vf_vec[OWNED][0]);
  }
}

void VirtualPower::compute_at_step_adjoint(
    int step,
    double scaled_virtual_power_mismatch,
    Array1D<double>& grad) {

  // gather data needed to solve the problem
  int const nsteps = m_state->disc->num_time_steps();

  if (step == nsteps) {
    initialize_adjoint_history_vectors();
  }

  // evaluate the gradient
  eval_vfm_adjoint_gradient(m_state, m_disc, grad,
      m_local_history_vectors, step, scaled_virtual_power_mismatch);
}

VirtualPower::~VirtualPower() {
  resize(m_vf_vec[OWNED], 0);
  resize(m_vf_vec[GHOST], 0);
  resize(m_mvec[OWNED], 0);
  resize(m_mvec[GHOST], 0);
}

}
