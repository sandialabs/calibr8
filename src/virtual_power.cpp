#include "control.hpp"
#include "dbcs.hpp"
#include "evaluations.hpp"
#include "linear_solve.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "nested.hpp"
#include "state.hpp"
#include "tbcs.hpp"
#include "virtual_power.hpp"

namespace calibr8 {

VirtualPower::VirtualPower(
    RCP<ParameterList> params_in,
    RCP<State> state_in,
    RCP<Disc> disc_in) {
  m_params = params_in;
  m_state = state_in;
  m_disc = disc_in;
  m_disc->create_primal(m_state->residuals, 0);
  ParameterList& vf_list = m_params->sublist("virtual fields", true);
  m_disc->create_virtual(m_state->residuals, vf_list);
  resize(m_vf_vec, 1);
  RCP<const MapT> map = m_disc->map(0, 0);
  m_vf_vec[0] = rcp(new VectorT(map));
  m_disc->populate_vector(m_disc->virtual_fields(0).virtual_field, m_vf_vec);
  m_state->residuals->local->init_variables(m_state);
  m_state->d_residuals->local->init_variables(m_state);
}

double VirtualPower::compute_at_step(int step, double t, double) {

  // gather data needed to solve the problem
  Array1D<RCP<VectorT>>& R = m_state->la->b[OWNED];
  ALWAYS_ASSERT(R.size() == 1);
  Array1D<RCP<VectorT>>& R_ghost = m_state->la->b[GHOST];
  ParameterList& resids = m_params->sublist("residuals", true);
  ParameterList& global = resids.sublist("global residual", true);
  bool const do_print = global.get<bool>("print step", false);
  bool const use_measured = true;

  // print the step information
  if (do_print) print("ON VIRTUAL POWER STEP (%d)", step);

  // fill in the measured field
  m_disc->create_primal(m_state->residuals, step, use_measured);

  // evaluate the residual
  m_state->la->zero_b();                         // zero the residual
  eval_measured_residual(m_state, m_disc, step); // fill in the residual

  // gather the parallel objects to their OWNED state
  m_state->la->gather_b();  // gather the residual R

  double const internal_virtual_power = R[0]->dot(*m_vf_vec[0]);

  // print("Residual norm = %e", m_state->la->norm_b());
  // print("Virtual field norm = %e", m_vf_vec[0]->norm2());
  print("Internal Virtual Power = %e", internal_virtual_power);

  // compute inner product of R with the virutal field
  // some kind of Tpetra nonsense
  double const virtual_power_at_step = 0.;
  return std::pow(virtual_power_at_step, 2);

}

}
