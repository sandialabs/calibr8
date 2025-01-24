#include "control.hpp"
#include "dbcs.hpp"
#include "equilibrium_gap.hpp"
#include "evaluations.hpp"
#include "global_residual.hpp"
#include "linear_solve.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "state.hpp"
#include "tbcs.hpp"

namespace calibr8 {

EquilibriumGap::EquilibriumGap(
    RCP<ParameterList> params_in,
    RCP<State> state_in,
    RCP<Disc> disc_in,
    int num_params)
{
  m_params = params_in;
  m_state = state_in;
  m_disc = disc_in;
}

EquilibriumGap::~EquilibriumGap() {}

double EquilibriumGap::compute_at_step(int step)
{

  /* gather data needed to solve the problem */
  Array1D<RCP<VectorT>>& R = m_state->la->b[OWNED];
  ALWAYS_ASSERT(R.size() == 1);
  ParameterList& resids = m_params->sublist("residuals", true);
  ParameterList& global = resids.sublist("global residual", true);
  ParameterList& tbcs = m_params->sublist("traction boundaries", true);
  bool const do_print = global.get<bool>("print step", false);
  bool const use_measured = true;
  if (do_print) print("ON EQUILIBRIUM GAP STEP (%d)", step);

  /* fill in the measured field */
  m_disc->create_primal(m_state->residuals, step, use_measured);

  /* evaluate the residual term */
  /* This will solve for the local state variables */
  m_state->la->zero_b();
  eval_measured_residual(m_state, m_disc, step);
  compute_eq_gap_tractions(tbcs, m_state, m_disc, step);
  m_state->la->gather_b();
  double const eq_gap = 0.5 * R[0]->dot(*(R[0]));

  /* evaulate the load mismatch term */
  double const load_mismatch = eval_qoi(m_state, m_disc, step);

  /* compute the objective function */
  double const gap_scale = 1.;
  double const load_scale = 0.;
  return gap_scale * eq_gap + load_scale * load_mismatch;
}

void EquilibriumGap::compute_at_step_adjoint(
    int step,
    Array1D<double>& grad)
{

  /* gather data */
  int const nsteps = m_state->disc->num_time_steps();

  if (step == nsteps) {
    initialize_adjoint_history_vectors();
  }

  /* evaluate the gradient */
  // eval_equil_gap_adjoint_gradient();

  (void)step;
  (void)grad;
}

void EquilibriumGap::initialize_adjoint_history_vectors()
{
}

}
