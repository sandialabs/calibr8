#include "control.hpp"
#include "evaluations.hpp"
#include "femu_objective.hpp"
#include "local_residual.hpp"
#include "primal.hpp"
#include "state.hpp"

namespace calibr8 {

FEMU_Objective::FEMU_Objective(RCP<ParameterList> params) {
  m_params = params;
  m_state = rcp(new State(*m_params));
  m_primal = rcp(new Primal(m_params, m_state, m_state->disc));
  m_num_opt_params = m_state->residuals->local->params().size();
}

FEMU_Objective::~FEMU_Objective() {
}

Array1D<double> FEMU_Objective::opt_params() const {
  return m_state->residuals->local->params();
}


double FEMU_Objective::value(ROL::Vector<double> const& p, double&) {
  ROL::Ptr<Array1D<double> const> xp = getVector(p);
  m_state->residuals->local->set_params(*xp);
  m_state->d_residuals->local->set_params(*xp);

  ParameterList problem_params = m_params->sublist("problem", true);
  int const nsteps = problem_params.get<int>("num steps");
  double const dt = problem_params.get<double>("step size");
  double t = 0.;
  double J = 0.;
  for (int step = 1; step <= nsteps; ++step) {
    t += dt;
    m_primal->solve_at_step(step, t, dt);
    J += eval_qoi(m_state, m_state->disc, step);
  }
  m_state->disc->destroy_primal();
  return J;
}

ROL::Ptr<Array1D<double> const> FEMU_Objective::getVector(const V& vec) {
  return dynamic_cast<const SV&>(vec).getVector();
}

ROL::Ptr<Array1D<double>> FEMU_Objective::getVector(V& vec) {
  return dynamic_cast<SV&>(vec).getVector();
}

}
