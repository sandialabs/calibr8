#include <PCU.h>
#include "control.hpp"
#include "evaluations.hpp"
#include "femu_objective.hpp"
#include "local_residual.hpp"

namespace calibr8 {

FEMU_Objective::FEMU_Objective(RCP<ParameterList> params) : Objective(params) {}

FEMU_Objective::~FEMU_Objective() {}

double FEMU_Objective::value(ROL::Vector<double> const& p, double&) {

  ROL::Ptr<Array1D<double> const> xp = getVector(p);

  if (param_diff(*xp)) {
    Array1D<double> const unscaled_params = transform_params(*xp, false);
    m_state->residuals->local->set_params(unscaled_params);
    m_state->d_residuals->local->set_params(unscaled_params);

    ParameterList problem_params = m_params->sublist("problem", true);
    int const nsteps = problem_params.get<int>("num steps");
    double const dt = problem_params.get<double>("step size");
    double t = 0.;
    double J = 0.;
    m_state->disc->destroy_primal();
    for (int step = 1; step <= nsteps; ++step) {
      t += dt;
      m_primal->solve_at_step(step, t, dt);
      J += eval_qoi(m_state, m_state->disc, step);
    }

    PCU_Add_Double(J);
    m_J_old = J;

  }

  return m_J_old;
}

}
