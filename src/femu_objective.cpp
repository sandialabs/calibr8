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

  //TODO: generalize to multiple problems
  // m_primal and m_state should be looped over
  if (param_diff(*xp)) {
    Array1D<double> const unscaled_params = transform_params(*xp, false);
    m_state[0]->residuals->local[m_model_form]->set_params(unscaled_params);
    m_state[0]->d_residuals->local[m_model_form]->set_params(unscaled_params);

    ParameterList problem_params = m_params->sublist("problem", true);
    int const nsteps = m_state[0]->disc->num_time_steps();
    double J = 0.;
    m_state[0]->disc->destroy_primal();
    for (int step = 1; step <= nsteps; ++step) {
      m_primal[0]->solve_at_step(step);
      J += eval_qoi(m_state[0], m_state[0]->disc, step);
    }

    PCU_Add_Double(J);
    m_J_old = J;

  }

  return m_J_old;
}

}
