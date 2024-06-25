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
    // loop over the problems described in the input
    double J = 0.;
    Array1D<double> const unscaled_params = transform_params(*xp, false);
    for (int prob = 0; prob < m_num_problems; ++prob) {
      m_state[prob]->residuals->local[m_model_form]->set_params(unscaled_params);
      m_state[prob]->d_residuals->local[m_model_form]->set_params(unscaled_params);
      int const nsteps = m_state[prob]->disc->num_time_steps();
      m_state[prob]->disc->destroy_primal();
      for (int step = 1; step <= nsteps; ++step) {
        m_primal[prob]->solve_at_step(step);
        J += eval_qoi(m_state[prob], m_state[prob]->disc, step);
      }
    }
    PCU_Add_Double(J);
    m_J_old = J;
  }

  return m_J_old;
}

}
