#include <PCU.h>
#include "adjoint_objective.hpp"
#include "evaluations.hpp"
#include "control.hpp"
#include "local_residual.hpp"

namespace calibr8 {

Adjoint_Objective::Adjoint_Objective(RCP<ParameterList> params) :
    Objective(params) {
  m_adjoint.resize(m_num_problems);
  for (int prob = 0; prob < m_num_problems; ++prob) {
    m_adjoint[prob] = rcp(new Adjoint(m_params, m_state[prob],
        m_state[prob]->disc));
  }
}

Adjoint_Objective::~Adjoint_Objective() {}

double Adjoint_Objective::value(ROL::Vector<double> const& p, double&) {

  ROL::Ptr<Array1D<double> const> xp = getVector(p);

  if (param_diff(*xp)) {
    // loop over the problems described in the input
    double J = 0.;
    for (int prob = 0; prob < m_num_problems; ++prob) {
      Array1D<double> const unscaled_params = transform_params(*xp, false);
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

void Adjoint_Objective::gradient(
    ROL::Vector<double>& g,
    ROL::Vector<double> const& p,
    double&) {
  
  //TODO: generalize this to multiple problems

  ROL::Ptr<Array1D<double>> gp = getVector(g);
  ROL::Ptr<Array1D<double> const> xp = getVector(p);
  Array1D<double> const unscaled_params = transform_params(*xp, false);
  Array1D<double> grad(m_num_opt_params, 0.);
  double J = 0.;

  for (int prob = 0; prob < m_num_problems; ++prob) {

    m_state[prob]->residuals->local[m_model_form]->set_params(unscaled_params);
    m_state[prob]->d_residuals->local[m_model_form]->set_params(unscaled_params);

    ParameterList problem_params = m_params->sublist("problem", true);
    int const nsteps = m_state[prob]->disc->num_time_steps();

    Array1D<double> grad_at_step(m_num_opt_params);

    if (param_diff(*xp)) {

      m_state[prob]->disc->destroy_primal();

      for (int step = 1; step <= nsteps; ++step) {
        m_primal[prob]->solve_at_step(step);
        J += eval_qoi(m_state[prob], m_state[prob]->disc, step);
      }
    }

    // create adjoint variables
    m_state[prob]->disc->create_adjoint(m_state[prob]->residuals, nsteps);

    // solve the adjoint problem
    for (int step = nsteps; step > 0; --step) {
      // TODO: put time dependence in adjoint
      //t += dt;
      m_adjoint[prob]->solve_at_step(step);
      grad_at_step = eval_qoi_gradient(m_state[prob], step);
      for (int i = 0; i < m_num_opt_params; ++i) {
        grad[i] += grad_at_step[i];
      }
    }

    m_state[prob]->disc->destroy_adjoint();

  }

  if (param_diff(*xp)) {
    J = PCU_Add_Double(J);
    m_J_old = J;
  }

  PCU_Add_Doubles(grad.data(), m_num_opt_params);

  Array1D<double> const canonical_grad = transform_gradient(grad);

  for (int i = 0; i < m_num_opt_params; ++i) {
    (*gp)[i] = canonical_grad[i];
  }


}

}
