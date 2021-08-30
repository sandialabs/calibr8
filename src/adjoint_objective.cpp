#include "adjoint.hpp"
#include "adjoint_objective.hpp"
#include "control.hpp"
#include "evaluations.hpp"
#include "local_residual.hpp"
#include "primal.hpp"
#include "state.hpp"

namespace calibr8 {

Adjoint_Objective::Adjoint_Objective(RCP<ParameterList> params) :
  FEMU_Objective(params) {
  m_adjoint = rcp(new Adjoint(m_params, m_state, m_state->disc));
}

Adjoint_Objective::~Adjoint_Objective() {
}

void Adjoint_Objective::gradient(
    ROL::Vector<double>& g,
    ROL::Vector<double> const& p,
    double&) {
  ROL::Ptr<Array1D<double>> gp = getVector(g);
  ROL::Ptr<Array1D<double> const> xp = getVector(p);
  m_state->residuals->local->set_params(*xp);
  m_state->d_residuals->local->set_params(*xp);

  ParameterList problem_params = m_params->sublist("problem", true);
  int const nsteps = problem_params.get<int>("num steps");
  double const dt = problem_params.get<double>("step size");
  double t = 0.;

  Array1D<double> grad_at_step(m_num_opt_params);
  Array1D<double> grad(m_num_opt_params, 0.);
  size_t const num_opt_params = static_cast<size_t>(m_num_opt_params);

  // solve the forward problem
  for (int step = 1; step <= nsteps; ++step) {
    t += dt;
    m_primal->solve_at_step(step, t, dt);
  }

  // create adjoint variables
  m_state->disc->create_adjoint(m_state->residuals, nsteps);

  // solve the adjoint problem
  for (int step = nsteps; step > 0; --step) {
    // TODO: put time dependence in adjoint
    //t += dt;
    m_adjoint->solve_at_step(step);
    grad_at_step = eval_qoi_gradient(m_state, step);
    for (size_t i = 0; i < num_opt_params; ++i) {
      grad[i] += grad_at_step[i];
    }
  }

  for (size_t i = 0; i < num_opt_params; ++i) {
    (*gp)[i] = grad[i];
  }

  m_state->disc->destroy_primal();
  m_state->disc->destroy_adjoint();

}

}
