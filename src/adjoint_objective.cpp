#include "adjoint.hpp"
#include "adjoint_objective.hpp"
#include "control.hpp"
#include "evaluations.hpp"
#include "local_residual.hpp"
#include "primal.hpp"
#include "state.hpp"

namespace calibr8 {

Adjoint_Objective::Adjoint_Objective(RCP<ParameterList> params) :
  Objective(params) {
  m_adjoint = rcp(new Adjoint(m_params, m_state, m_state->disc));
}

Adjoint_Objective::~Adjoint_Objective() {}

double Adjoint_Objective::value(ROL::Vector<double> const& p, double&) {
  ROL::Ptr<Array1D<double> const> xp = getVector(p);
  Array1D<double> const unscaled_params = transform_params(*xp, false);
  m_state->residuals->local->set_params(unscaled_params);
  m_state->d_residuals->local->set_params(unscaled_params);

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
  for (int i = 0; i < m_num_opt_params; ++i) {
    print("Unscaled x[%i] = %.16e", i, unscaled_params[i]);
  }
  print("J = %.16e", J);

  return J;
}

void Adjoint_Objective::gradient(
    ROL::Vector<double>& g,
    ROL::Vector<double> const& p,
    double&) {

  ROL::Ptr<Array1D<double>> gp = getVector(g);

  ROL::Ptr<Array1D<double> const> xp = getVector(p);
  Array1D<double> const unscaled_params = transform_params(*xp, false);
  m_state->residuals->local->set_params(unscaled_params);
  m_state->d_residuals->local->set_params(unscaled_params);

  ParameterList problem_params = m_params->sublist("problem", true);
  int const nsteps = problem_params.get<int>("num steps");
  double const dt = problem_params.get<double>("step size");
  double t = 0.;

  Array1D<double> grad_at_step(m_num_opt_params);
  Array1D<double> grad(m_num_opt_params, 0.);

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
    for (int i = 0; i < m_num_opt_params; ++i) {
      grad[i] += grad_at_step[i];
    }
  }

  Array1D<double> const canonical_grad = transform_gradient(grad);

  for (int i = 0; i < m_num_opt_params; ++i) {
    print("x[%i] = %.16e -> g[%i] = %.16e", i, unscaled_params[i],
        i, canonical_grad[i]);
  }

  for (int i = 0; i < m_num_opt_params; ++i) {
    (*gp)[i] = canonical_grad[i];
  }

  m_state->disc->destroy_primal();
  m_state->disc->destroy_adjoint();

}

}
