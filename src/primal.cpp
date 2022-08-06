#include "control.hpp"
#include "dbcs.hpp"
#include "evaluations.hpp"
#include "linear_solve.hpp"
#include "local_residual.hpp"
#include "nested.hpp"
#include "primal.hpp"
#include "state.hpp"
#include "tbcs.hpp"

namespace calibr8 {

Primal::Primal(
    RCP<ParameterList> params_in,
    RCP<State> state_in,
    RCP<Disc> disc_in) {
  m_params = params_in;
  m_state = state_in;
  m_disc = disc_in;
  if (m_disc->type() == COARSE) {
    m_disc->create_primal(m_state->residuals, 0);
    m_state->residuals->local->init_variables(m_state);
    m_state->d_residuals->local->init_variables(m_state);
  }
}

void Primal::solve_at_step(int step, double t, double) {

  // gather data needed to solve the problem
  Array2D<RCP<MatrixT>>& dR_dx = m_state->la->A[OWNED];
  Array1D<RCP<VectorT>>& dx = m_state->la->x[OWNED];
  Array1D<RCP<VectorT>>& R = m_state->la->b[OWNED];
  Array1D<RCP<VectorT>>& R_ghost = m_state->la->b[GHOST];
  ParameterList& dbcs = m_params->sublist("dirichlet bcs", true);
  ParameterList& lin_alg = m_params->sublist("linear algebra", true);
  ParameterList& resids = m_params->sublist("residuals", true);
  ParameterList& global = resids.sublist("global residual", true);
  ParameterList& tbcs = m_params->sublist("traction bcs");
  int const max_iters = global.get<int>("nonlinear max iters");
  double const abs_tol = global.get<double>("nonlinear absolute tol");
  double const rel_tol = global.get<double>("nonlinear relative tol");
  bool const do_print = global.get<bool>("print convergence");
  int const max_line_search_evals = global.get<int>("max line search evals", 5);

  // print the step information
  if (do_print) print("ON PRIMAL STEP (%d)", step);

  // create or grab the the primal fields at this step
  if (m_disc->type() == COARSE || m_disc->type() == TRUTH) {
    m_disc->create_primal(m_state->residuals, step);
  }
  Array1D<apf::Field*> x = m_disc->primal(step).global;
  if (m_disc->type() == VERIFICATION) {
    RCP<NestedDisc> nested = Teuchos::rcp_static_cast<NestedDisc>(m_disc);
    x = nested->primal_fine(step).global;
  }

  // Newton's method below
  int iter = 1;
  bool converged = false;
  double resid_norm_0 = 1.;

  while ((iter <= max_iters) && (!converged)) {

    // print the current iteration
    if (do_print) print(" > (%d) Newton iteration", iter);

    // evaluate the Jacobian and residual without BCs
    m_state->la->resume_fill_A();                 // let Tpetra know we're filling dR_dx
    m_state->la->zero_all();                      // zero all linear algebra containers
    eval_forward_jacobian(m_state, m_disc, step); // fill in dR_dx accordingly

    // apply traction boundary conditions
    apply_primal_tbcs(tbcs, m_disc, R_ghost, t);

    // gather the parallel objects to their OWNED state
    m_state->la->gather_A();  // gather the Jacobian dR_dx
    m_state->la->gather_b();  // gather the residual R

    // apply Dirichlet boundary conditions
    apply_primal_dbcs(dbcs, m_disc, dR_dx, R, x, t, step);

    // check if the residual has converged
    double const abs_resid_norm = m_state->la->norm_b();
    if (iter == 1) resid_norm_0 = abs_resid_norm;
    double const rel_resid_norm = abs_resid_norm / resid_norm_0;
    if (do_print) {
      print(" > absolute ||R|| = %e", abs_resid_norm);
      print(" > relative ||R|| = %e", rel_resid_norm);
    }
    if ((abs_resid_norm < abs_tol) || (rel_resid_norm < rel_tol)) {
      converged = true;
      break;
    }

    // prepare the linear system for solving
    m_state->la->complete_fill_A(); // complete filling dR_dx
    m_state->la->scale_b(-1.);      // scale the residual to -R

    // solve the linear system
    calibr8::solve(lin_alg, m_disc, dR_dx, dx, R);

    // add the increment to the current solution fields
    m_disc->add_to_soln(x, dx);

    {
      // backtracking line search parameters
      double const beta = 1.0e-4;
      double const eta = 0.1;

      // check the current residual value
      // this is not optimized in any sense
      m_state->la->resume_fill_A();
      m_state->la->zero_A();
      m_state->la->zero_b();
      eval_forward_jacobian(m_state, m_disc, step);
      apply_primal_tbcs(tbcs, m_disc, R_ghost, t);
      m_state->la->gather_A();
      m_state->la->gather_b();
      apply_primal_dbcs(dbcs, m_disc, dR_dx, R, x, t, step);

      double const R_0 = abs_resid_norm;
      double const psi_0 = 0.5 * R_0 * R_0;
      double const psi_0_deriv = -2. * psi_0;

      int j = 1;
      double alpha_prev = 1.;
      double alpha_j = 1.;
      double R_j = m_state->la->norm_b();
      double psi_j = 0.5 * R_j * R_j;

      while (psi_j >= ((1. - 2. * beta * alpha_j) * psi_0)) {

        alpha_prev = alpha_j;
        alpha_j  = std::max(eta * alpha_j,
            -(std::pow(alpha_j, 2) * psi_0_deriv) /
             (2. * (psi_j - psi_0 - alpha_j * psi_0_deriv)));

        if (do_print) {
          print(" > residual increase -- line search alpha_%d = %.2e",
              j, alpha_j);
        }

        if (j == max_line_search_evals) {
          break;
        }

        ++j;

        m_disc->add_to_soln(x, dx, alpha_j - alpha_prev);

        m_state->la->resume_fill_A();
        m_state->la->zero_A();
        m_state->la->zero_b();
        eval_forward_jacobian(m_state, m_disc, step);
        apply_primal_tbcs(tbcs, m_disc, R_ghost, t);
        m_state->la->gather_A();
        m_state->la->gather_b();
        apply_primal_dbcs(dbcs, m_disc, dR_dx, R, x, t, step);

        R_j = m_state->la->norm_b();
        psi_j = 0.5 * R_j * R_j;

      }
    }

    iter++;

  }

  // no hope
  if ((iter > max_iters) && (!converged)) {
    fail("Newton's method failed in %d iterations", max_iters);
  }

}

}
