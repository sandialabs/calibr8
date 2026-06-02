#include <PCU.h>
#include "control.hpp"
#include "dbcs.hpp"
#include "evaluations.hpp"
#include "line_search.hpp"
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
    int const model_form = m_state->model_form;
    m_state->residuals->local[model_form]->init_variables(m_state, false);
    m_state->d_residuals->local[model_form]->init_variables(m_state);
    m_state->dfad_residuals->local[model_form]->init_variables(m_state);
  }
}

void Primal::solve_at_step(int step) {

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
  double const t = m_state->disc->time(step);

  // print the step information
  if (do_print) print("ON PRIMAL STEP (%d)", step);

  // create or grab the the primal fields at this step
  if (m_disc->type() == COARSE || m_disc->type() == TRUTH) {
    m_disc->create_primal(m_state->residuals, step);
  }
  Array1D<apf::Field*> x = m_disc->primal(step).global;
  if (m_disc->type() == VERIFICATION) {
    int const model_form = m_state->model_form;
    m_disc->initialize_primal_fine(m_state->residuals, step, model_form);
    x = m_disc->primal_fine(step).global;
  }

  // Scratch for the line-search slope R . (A . dx), reused across iterations.
  int const num_resids = R.size();
  Array1D<RCP<VectorT>> Adx;
  resize(Adx, num_resids);
  for (int i = 0; i < num_resids; ++i)
    Adx[i] = rcp(new VectorT(m_disc->map(OWNED, i)));

  // The local constitutive state is re-solved at every residual evaluation, so a
  // line search probing several steps would mutate it and make the merit a
  // non-fixed function of alpha. Snapshot it before each search and restore it
  // before every trial. Create the snapshot fields once here.
  Array1D<apf::Field*>& local_state =
      (m_disc->type() == VERIFICATION)
      ? m_disc->primal_fine(step).local[m_state->model_form]
      : m_disc->primal(step).local[m_state->model_form];
  int const num_local_fields = local_state.size();
  Array1D<apf::Field*> saved_local_state;
  resize(saved_local_state, num_local_fields);
  for (int i = 0; i < num_local_fields; ++i)
    saved_local_state[i] = apf::createField(m_disc->apf_mesh(),
        ("primal_ls_saved_local_" + std::to_string(i)).c_str(),
        apf::getValueType(local_state[i]), apf::getShape(local_state[i]));

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
    int base_status = eval_forward_jacobian(m_state, m_disc, step); // fill in dR_dx
    base_status = PCU_Add_Int(base_status);
    if (base_status != 0) {
      fail("primal step %d, Newton iter %d: local solve failed at the base point "
           "(load increment likely too large)", step, iter);
    }

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
      // Backtracking Armijo line search (line_search.hpp). Merit phi = 1/2||R||^2,
      // base slope phi'(0) = -||R_0||^2, trial slope phi'(alpha) = R(alpha).(A dx).
      double const R_0 = abs_resid_norm;
      double const psi_0 = 0.5 * R_0 * R_0;
      double const dpsi_0 = -2. * psi_0;

      // Snapshot the local state; each trial restores it first so the local
      // solves warm-start identically and phi(alpha) is a fixed function.
      for (int i = 0; i < num_local_fields; ++i)
        apf::copyData(saved_local_state[i], local_state[i]);

      LineSearchParams ls_params = read_line_search_params(global.sublist("line search"));

      double alpha_applied = 1.;   // the full Newton step was applied above
      auto eval = [&](double alpha, double& phi, double& slope) -> bool {
        for (int i = 0; i < num_local_fields; ++i)
          apf::copyData(local_state[i], saved_local_state[i]);
        m_disc->add_to_soln(x, dx, alpha - alpha_applied);
        alpha_applied = alpha;
        m_state->la->resume_fill_A();
        m_state->la->zero_A();
        m_state->la->zero_b();
        int status = eval_forward_jacobian(m_state, m_disc, step);
        status = PCU_Add_Int(status);
        if (status != 0) return false;   // failed assembly: the search contracts
        apply_primal_tbcs(tbcs, m_disc, R_ghost, t);
        m_state->la->gather_A();
        m_state->la->gather_b();
        apply_primal_dbcs(dbcs, m_disc, dR_dx, R, x, t, step);
        m_state->la->complete_fill_A();   // apply_A needs a fill-complete A
        double const R_alpha = m_state->la->norm_b();
        phi = 0.5 * R_alpha * R_alpha;
        m_state->la->apply_A(dx, Adx);    // Adx = A(alpha) . dx
        double slope_sum = 0.;
        for (int i = 0; i < num_resids; ++i) slope_sum += R[i]->dot(*Adx[i]);
        slope = slope_sum;                // phi'(alpha) = R(alpha) . (A dx)
        return true;
      };

      bool assembled = false;
      double const alpha = line_search(ls_params, psi_0, dpsi_0, eval, &assembled);
      if (!assembled) {
        fail("primal step %d, Newton iter %d: line search could not assemble at "
             "any trial step (local solve diverged; load increment likely too large)",
             step, iter);
      }
      // Move the solution to the accepted step (increment only; the next Newton
      // iteration reassembles from the updated solution).
      m_disc->add_to_soln(x, dx, alpha - alpha_applied);
    }

    iter++;

  }

  for (int i = 0; i < num_local_fields; ++i)
    apf::destroyField(saved_local_state[i]);

  // no hope
  if ((iter > max_iters) && (!converged)) {
    fail("Newton's method failed in %d iterations", max_iters);
  }

}

}
