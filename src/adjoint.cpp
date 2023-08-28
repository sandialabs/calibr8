#include "adjoint.hpp"
#include "control.hpp"
#include "dbcs.hpp"
#include "evaluations.hpp"
#include "global_residual.hpp"
#include "linear_solve.hpp"
#include "local_residual.hpp"
#include "state.hpp"

namespace calibr8 {

static int get_num_global_dofs(RCP<State> state) {
  RCP<GlobalResidual<double>> global = state->residuals->global;
  int const num_nodes = state->disc->num_gv_nodes_per_elem();
  int ndofs = 0;
  for (int i = 0; i < global->num_residuals(); ++i) {
    ndofs += global->num_eqs(i) * num_nodes;
  }
  return ndofs;
}

static int get_num_local_dofs(RCP<State> state) {
  int const model_form = state->model_form;
  RCP<LocalResidual<double>> local = state->residuals->local[model_form];
  int ndofs = 0;
  for (int i = 0; i < local->num_residuals(); ++i) {
    ndofs += local->num_eqs(i);
  }
  return ndofs;
}

Adjoint::Adjoint(RCP<ParameterList> params, RCP<State> state, RCP<Disc> disc) {
  m_params = params;
  m_state = state;
  m_disc = disc;
  (void)params;
}

void Adjoint::initialize_history_vectors() {
  int const nsets = m_disc->num_elem_sets();
  int const nglobal_dofs = get_num_global_dofs(m_state);
  int const nlocal_dofs = get_num_local_dofs(m_state);
  int const num_pts = m_disc->num_lv_nodes_per_elem();
  global_history.resize(nsets);
  local_history.resize(nsets);
  for (int set = 0; set < nsets; ++set) {
    std::string const& set_name = m_disc->elem_set_name(set);
    int const nelems = m_disc->elems(set_name).size();
    global_history[set].resize(nelems);
    local_history[set].resize(nelems);
    for (int elem = 0; elem < nelems; ++elem) {
      global_history[set][elem].resize(num_pts);
      local_history[set][elem].resize(num_pts);
      for (int pt = 0; pt < num_pts; ++pt) {
        global_history[set][elem][pt] = EVector::Zero(nglobal_dofs);
        local_history[set][elem][pt] = EVector::Zero(nlocal_dofs);
      }
    }
  }

}

void Adjoint::solve_at_step(int step) {

  // gather data needed to solve the problem
  Array2D<RCP<MatrixT>>& dR_dxT = m_state->la->A[OWNED];
  Array1D<RCP<VectorT>>& dx = m_state->la->x[OWNED];
  Array1D<RCP<VectorT>>& rhs = m_state->la->b[OWNED];
  ParameterList& dbcs = m_params->sublist("dirichlet bcs", true);
  ParameterList& lin_alg = m_params->sublist("linear algebra", true);
  ParameterList& resids = m_params->sublist("residuals", true);
  ParameterList& global = resids.sublist("global residual", true);
  ParameterList& problem_params = m_params->sublist("problem", true);
  int const max_iters = global.get<int>("nonlinear max iters");
  double const abs_tol = global.get<double>("nonlinear absolute tol");
  double const rel_tol = global.get<double>("nonlinear relative tol");
  bool const do_print = global.get<bool>("print convergence");
  int const nsteps = m_state->disc->num_time_steps();

  // set the history vectors to zero
  if (step == nsteps) {
    initialize_history_vectors();
  }

  // print the step information
  if (do_print) print("ON ADJOINT STEP (%d)", step);

  Array1D<apf::Field*> eta = m_disc->adjoint(step).global;
  if (m_disc->type() == NESTED) {
    eta = m_disc->adjoint_fine(step).global;
  }

  // Newton's method below
  int iter = 1;
  bool converged = false;
  double resid_norm_0 = 1.;

  int const num_global_resids = m_state->residuals->global->num_residuals();
  bool is_adjoint = true;

  while ((iter <= max_iters) && (!converged)) {

    // print the current iteration
    if (do_print) print(" > (%d) Newton iteration", iter);

    // evaluate the adjoint Jacobian and rhs without BCs
    // this only needs to be done once because the adjoint problem
    // is linear
    if (iter == 1) {
      m_state->la->resume_fill_A();    // let Tpetra know we're filling dR_dxT
      m_state->la->zero_all();         // zero all linear algebra containers

      eval_adjoint_jacobian(
          m_state,
          m_disc,
          local_history,
          global_history,
          step);

      // gather the parallel objects to their OWNED state
      m_state->la->gather_A();  // gather the adjoint Jacobian dR_dxT
      m_state->la->gather_b();  // gather the rhs

      // apply Dirichlet boundary conditions
      // DTS: should be renamed to apply dbcs?
      apply_primal_dbcs(dbcs, m_disc, dR_dxT, rhs, eta, 0., step, is_adjoint);

      // prepare the linear system for solving
      m_state->la->complete_fill_A(); // complete filling dR_dx
    }

    // solve the linear system
    calibr8::solve(lin_alg, m_disc, dR_dxT, dx, rhs);

    // add the increment to the current solution fields
    m_disc->add_to_soln(eta, dx);

    // check if the residual has converged
    for (int i = 0; i < num_global_resids; ++i) {
      for (int j = 0; j < num_global_resids; ++j) {
        dR_dxT[i][j]->apply(*dx[j], *rhs[i], Teuchos::NO_TRANS, -1., 1.);
      }
    }

    // zero the solution vectors for the next go
    for (int i = 0; i < num_global_resids; ++i) {
      dx[i]->putScalar(0.);
    }

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

    iter++;

    if ((iter > max_iters) && (!converged)) {
      fail("Newton's method failed in %d iterations", max_iters);
    }

  }

  solve_adjoint_local(
      m_state,
      m_disc,
      local_history,
      global_history,
      step);

}

}
