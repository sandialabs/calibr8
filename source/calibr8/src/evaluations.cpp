#include <Eigen/Dense>
#include "disc.hpp"
#include "evaluations.hpp"
#include "global_residual.hpp"
#include "local_residual.hpp"
#include "macros.hpp"
#include "qoi.hpp"
#include "state.hpp"

namespace calibr8 {

void eval_measured_residual_and_grad(
    RCP<State> state,
    RCP<Disc> disc,
    Array1D<RCP<MultiVectorT>>& dR,
    Array3D<EMatrix>& local_sens,
    int step) {

  // gather discretization information
  apf::Mesh* mesh = disc->apf_mesh();

  // gather information from the state object
  int const model_form = state->model_form;
  RCP<LocalResidual<FADT>> local = state->d_residuals->local[model_form];
  RCP<GlobalResidual<FADT>> global = state->d_residuals->global;
  global->set_time_info(state->disc->time(step), state->disc->dt(step));
  Array1D<RCP<VectorT>>& RHS = state->la->b[GHOST];

  // measured displacement field
  Array1D<apf::Field*> x = disc->primal(step).global;
  Array1D<apf::Field*> x_prev = disc->primal(step - 1).global;

  // local state variables
  Array1D<apf::Field*> xi = disc->primal(step).local[model_form];
  Array1D<apf::Field*> xi_prev = disc->primal(step - 1).local[model_form];

  // perform initializations of the residual objects
  global->before_elems(disc);

  // variable telling us the current number of derivatives
  int nderivs = -1;

  // loop over all element sets in the discretization
  for (int es = 0; es < disc->num_elem_sets(); ++es) {

    local->before_elems(es, disc);

    // gather the elements in the current element set
    std::string const& es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);

    // loop over all elements in the element set
    for (size_t elem = 0; elem < elems.size(); ++elem) {

      // get the current mesh element
      apf::MeshEntity* e = elems[elem];
      apf::MeshElement* me = apf::createMeshElement(mesh, e);

      // peform operations on element input
      global->set_elem(me);
      local->set_elem(me);
      global->gather(x, x_prev);

      // loop over domain ip sets
      // ip_set = 0 -> coupled
      Array1D<int> ip_sets = global->ip_sets();
      int const num_ip_sets = ip_sets.size();
      ALWAYS_ASSERT(num_ip_sets == 1);
      int const ip_set = 0;

      // get the quadrature order for the ip set
      int const q_order = ip_sets[ip_set];
      // loop over all integration points in the current element
      int const npts = apf::countIntPoints(me, q_order);

      for (int pt = 0; pt < npts; ++pt) {

        // get integration point specific information
        apf::Vector3 iota;
        apf::getIntPoint(me, q_order, pt, iota);
        double const w = apf::getIntWeight(me, q_order, pt);
        double const dv = apf::getDV(me, iota);

        // solve the local constitutive equations at the integration point
        // and store the resultant local residual and state variables
        global->interpolate(iota);
        local->gather(pt, xi, xi_prev);
        nderivs = local->seed_wrt_xi();
        int path = local->solve_nonlinear(global);
        local->scatter(pt, xi);
        EMatrix const dC_dxi = local->eigen_jacobian(nderivs);

        global->zero_residual();
        global->evaluate(local, iota, w, dv, ip_set);
        EVector const elem_resid = global->eigen_residual();
        EMatrix const dR_dxi = global->eigen_jacobian(nderivs);
        local->unseed_wrt_xi();

        nderivs = local->seed_wrt_xi_prev();
        local->evaluate(global);
        EMatrix const dC_dxi_prev = local->eigen_jacobian(nderivs);
        local->unseed_wrt_xi_prev();

        nderivs = local->seed_wrt_params(es);
        local->evaluate(global);
        EMatrix const dC_dp = local->eigen_jacobian(nderivs);

        global->zero_residual();
        global->evaluate(local, iota, w, dv, ip_set);
        EMatrix const dR_dp = global->eigen_jacobian(nderivs);

        local->unseed_wrt_params(es);

        EMatrix const local_sens_pt_prev = local_sens[es][elem][pt];
        EMatrix const local_sens_rhs = -dC_dp - dC_dxi_prev * local_sens_pt_prev;
        EMatrix const dxi_dp = dC_dxi.fullPivLu().solve(local_sens_rhs);
        local_sens[es][elem][pt] = dxi_dp;

        EMatrix const dR_dp_total = dR_dxi * dxi_dp + dR_dp;

        global->scatter_rhs(disc, elem_resid, RHS);
        global->scatter_sens(disc, dR_dp_total, dR);
      }

      // perform operations on element output
      apf::destroyMeshElement(me);
      global->unset_elem();
      local->unset_elem();

    }

  }

  // perform clean-ups of the residual objects
  local->after_elems();
  global->after_elems();

}

void eval_measured_residual(RCP<State> state, RCP<Disc> disc, int step) {

  // gather discretization information
  apf::Mesh* mesh = disc->apf_mesh();

  // gather information from the state object
  int const model_form = state->model_form;
  RCP<LocalResidual<FADT>> local = state->d_residuals->local[model_form];
  RCP<GlobalResidual<FADT>> global = state->d_residuals->global;
  global->set_time_info(state->disc->time(step), state->disc->dt(step));
  Array1D<RCP<VectorT>>& RHS = state->la->b[GHOST];

  // measured displacement field
  Array1D<apf::Field*> x = disc->primal(step).global;
  Array1D<apf::Field*> x_prev = disc->primal(step - 1).global;

  // local state variables
  Array1D<apf::Field*> xi = disc->primal(step).local[model_form];
  Array1D<apf::Field*> xi_prev = disc->primal(step - 1).local[model_form];

  // perform initializations of the residual objects
  global->before_elems(disc);

  // variable telling us the current number of derivatives
  int nderivs = -1;

  // loop over all element sets in the discretization
  for (int es = 0; es < disc->num_elem_sets(); ++es) {

    local->before_elems(es, disc);

    // gather the elements in the current element set
    std::string const& es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);

    // loop over all elements in the element set
    for (size_t elem = 0; elem < elems.size(); ++elem) {

      // get the current mesh element
      apf::MeshEntity* e = elems[elem];
      apf::MeshElement* me = apf::createMeshElement(mesh, e);

      // peform operations on element input
      global->set_elem(me);
      local->set_elem(me);
      global->gather(x, x_prev);

      // loop over domain ip sets
      // ip_set = 0 -> coupled
      Array1D<int> ip_sets = global->ip_sets();
      int const num_ip_sets = ip_sets.size();
      ALWAYS_ASSERT(num_ip_sets == 1);
      int const ip_set = 0;

      // get the quadrature order for the ip set
      int const q_order = ip_sets[ip_set];
      // loop over all integration points in the current element
      int const npts = apf::countIntPoints(me, q_order);

      for (int pt = 0; pt < npts; ++pt) {

        // get integration point specific information
        apf::Vector3 iota;
        apf::getIntPoint(me, q_order, pt, iota);
        double const w = apf::getIntWeight(me, q_order, pt);
        double const dv = apf::getDV(me, iota);

        // solve the local constitutive equations at the integration point
        // and store the resultant local residual and state variables
        global->interpolate(iota);
        local->gather(pt, xi, xi_prev);
        nderivs = local->seed_wrt_xi();
        int path = local->solve_nonlinear(global);
        local->scatter(pt, xi);
        local->unseed_wrt_xi();

        global->zero_residual();
        global->evaluate(local, iota, w, dv, ip_set);
        EMatrix const elem_resid = global->eigen_residual();
        global->scatter_rhs(disc, elem_resid, RHS);

      }

      // perform operations on element output
      apf::destroyMeshElement(me);
      global->unset_elem();
      local->unset_elem();

    }

  }

  // perform clean-ups of the residual objects
  local->after_elems();
  global->after_elems();
}

int eval_forward_jacobian(RCP<State> state, RCP<Disc> disc, int step) {

  // gather discretization information
  apf::Mesh* mesh = disc->apf_mesh();

  // gather information from the state object
  int const model_form = state->model_form;
  RCP<LocalResidual<FADT>> local = state->d_residuals->local[model_form];
  RCP<GlobalResidual<FADT>> global = state->d_residuals->global;
  global->set_time_info(state->disc->time(step), state->disc->dt(step));
  Array1D<RCP<VectorT>>& RHS = state->la->b[GHOST];
  Array2D<RCP<MatrixT>>& LHS = state->la->A[GHOST];
  Array1D<apf::Field*> x = disc->primal(step).global;
  Array1D<apf::Field*> xi = disc->primal(step).local[model_form];
  Array1D<apf::Field*> x_prev = disc->primal(step - 1).global;
  Array1D<apf::Field*> xi_prev = disc->primal(step - 1).local[model_form];

  // modify the fields if we are doing a verification
  bool const is_verification = (disc->type() == VERIFICATION);
  if (is_verification) {
    x = disc->primal_fine(step).global;
    xi = disc->primal_fine(step).local[model_form];
    x_prev = disc->primal_fine(step - 1).global;
    xi_prev = disc->primal_fine(step - 1).local[model_form];
  }

  // perform initializations of the residual objects
  global->before_elems(disc);

  // variable telling us the current number of derivatives
  int nderivs = -1;

  // loop over all element sets in the discretization
  for (int es = 0; es < disc->num_elem_sets(); ++es) {

    local->before_elems(es, disc);

    // gather the elements in the current element set
    std::string const& es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);

    // loop over all elements in the element set
    for (size_t elem = 0; elem < elems.size(); ++elem) {

      // get the current mesh element
      apf::MeshEntity* e = elems[elem];
      apf::MeshElement* me = apf::createMeshElement(mesh, e);

      // peform operations on element input
      global->set_elem(me);
      local->set_elem(me);
      global->gather(x, x_prev);

      // loop over domain ip sets
      // ip_set = 0 -> coupled
      // ip_set > 0 -> global only
      Array1D<int> ip_sets = global->ip_sets();
      int const num_ip_sets = ip_sets.size();

      for (int ip_set = 0; ip_set < num_ip_sets; ++ip_set) {

        // get the quadrature order for the ip set
        int const q_order = ip_sets[ip_set];
        // loop over all integration points in the current element
        int const npts = apf::countIntPoints(me, q_order);

        for (int pt = 0; pt < npts; ++pt) {

          // get integration point specific information
          apf::Vector3 iota;
          apf::getIntPoint(me, q_order, pt, iota);
          double const w = apf::getIntWeight(me, q_order, pt);
          double const dv = apf::getDV(me, iota);

          if (ip_set == 0) {

            // solve the local constitutive equations at the integration point
            // and store the resultant local residual and its derivatives (dC_dxi)
            global->interpolate(iota);
            local->gather(pt, xi, xi_prev);
            nderivs = local->seed_wrt_xi();
            int path = local->solve_nonlinear(global);
            if (path == -1) {
              return path;
            }
            if (is_verification) {
              disc->branch_paths()[step][es][elem] = path;
            }
            local->scatter(pt, xi);
            EMatrix const dC_dxi = local->eigen_jacobian(nderivs);

            // re-evaluate the constitutive equations to obtain dC_dx
            local->unseed_wrt_xi();
            nderivs = global->seed_wrt_x();
            global->interpolate(iota);
            local->evaluate(global);
            EMatrix const dC_dx = local->eigen_jacobian(nderivs);

            // solve the forward sensitivty system to obtain dxi_dx
            EMatrix const dxi_dx = dC_dxi.fullPivLu().solve(-dC_dx);

            // evaluate and scatter point contributions to the global residual
            local->seed_wrt_x(dxi_dx);

          }

          else {

            nderivs = global->seed_wrt_x();
            global->interpolate(iota);

          }

          global->zero_residual();
          global->evaluate(local, iota, w, dv, ip_set);
          EMatrix const dtotal = global->eigen_jacobian(nderivs);
          EMatrix const elem_resid = global->eigen_residual();
          global->scatter_lhs(disc, dtotal, LHS);
          global->scatter_rhs(disc, elem_resid, RHS);
          global->unseed_wrt_x();

        }

      }

      // perform operations on element output
      apf::destroyMeshElement(me);
      global->unset_elem();
      local->unset_elem();

    }

  }

  // perform clean-ups of the residual objects
  local->after_elems();
  global->after_elems();

  return 0;

}

void eval_global_residual(RCP<State> state, RCP<Disc> disc, int step,
    bool evaluate_error, Array1D<apf::Field*> const& adjoint_fields) {

  // gather discretization information
  apf::Mesh* mesh = disc->apf_mesh();

  // gather information from the state object
  int const model_form = state->model_form;
  RCP<LocalResidual<double>> local = state->residuals->local[model_form];
  RCP<GlobalResidual<double>> global = state->residuals->global;
  global->set_time_info(state->disc->time(step), state->disc->dt(step));
  Array1D<RCP<VectorT>>& RHS = state->la->b[GHOST];
  Array2D<RCP<MatrixT>>& LHS = state->la->A[GHOST];
  Array1D<apf::Field*> x = disc->primal(step).global;
  Array1D<apf::Field*> xi = disc->primal(step).local[model_form];
  Array1D<apf::Field*> x_prev = disc->primal(step - 1).global;
  Array1D<apf::Field*> xi_prev = disc->primal(step - 1).local[model_form];

  // perform initializations of the residual objects
  if (evaluate_error) {
    global->before_elems(disc, ERROR_WEIGHT, adjoint_fields);
  } else {
    global->before_elems(disc);
  }

  // variable telling us the current number of derivatives
  int nderivs = -1;

  // loop over all element sets in the discretization
  for (int es = 0; es < disc->num_elem_sets(); ++es) {

    local->before_elems(es, disc);

    // gather the elements in the current element set
    std::string const& es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);

    // loop over all elements in the element set
    for (size_t elem = 0; elem < elems.size(); ++elem) {

      // get the current mesh element
      apf::MeshEntity* e = elems[elem];
      apf::MeshElement* me = apf::createMeshElement(mesh, e);

      // peform operations on element input
      global->set_elem(me);
      local->set_elem(me);
      global->gather(x, x_prev);

      // loop over domain ip sets
      // ip_set = 0 -> coupled
      // ip_set > 0 -> global only
      Array1D<int> ip_sets = global->ip_sets();
      int const num_ip_sets = ip_sets.size();

      for (int ip_set = 0; ip_set < num_ip_sets; ++ip_set) {

        // get the quadrature order for the ip set
        int q_order = ip_sets[ip_set];
        if (evaluate_error) q_order += 1;
        // loop over all integration points in the current element
        int const npts = apf::countIntPoints(me, q_order);

        for (int pt = 0; pt < npts; ++pt) {

          // get integration point specific information
          apf::Vector3 iota;
          apf::getIntPoint(me, q_order, pt, iota);
          double const w = apf::getIntWeight(me, q_order, pt);
          double const dv = apf::getDV(me, iota);

          if (ip_set == 0) {
            local->gather(0, xi, xi_prev);
          }

          // evaluate the global residual at the integration point
          if (evaluate_error) {
            global->interpolate_with_error(iota);
          } else {
            global->interpolate(iota);
          }
          global->zero_residual();
          global->evaluate(local, iota, w, dv, ip_set);
          EMatrix const elem_resid = global->eigen_residual();
          global->scatter_rhs(disc, elem_resid, RHS);

        }

      }

      // perform operations on element output
      apf::destroyMeshElement(me);
      global->unset_elem();
      local->unset_elem();

    }

  }

  // perform clean-ups of the residual objects
  local->after_elems();
  global->after_elems();

}

template<typename T>
void preprocess_qoi(RCP<QoI<T>> qoi,
    RCP<LocalResidual<T>> local,
    RCP<GlobalResidual<T>> global,
    RCP<State> state,
    RCP<Disc> disc,
    int step) {

  // gather discretization information
  apf::Mesh* mesh = disc->apf_mesh();
  int const q_order = disc->lv_shape()->getOrder();

  // gather information from the state object
  int const model_form = state->model_form;
  Array1D<apf::Field*> x = disc->primal(step).global;
  Array1D<apf::Field*> xi = disc->primal(step).local[model_form];
  Array1D<apf::Field*> x_prev = disc->primal(step - 1).global;
  Array1D<apf::Field*> xi_prev = disc->primal(step - 1).local[model_form];

  if (disc->type() == VERIFICATION) {
    x = disc->primal_fine(step).global;
    xi = disc->primal_fine(step).local[model_form];
    x_prev = disc->primal_fine(step - 1).global;
    xi_prev = disc->primal_fine(step - 1).local[model_form];
  }

  // perform initializations of the residual objects
  global->before_elems(disc);
  qoi->before_elems(disc, step);

  // loop over all element sets in the discretization
  for (int es = 0; es < disc->num_elem_sets(); ++es) {

    local->before_elems(es, disc);

    // gather the elements in the current element set
    std::string const& es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);

    // loop over all elements in the element set
    for (size_t elem = 0; elem < elems.size(); ++elem) {

      // get the current mesh element
      apf::MeshEntity* e = elems[elem];
      apf::MeshElement* me = apf::createMeshElement(mesh, e);

      // peform operations on element input
      global->set_elem(me);
      local->set_elem(me);
      qoi->set_elem(me);
      global->gather(x, x_prev);

      // loop over all integration points in the current element
      int const npts = apf::countIntPoints(me, q_order);
      for (int pt = 0; pt < npts; ++pt) {

        // get integration point specific information
        apf::Vector3 iota;
        apf::getIntPoint(me, q_order, pt, iota);
        double const w = apf::getIntWeight(me, q_order, pt);
        double const dv = apf::getDV(me, iota);

        // preprocess the quantities needed for QoI evaluation
        global->interpolate(iota);
        local->gather(pt, xi, xi_prev);
        qoi->preprocess(es, elem, global, local, iota, w, dv);

      }

      // perform operations on element output
      apf::destroyMeshElement(me);
      global->unset_elem();
      local->unset_elem();
      qoi->unset_elem();

    }

  }

  qoi->preprocess_finalize(step);

  // perform clean-ups of the residual objects
  local->after_elems();
  global->after_elems();
  qoi->after_elems();

}

void eval_adjoint_jacobian(
    RCP<State> state,
    RCP<Disc> disc,
    Array3D<EVector>& g,
    Array3D<EVector>& f,
    int step) {

  // gather discretization information
  apf::Mesh* mesh = disc->apf_mesh();

  // preprocess the QoI
  int const model_form = state->model_form;
  RCP<LocalResidual<FADT>> local = state->d_residuals->local[model_form];
  RCP<GlobalResidual<FADT>> global = state->d_residuals->global;
  global->set_time_info(state->disc->time(step), state->disc->dt(step));
  RCP<QoI<FADT>> qoi = state->d_qoi;
  preprocess_qoi(qoi, local, global, state, disc, step);

  // gather information from the state object
  Array2D<RCP<MatrixT>>& LHS = state->la->A[GHOST];
  Array1D<RCP<VectorT>>& RHS = state->la->b[GHOST];
  Array1D<apf::Field*> x = disc->primal(step).global;
  Array1D<apf::Field*> xi = disc->primal(step).local[model_form];
  Array1D<apf::Field*> x_prev = disc->primal(step - 1).global;
  Array1D<apf::Field*> xi_prev = disc->primal(step - 1).local[model_form];

  // determine if we are doing verification
  bool force_path = false;
  int path = 0;
  if (disc->type() == VERIFICATION) {
    force_path = true;
  }

  // perform initializations of the residual objects
  global->before_elems(disc);
  qoi->before_elems(disc, step);

  // variable telling us the current number of derivatives
  int nderivs = -1;

  // loop over all element sets in the discretization
  for (int es = 0; es < disc->num_elem_sets(); ++es) {

    local->before_elems(es, disc);

    // gather the elements in the current element set
    std::string const& es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);

    // loop over all elements in the element set
    for (size_t elem = 0; elem < elems.size(); ++elem) {

      // get the current mesh element
      apf::MeshEntity* e = elems[elem];
      apf::MeshElement* me = apf::createMeshElement(mesh, e);

      // peform operations on element input
      global->set_elem(me);
      local->set_elem(me);
      qoi->set_elem(me);
      global->gather(x, x_prev);

      // grab the forced path if required
      if (force_path) {
        path = disc->branch_paths()[step][es][elem];
      }

      // loop over domain ip sets
      // ip_set = 0 -> coupled
      // ip_set > 0 -> global only
      Array1D<int> ip_sets = global->ip_sets();
      int const num_ip_sets = ip_sets.size();

      for (int ip_set = 0; ip_set < num_ip_sets; ++ip_set) {

        // get the quadrature order for the ip set
        int const q_order = ip_sets[ip_set];
        // loop over all integration points in the current element
        int const npts = apf::countIntPoints(me, q_order);

        for (int pt = 0; pt < npts; ++pt) {

          // get integration point specific information
          apf::Vector3 iota;
          apf::getIntPoint(me, q_order, pt, iota);
          double const w = apf::getIntWeight(me, q_order, pt);
          double const dv = apf::getDV(me, iota);

          if (ip_set == 0) {

            // solve the local constitutive equations at the integration point
            // and store the resultant local residual and its derivatives (dC_dxi)
            global->interpolate(iota);
            local->gather(pt, xi, xi_prev);
            nderivs = local->seed_wrt_xi();
            local->evaluate(global, force_path, path);
            EMatrix const dC_dxi = local->eigen_jacobian(nderivs);

            // re-evaluate the constitutive equations to obtain dC_dx
            local->unseed_wrt_xi();
            nderivs = global->seed_wrt_x();
            global->interpolate(iota);
            local->evaluate(global, force_path, path);
            EMatrix const dC_dx = local->eigen_jacobian(nderivs);

            // solve the forward sensitivty system to obtain dxi_dx
            EMatrix const dxi_dx = dC_dxi.fullPivLu().solve(-dC_dx);

            // evaluate and scatter point contributions to the global LHS
            local->seed_wrt_x(dxi_dx);

            global->zero_residual();
            global->evaluate(local, iota, w, dv, ip_set);
            EMatrix const dtotal = global->eigen_jacobian(nderivs);
            EMatrix const dtotalT = dtotal.transpose();
            global->scatter_lhs(disc, dtotalT, LHS);
            local->unseed_wrt_xi();

            // evaluate the QoI derivatives to obtain dJ_dx
            qoi->evaluate(es, elem, global, local, iota, w, dv);
            EVector const dJ_dx = qoi->eigen_dvector(nderivs);
            global->unseed_wrt_x();

            // evaluate the QoI derivatives to obtain dJ_dxi
            nderivs = local->seed_wrt_xi();
            global->interpolate(iota);
            qoi->evaluate(es, elem, global, local, iota, w, dv);
            EVector const dJ_dxi = qoi->eigen_dvector(nderivs);
            local->unseed_wrt_xi();

            // update the local history variable
            g[es][elem][pt] -= dJ_dxi;
            EVector const g_pt = g[es][elem][pt];
            EVector const f_pt = f[es][elem][pt];

            // evaluate and scatter point contributions to the global RHS
            EMatrix const dxi_dxT = dxi_dx.transpose();
            EVector const rhs = -dJ_dx + f_pt + dxi_dxT * g_pt;
            global->scatter_rhs(disc, rhs, RHS);

          }

          else {

            nderivs = global->seed_wrt_x();
            global->interpolate(iota);
            global->zero_residual();
            global->evaluate(local, iota, w, dv, ip_set);
            EMatrix const dtotal = global->eigen_jacobian(nderivs);
            EMatrix const dtotalT = dtotal.transpose();
            global->scatter_lhs(disc, dtotalT, LHS);

          }

        }

      }

      // perform operations on element output
      apf::destroyMeshElement(me);
      global->unset_elem();
      local->unset_elem();
      qoi->unset_elem();

    }

  }

  qoi->modify_state(state);

  // perform clean-ups of the residual objects
  local->after_elems();
  global->after_elems();
  qoi->after_elems();

}

void solve_adjoint_local(
    RCP<State> state,
    RCP<Disc> disc,
    Array3D<EVector>& g,
    Array3D<EVector>& f,
    int step) {

  // gather discretization information
  apf::Mesh* mesh = disc->apf_mesh();
  int const q_order = disc->lv_shape()->getOrder();

  // gather information from the state object
  int const model_form = state->model_form;
  RCP<LocalResidual<FADT>> local = state->d_residuals->local[model_form];
  RCP<GlobalResidual<FADT>> global = state->d_residuals->global;
  global->set_time_info(state->disc->time(step), state->disc->dt(step));
  Array1D<apf::Field*> x = disc->primal(step).global;
  Array1D<apf::Field*> xi = disc->primal(step).local[model_form];
  Array1D<apf::Field*> x_prev = disc->primal(step - 1).global;
  Array1D<apf::Field*> xi_prev = disc->primal(step - 1).local[model_form];
  Array1D<apf::Field*> z = disc->adjoint(step).global;
  Array1D<apf::Field*> phi = disc->adjoint(step).local[model_form];

  // modify the fields if we are doing a nested
  bool const is_nested = (disc->type() == NESTED);
  if (is_nested) {
    z = disc->adjoint_fine(step).global;
    phi = disc->adjoint_fine(step).local[model_form];
  }

  // determine if we are doing verification
  bool force_path = false;
  int path = 0;
  if (disc->type() == VERIFICATION) {
    force_path = true;
  }

  // perform initializations of the residual objects
  global->before_elems(disc);

  // variable telling us the current number of derivatives
  int nderivs = -1;

  // loop over all element sets in the discretization
  for (int es = 0; es < disc->num_elem_sets(); ++es) {

    local->before_elems(es, disc);

    // gather the elements in the current element set
    std::string const& es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);

    // loop over all elements in the element set
    for (size_t elem = 0; elem < elems.size(); ++elem) {

      // get the current mesh element
      apf::MeshEntity* e = elems[elem];
      apf::MeshElement* me = apf::createMeshElement(mesh, e);

      // peform operations on element input
      global->set_elem(me);
      local->set_elem(me);
      global->gather(x, x_prev);

      // grab the forced path if required
      if (force_path) {
        path = disc->branch_paths()[step][es][elem];
      }

      // grab the adjoint nodal solution at the element
      EVector const z_nodes = global->gather_adjoint(z);

      // loop over all integration points in the current element
      int const npts = apf::countIntPoints(me, q_order);

      for (int pt = 0; pt < npts; ++pt) {

        // get integration point specific information
        apf::Vector3 iota;
        apf::getIntPoint(me, q_order, pt, iota);
        double const w = apf::getIntWeight(me, q_order, pt);
        double const dv = apf::getDV(me, iota);

        // evaluate local/global residuals and their derivatives, store
        // their transpose Jacobians, and grab g at the integration point
        global->interpolate(iota);
        local->gather(pt, xi, xi_prev);
        nderivs = local->seed_wrt_xi();
        global->zero_residual();
        global->evaluate(local, iota, w, dv, 0);
        local->evaluate(global, force_path, path);
        EMatrix const dC_dxiT = local->eigen_jacobian(nderivs).transpose();
        EMatrix const dR_dxiT = global->eigen_jacobian(nderivs).transpose();
        EVector const g_pt = g[es][elem][pt];

        // Solve for the local adjoint variables and scatter them into fields
        EVector const phi_pt = dC_dxiT.fullPivLu().solve(g_pt - dR_dxiT * z_nodes);
        local->scatter_adjoint(pt, phi_pt, phi);

        // Solve for the global history vector
        local->unseed_wrt_xi();
        nderivs = global->seed_wrt_x_prev();
        global->interpolate(iota);
        local->evaluate(global, force_path, path);
        EMatrix const dC_dx_prevT = local->eigen_jacobian(nderivs).transpose();
        f[es][elem][pt] = -dC_dx_prevT * phi_pt;

        // Solve for the local history vector
        global->unseed_wrt_x_prev();
        global->interpolate(iota);
        nderivs = local->seed_wrt_xi_prev();
        local->evaluate(global, force_path, path);
        EMatrix const dC_dxi_prevT = local->eigen_jacobian(nderivs).transpose();
        g[es][elem][pt] = -dC_dxi_prevT * phi_pt;
        local->unseed_wrt_xi_prev();

      }

      // perform operations on element output
      apf::destroyMeshElement(me);
      global->unset_elem();
      local->unset_elem();

    }

  }

  // perform clean-ups of the residual objects
  local->after_elems();
  global->after_elems();

}


double eval_qoi(RCP<State> state, RCP<Disc> disc, int step) {

  // gather discretization information
  apf::Mesh* mesh = disc->apf_mesh();
  int const q_order = disc->lv_shape()->getOrder();

  // preprocess the QoI
  int const model_form = state->model_form;
  RCP<LocalResidual<double>> local = state->residuals->local[model_form];
  RCP<GlobalResidual<double>> global = state->residuals->global;
  global->set_time_info(state->disc->time(step), state->disc->dt(step));
  RCP<QoI<double>> qoi = state->qoi;
  preprocess_qoi(qoi, local, global, state, disc, step);

  // gather information from the state object
  Array1D<apf::Field*> x = disc->primal(step).global;
  Array1D<apf::Field*> xi = disc->primal(step).local[model_form];
  Array1D<apf::Field*> x_prev = disc->primal(step - 1).global;
  Array1D<apf::Field*> xi_prev = disc->primal(step - 1).local[model_form];

  if (disc->type() == VERIFICATION) {
    x = disc->primal_fine(step).global;
    xi = disc->primal_fine(step).local[model_form];
    x_prev = disc->primal_fine(step - 1).global;
    xi_prev = disc->primal_fine(step - 1).local[model_form];
  }

  // perform initializations of the residual objects
  global->before_elems(disc);
  qoi->before_elems(disc, step);

  // initialize the QoI value at the step
  double J = 0.;

  // loop over all element sets in the discretization
  for (int es = 0; es < disc->num_elem_sets(); ++es) {

    local->before_elems(es, disc);

    // gather the elements in the current element set
    std::string const& es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);

    // loop over all elements in the element set
    for (size_t elem = 0; elem < elems.size(); ++elem) {

      // get the current mesh element
      apf::MeshEntity* e = elems[elem];
      apf::MeshElement* me = apf::createMeshElement(mesh, e);

      // peform operations on element input
      global->set_elem(me);
      local->set_elem(me);
      qoi->set_elem(me);
      global->gather(x, x_prev);

      // loop over all integration points in the current element
      int const npts = apf::countIntPoints(me, q_order);
      for (int pt = 0; pt < npts; ++pt) {

        // get integration point specific information
        apf::Vector3 iota;
        apf::getIntPoint(me, q_order, pt, iota);
        double const w = apf::getIntWeight(me, q_order, pt);
        double const dv = apf::getDV(me, iota);

        // solve the local constitutive equations at the integration point
        // and store the resultant local residual and its derivatives (dC_dxi)
        global->interpolate(iota);
        local->gather(pt, xi, xi_prev);
        qoi->evaluate(es, elem, global, local, iota, w, dv);
        qoi->scatter(J);

      }

      // perform operations on element output
      apf::destroyMeshElement(me);
      global->unset_elem();
      local->unset_elem();
      qoi->unset_elem();

    }

  }

  qoi->postprocess(J);

  // perform clean-ups of the residual objects
  local->after_elems();
  global->after_elems();
  qoi->after_elems();

  return J;

}

Array1D<double> eval_qoi_gradient(RCP<State> state, int step) {

  int const model_form = state->model_form;
  int const num_active_params = state->residuals->local[model_form]->num_active_params();
  int const num_dfad_params = state->residuals->local[model_form]->num_dfad_params();
  int const num_params = num_active_params + num_dfad_params;
  Array1D<double> grad(num_params);
  EVector Egrad = EVector::Zero(num_active_params);
  EVector Egrad_dfad = EVector::Zero(num_dfad_params);

  // gather discretization information
  RCP<Disc> disc = state->disc;
  apf::Mesh* mesh = disc->apf_mesh();

  // preprocess the QoI
  RCP<LocalResidual<FADT>> local = state->d_residuals->local[model_form];
  RCP<LocalResidual<DFADT>> dfad_local = state->dfad_residuals->local[model_form];
  RCP<GlobalResidual<FADT>> global = state->d_residuals->global;
  RCP<GlobalResidual<DFADT>> dfad_global = state->dfad_residuals->global;
  global->set_time_info(state->disc->time(step), state->disc->dt(step));
  dfad_global->set_time_info(state->disc->time(step), state->disc->dt(step));
  RCP<QoI<FADT>> qoi = state->d_qoi;
  preprocess_qoi(qoi, local, global, state, disc, step);

  // gather information from the state object
  Array1D<apf::Field*> x = disc->primal(step).global;
  Array1D<apf::Field*> xi = disc->primal(step).local[model_form];
  Array1D<apf::Field*> x_prev = disc->primal(step - 1).global;
  Array1D<apf::Field*> xi_prev = disc->primal(step - 1).local[model_form];
  Array1D<apf::Field*> z = disc->adjoint(step).global;
  Array1D<apf::Field*> phi = disc->adjoint(step).local[model_form];

  // perform initializations of the residual objects
  global->before_elems(disc);
  qoi->before_elems(disc, step);
  if (num_dfad_params > 0)
    dfad_global->before_elems(disc);

  // variable telling us the current number of derivatives
  int nderivs = -1;

  // loop over all element sets in the discretization
  for (int es = 0; es < disc->num_elem_sets(); ++es) {

    local->before_elems(es, disc);
    if (num_dfad_params > 0)
      dfad_local->before_elems(es, disc);

    // gather the elements in the current element set
    std::string const& es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);

    // loop over all elements in the element set
    for (size_t elem = 0; elem < elems.size(); ++elem) {

      // get the current mesh element
      apf::MeshEntity* e = elems[elem];
      apf::MeshElement* me = apf::createMeshElement(mesh, e);

      // peform operations on element input
      global->set_elem(me);
      local->set_elem(me);
      qoi->set_elem(me);
      global->gather(x, x_prev);

      if (num_dfad_params > 0) {
        dfad_global->set_elem(me);
        dfad_local->set_elem(me);
        dfad_global->gather(x, x_prev);
      }

      // grab the adjoint nodal solution at the element
      EVector const z_nodes = global->gather_adjoint(z);

      // loop over domain ip sets
      // ip_set = 0 -> coupled
      // ip_set > 0 -> global only
      Array1D<int> ip_sets = global->ip_sets();
      int const num_ip_sets = ip_sets.size();

      for (int ip_set = 0; ip_set < num_ip_sets; ++ip_set) {

        // get the quadrature order for the ip set
        int const q_order = ip_sets[ip_set];
        // loop over all integration points in the current element
        int const npts = apf::countIntPoints(me, q_order);

        for (int pt = 0; pt < npts; ++pt) {
          // get integration point specific information
          apf::Vector3 iota;
          apf::getIntPoint(me, q_order, pt, iota);
          double const w = apf::getIntWeight(me, q_order, pt);
          double const dv = apf::getDV(me, iota);

          // evaluate local/global residuals and their derivatives
          // and dot with the corresponding adjoint solutions to
          // compute gradient contributions
          global->interpolate(iota);
          nderivs = local->seed_wrt_params(es);

          if (ip_set == 0) {
            local->gather(pt, xi, xi_prev);
            local->evaluate(global);
            EMatrix const dC_dpT = local->eigen_jacobian(nderivs).transpose();
            EVector const phi_pt = local->gather_adjoint(pt, phi);
            Egrad += dC_dpT * phi_pt;

            // evaluate the QoI derivatives to obtain dJ_dp
            qoi->evaluate(es, elem, global, local, iota, w, dv);
            EVector const dJ_dp = qoi->eigen_dvector(nderivs);
            Egrad += dJ_dp;

            if (num_dfad_params > 0) {
              dfad_global->interpolate(iota);
              dfad_local->gather(pt, xi, xi_prev);
              dfad_local->evaluate(dfad_global);
              EMatrix const dC_dpT_dfad = dfad_local->eigen_jacobian(num_dfad_params).transpose();
              Egrad_dfad += dC_dpT_dfad * phi_pt;
            }

          }

          global->zero_residual();
          global->evaluate(local, iota, w, dv, ip_set);
          EMatrix const dR_dpT = global->eigen_jacobian(nderivs).transpose();
          Egrad += dR_dpT * z_nodes;
          local->unseed_wrt_params(es);

        }

      }

      // perform operations on element output
      apf::destroyMeshElement(me);
      global->unset_elem();
      local->unset_elem();
      qoi->unset_elem();
      if (num_dfad_params > 0) {
        dfad_global->unset_elem();
        dfad_local->unset_elem();
      }

    }

  }

  // perform clean-ups of the residual objects
  local->after_elems();
  global->after_elems();
  qoi->after_elems();
  if (num_dfad_params > 0) {
    dfad_local->after_elems();
    dfad_global->after_elems();
  }

  // concatenate conventional and dfad parameter gradients
  EVector Egrad_total(num_params);
  Egrad_total << Egrad, Egrad_dfad;

  EVector::Map(&grad[0], num_params) = Egrad_total;

  return grad;
}

// TODO: write this using global->interpolate_error
//void eval_error_contributions_nodal();

void eval_error_contributions(
    RCP<State> state,
    RCP<Disc> disc,
    apf::Field* R_error_field,
    apf::Field* C_error_field,
    int step) {

  // gather the residuals from the state object
  int const model_form = state->model_form;
  RCP<LocalResidual<double>> local = state->residuals->local[model_form];
  RCP<GlobalResidual<double>> global = state->residuals->global;
  global->set_time_info(state->disc->time(step), state->disc->dt(step));
  Array1D<RCP<VectorT>>& resid_vec = state->la->b[GHOST];
  Array1D<RCP<VectorT>>& z_vec = state->la->x[GHOST];

  // gather discretization information
  apf::Mesh* mesh = disc->apf_mesh();

  // gather the prolonged forward state variables
  Array1D<apf::Field*> x = disc->primal(step).global;
  Array1D<apf::Field*> xi = disc->primal(step).local[model_form];
  Array1D<apf::Field*> x_prev = disc->primal(step - 1).global;
  Array1D<apf::Field*> xi_prev = disc->primal(step - 1).local[model_form];

  // gather the enriched adjoint state variables
  Array1D<apf::Field*> z = disc->adjoint(step).global;
  Array1D<apf::Field*> phi = disc->adjoint(step).local[model_form];

  // determine if we are doing verification
  bool force_path = false;
  int path = 0;
  if (disc->type() == VERIFICATION) {
    force_path = true;
  }

  // perform initializations of the residual objects
  global->before_elems(disc);

  // loop over all element sets in the discretization
  for (int es = 0; es < disc->num_elem_sets(); ++es) {

    local->before_elems(es, disc);

    // gather the elements in the current element set
    std::string const& es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);

    // loop over all elements in the element set
    for (size_t elem = 0; elem < elems.size(); ++elem) {

      // get the current mesh element
      apf::MeshEntity* e = elems[elem];
      apf::MeshElement* me = apf::createMeshElement(mesh, e);

      // peform operations on element input
      global->set_elem(me);
      local->set_elem(me);
      global->gather(x, x_prev);

      // grab the adjoint nodal solution at the element
      EVector const z_nodes = global->gather_adjoint(z);
      global->assign_rhs(disc, z_nodes, z_vec);

      // grab the forced path if required
      if (force_path) {
        path = disc->branch_paths()[step][es][elem];
      }

      // loop over domain ip sets
      // ip_set = 0 -> coupled
      // ip_set > 0 -> global only
      Array1D<int> ip_sets = global->ip_sets();
      int const num_ip_sets = ip_sets.size();

      for (int ip_set = 0; ip_set < num_ip_sets; ++ip_set) {

        // get the quadrature order for the ip set
        int const q_order = ip_sets[ip_set];
        // loop over all integration points in the current element
        int const npts = apf::countIntPoints(me, q_order);

        for (int pt = 0; pt < npts; ++pt) {

          // get integration point specific information
          apf::Vector3 iota;
          apf::getIntPoint(me, q_order, pt, iota);
          double const w = apf::getIntWeight(me, q_order, pt);
          double const dv = apf::getDV(me, iota);

          if (ip_set == 0) {

            // evaluate the global residual error contributions
            local->gather(pt, xi, xi_prev);
            global->zero_residual();
            global->interpolate(iota);
            global->evaluate(local, iota, w, dv, ip_set);
            EVector const R = global->eigen_residual();
            double const E_R_elem = z_nodes.dot(R);
            double E_R = apf::getScalar(R_error_field, e, 0);
            apf::setScalar(R_error_field, e, 0, E_R + E_R_elem);
            global->scatter_rhs(disc, R, resid_vec);

            // evaluate the local residual error contributions
            local->evaluate(global, force_path, path);
            EVector const C = local->eigen_residual();
            EVector const phi_pt = local->gather_adjoint(pt, phi);
            double const E_C_elem = phi_pt.dot(C);
            double E_C = apf::getScalar(C_error_field, e, 0);
            apf::setScalar(C_error_field, e, 0, E_C + E_C_elem);

          }

          else {

            // evaluate the global residual error contributions
            global->zero_residual();
            global->interpolate(iota);
            global->evaluate(local, iota, w, dv, ip_set);
            EVector const R = global->eigen_residual();
            double const E_R_elem = z_nodes.dot(R);
            double E_R = apf::getScalar(R_error_field, e, 0);
            apf::setScalar(R_error_field, e, 0, E_R + E_R_elem);
            global->scatter_rhs(disc, R, resid_vec);

          }

        }

      }

      // perform operations on element output
      apf::destroyMeshElement(me);
      global->unset_elem();
      local->unset_elem();

    }

  }

  // perform clean-ups of the residual objects
  local->after_elems();
  global->after_elems();

}

void eval_linearization_errors(
    RCP<State> state,
    RCP<Disc> disc,
    int step,
    double& E_lin_R,
    double& E_lin_C) {

  // we must be a verification mesh to do this evaluation
  ALWAYS_ASSERT(disc->type() == VERIFICATION);
  bool force_path = true;

  // gather the residuals from the state object
  int const model_form = state->model_form;
  RCP<LocalResidual<FADT>> local = state->d_residuals->local[model_form];
  RCP<GlobalResidual<FADT>> global = state->d_residuals->global;
  global->set_time_info(state->disc->time(step), state->disc->dt(step));

  // gather discretization information
  apf::Mesh* mesh = disc->apf_mesh();

  // gather the prolonged forward state variables
  Array1D<apf::Field*> x = disc->primal(step).global;
  Array1D<apf::Field*> xi = disc->primal(step).local[model_form];
  Array1D<apf::Field*> x_prev = disc->primal(step - 1).global;
  Array1D<apf::Field*> xi_prev = disc->primal(step - 1).local[model_form];

  // gather the enriched forward state variables
  Array1D<apf::Field*> x_fine = disc->primal_fine(step).global;
  Array1D<apf::Field*> xi_fine = disc->primal_fine(step).local[model_form];
  Array1D<apf::Field*> x_prev_fine = disc->primal_fine(step - 1).global;
  Array1D<apf::Field*> xi_prev_fine = disc->primal_fine(step - 1).local[model_form];

  // gather the enriched adjoint state variables
  Array1D<apf::Field*> z = disc->adjoint(step).global;
  Array1D<apf::Field*> phi = disc->adjoint(step).local[model_form];

  // perform initializations of the residual objects
  global->before_elems(disc);

  // variable telling us the current number of derivatives
  int nderivs = -1;

  // loop over all element sets in the discretization
  for (int es = 0; es < disc->num_elem_sets(); ++es) {

    local->before_elems(es, disc);

    // gather the elements in the current element set
    std::string const& es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);

    // loop over all elements in the element set
    for (size_t elem = 0; elem < elems.size(); ++elem) {

      // get the current mesh element
      apf::MeshEntity* e = elems[elem];
      apf::MeshElement* me = apf::createMeshElement(mesh, e);

      // peform operations on element input
      global->set_elem(me);
      local->set_elem(me);
      global->gather(x, x_prev);

      // grab some nodal solution information at the element
      EVector const z_nodes = global->gather_adjoint(z);
      EVector const x_diff = global->gather_difference(x_fine, x);
      EVector const x_prev_diff =
          global->gather_difference(x_prev_fine, x_prev);

      // initialize the element level global linearization error
      EVector ELR_e = EVector::Zero(x_diff.size());

      // grab the forced path
      int const path = disc->branch_paths()[step][es][elem];

      // loop over domain ip sets
      // ip_set = 0 -> coupled
      // ip_set > 0 -> global only
      Array1D<int> ip_sets = global->ip_sets();
      int const num_ip_sets = ip_sets.size();

      for (int ip_set = 0; ip_set < num_ip_sets; ++ip_set) {

        // get the quadrature order for the ip set
        int const q_order = ip_sets[ip_set];
        // loop over all integration points in the current element
        int const npts = apf::countIntPoints(me, q_order);

        for (int pt = 0; pt < npts; ++pt) {

          // get integration point specific information
          apf::Vector3 iota;
          apf::getIntPoint(me, q_order, pt, iota);
          double const w = apf::getIntWeight(me, q_order, pt);
          double const dv = apf::getDV(me, iota);

          if (ip_set == 0) {

            // grab local state variable data at the point
            EVector const phi_pt = local->gather_adjoint(pt, phi);
            EVector const xi_diff = local->gather_difference(pt, xi_fine, xi);
            EVector const xi_prev_diff =
                local->gather_difference(pt, xi_prev_fine, xi_prev);

            // evaluate derivatives wrt x
            global->zero_residual();
            nderivs = global->seed_wrt_x();
            global->interpolate(iota);
            local->gather(pt, xi, xi_prev);
            global->evaluate(local, iota, w, dv, ip_set);
            local->evaluate(global, force_path, path);
            EVector const R = global->eigen_residual();
            EVector const C = local->eigen_residual();
            EMatrix const dR_dx = global->eigen_jacobian(nderivs);
            EMatrix const dC_dx = local->eigen_jacobian(nderivs);

            // evaluate derivatives wrt xi
            global->unseed_wrt_x();
            global->zero_residual();
            nderivs = local->seed_wrt_xi();
            global->interpolate(iota);
            global->evaluate(local, iota, w, dv, ip_set);
            local->evaluate(global, force_path, path);
            EMatrix const dR_dxi = global->eigen_jacobian(nderivs);
            EMatrix const dC_dxi = local->eigen_jacobian(nderivs);

            // evaluate derivatives wrt x_prev
            local->unseed_wrt_xi();
            nderivs = global->seed_wrt_x_prev();
            global->interpolate(iota);
            local->evaluate(global, force_path, path);
            EMatrix const dC_dx_prev = local->eigen_jacobian(nderivs);

            // evaluate derivatives wrt xi_prev
            global->unseed_wrt_x_prev();
            global->interpolate(iota);
            nderivs = local->seed_wrt_xi_prev();
            local->evaluate(global, force_path, path);
            EMatrix const dC_dxi_prev = local->eigen_jacobian(nderivs);

            // evaluate the point level local linearization error
            EVector const ELC_e =
              -C - (dC_dx * x_diff) - (dC_dxi * xi_diff) -
              (dC_dx_prev * x_prev_diff) - (dC_dxi_prev * xi_prev_diff);
            E_lin_C += phi_pt.dot(ELC_e);

            // evaluate point contribs to the global linearization error
            EVector const ELR_e = -R - (dR_dx * x_diff) - (dR_dxi * xi_diff);
            E_lin_R += z_nodes.dot(ELR_e);

            // unseed on output
            local->unseed_wrt_xi_prev();

          }

          else {

            // evaluate the global residual linearization error contributions
            global->zero_residual();
            nderivs = global->seed_wrt_x();
            global->interpolate(iota);
            global->evaluate(local, iota, w, dv, ip_set);
            EVector const R = global->eigen_residual();
            EMatrix const dR_dx = global->eigen_jacobian(nderivs);

            // evaluate point contribs to the global linearization error
            EVector const ELR_e = -R - (dR_dx * x_diff);
            E_lin_R += z_nodes.dot(ELR_e);

            // unseed on output
            global->unseed_wrt_x();

          }

        }

      }

      // perform operations on element output
      apf::destroyMeshElement(me);
      global->unset_elem();
      local->unset_elem();

    }

  }

  // perform clean-ups of the residual objects
  local->after_elems();
  global->after_elems();

}

void eval_linearization_error_terms(
    RCP<State> state,
    RCP<Disc> disc,
    int step,
    Array1D<RCP<VectorT>>& ELR,
    apf::Field* C_error) {

  // we must be a verification mesh to do this evaluation
  ALWAYS_ASSERT(disc->type() == VERIFICATION);
  bool force_path = true;

  // gather the residuals from the state object
  int const model_form = state->model_form;
  RCP<LocalResidual<FADT>> local = state->d_residuals->local[model_form];
  RCP<GlobalResidual<FADT>> global = state->d_residuals->global;
  global->set_time_info(state->disc->time(step), state->disc->dt(step));

  // gather discretization information
  apf::Mesh* mesh = disc->apf_mesh();

  // gather the prolonged forward state variables
  Array1D<apf::Field*> x = disc->primal(step).global;
  Array1D<apf::Field*> xi = disc->primal(step).local[model_form];
  Array1D<apf::Field*> x_prev = disc->primal(step - 1).global;
  Array1D<apf::Field*> xi_prev = disc->primal(step - 1).local[model_form];

  // gather the enriched forward state variables
  Array1D<apf::Field*> x_fine = disc->primal_fine(step).global;
  Array1D<apf::Field*> xi_fine = disc->primal_fine(step).local[model_form];
  Array1D<apf::Field*> x_prev_fine = disc->primal_fine(step - 1).global;
  Array1D<apf::Field*> xi_prev_fine = disc->primal_fine(step - 1).local[model_form];

  // gather the enriched local adjoint state variables
  Array1D<apf::Field*> phi = disc->adjoint(step).local[model_form];

  // perform initializations of the residual objects
  global->before_elems(disc);

  // variable telling us the current number of derivatives
  int nderivs = -1;

  // loop over all element sets in the discretization
  for (int es = 0; es < disc->num_elem_sets(); ++es) {

    local->before_elems(es, disc);

    // gather the elements in the current element set
    std::string const& es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);

    // loop over all elements in the element set
    for (size_t elem = 0; elem < elems.size(); ++elem) {

      // get the current mesh element
      apf::MeshEntity* e = elems[elem];
      apf::MeshElement* me = apf::createMeshElement(mesh, e);

      // peform operations on element input
      global->set_elem(me);
      local->set_elem(me);
      global->gather(x, x_prev);

      // grab some nodal solution information at the element
      EVector const x_diff = global->gather_difference(x_fine, x);
      EVector const x_prev_diff =
          global->gather_difference(x_prev_fine, x_prev);

      // initialize the element level global linearization error
      EVector ELR_e = EVector::Zero(x_diff.size());

      // grab the forced path
      int const path = disc->branch_paths()[step][es][elem];

      // loop over domain ip sets
      // ip_set = 0 -> coupled
      // ip_set > 0 -> global only
      Array1D<int> ip_sets = global->ip_sets();
      int const num_ip_sets = ip_sets.size();

      for (int ip_set = 0; ip_set < num_ip_sets; ++ip_set) {

        // get the quadrature order for the ip set
        int const q_order = ip_sets[ip_set];
        // loop over all integration points in the current element
        int const npts = apf::countIntPoints(me, q_order);

        for (int pt = 0; pt < npts; ++pt) {

          // get integration point specific information
          apf::Vector3 iota;
          apf::getIntPoint(me, q_order, pt, iota);
          double const w = apf::getIntWeight(me, q_order, pt);
          double const dv = apf::getDV(me, iota);

          if (ip_set == 0) {

            // grab local state variable data at the point
            EVector const phi_pt = local->gather_adjoint(pt, phi);
            EVector const xi_diff = local->gather_difference(pt, xi_fine, xi);
            EVector const xi_prev_diff =
                local->gather_difference(pt, xi_prev_fine, xi_prev);

            // evaluate derivatives wrt x
            global->zero_residual();
            nderivs = global->seed_wrt_x();
            global->interpolate(iota);
            local->gather(pt, xi, xi_prev);
            global->evaluate(local, iota, w, dv, ip_set);
            local->evaluate(global, force_path, path);
            EVector const R = global->eigen_residual();
            EVector const C = local->eigen_residual();
            EMatrix const dR_dx = global->eigen_jacobian(nderivs);
            EMatrix const dC_dx = local->eigen_jacobian(nderivs);

            // evaluate derivatives wrt xi
            global->unseed_wrt_x();
            global->zero_residual();
            nderivs = local->seed_wrt_xi();
            global->interpolate(iota);
            global->evaluate(local, iota, w, dv, ip_set);
            local->evaluate(global, force_path, path);
            EMatrix const dR_dxi = global->eigen_jacobian(nderivs);
            EMatrix const dC_dxi = local->eigen_jacobian(nderivs);

            // evaluate derivatives wrt x_prev
            local->unseed_wrt_xi();
            nderivs = global->seed_wrt_x_prev();
            global->interpolate(iota);
            local->evaluate(global, force_path, path);
            EMatrix const dC_dx_prev = local->eigen_jacobian(nderivs);

            // evaluate derivatives wrt xi_prev
            global->unseed_wrt_x_prev();
            global->interpolate(iota);
            nderivs = local->seed_wrt_xi_prev();
            local->evaluate(global, force_path, path);
            EMatrix const dC_dxi_prev = local->eigen_jacobian(nderivs);

            // evaluate the point level total level local residual error
            // standard + linearization error
            EVector const ELC_e = -(dC_dx * x_diff) - (dC_dxi * xi_diff) -
              (dC_dx_prev * x_prev_diff) - (dC_dxi_prev * xi_prev_diff);
            //EVector const ELC_e = C;
            double const E_C_elem = phi_pt.dot(ELC_e);
            double E_C = apf::getScalar(C_error, e, 0);
            apf::setScalar(C_error, e, 0, E_C + E_C_elem);

            // evaluate point contribs to the global linearization error
            EVector const ELR_e = -R - (dR_dx * x_diff) - (dR_dxi * xi_diff);
            global->scatter_rhs(disc, ELR_e, ELR);

            // unseed on output
            local->unseed_wrt_xi_prev();

          }

          else {

            // evaluate the global residual linearization error contributions
            global->zero_residual();
            nderivs = global->seed_wrt_x();
            global->interpolate(iota);
            global->evaluate(local, iota, w, dv, ip_set);
            EVector const R = global->eigen_residual();
            EMatrix const dR_dx = global->eigen_jacobian(nderivs);

            // evaluate point contribs to the global linearization error
            EVector const ELR_e = -R - (dR_dx * x_diff);
            global->scatter_rhs(disc, ELR_e, ELR);

            // unseed on output
            global->unseed_wrt_x();

          }

        }

      }

      // perform operations on element output
      apf::destroyMeshElement(me);
      global->unset_elem();
      local->unset_elem();

    }

  }

  // perform clean-ups of the residual objects
  local->after_elems();
  global->after_elems();

}

void eval_exact_errors(
    RCP<State> state,
    RCP<Disc> disc,
    apf::Field* R_error_field,
    apf::Field* C_error_field,
    int step) {

  // we must be a verification mesh to do this evaluation
  ALWAYS_ASSERT(disc->type() == VERIFICATION);
  bool force_path = true;

  // gather the residuals from the state object
  int const model_form = state->model_form;
  RCP<LocalResidual<FADT>> local = state->d_residuals->local[model_form];
  RCP<GlobalResidual<FADT>> global = state->d_residuals->global;
  global->set_time_info(state->disc->time(step), state->disc->dt(step));

  // gather discretization information
  apf::Mesh* mesh = disc->apf_mesh();

  // gather the prolonged forward state variables
  Array1D<apf::Field*> x = disc->primal(step).global;
  Array1D<apf::Field*> xi = disc->primal(step).local[model_form];
  Array1D<apf::Field*> x_prev = disc->primal(step - 1).global;
  Array1D<apf::Field*> xi_prev = disc->primal(step - 1).local[model_form];

  // gather the enriched forward state variables
  Array1D<apf::Field*> x_fine = disc->primal_fine(step).global;
  Array1D<apf::Field*> xi_fine = disc->primal_fine(step).local[model_form];
  Array1D<apf::Field*> x_prev_fine = disc->primal_fine(step - 1).global;
  Array1D<apf::Field*> xi_prev_fine = disc->primal_fine(step - 1).local[model_form];

  // gather the enriched adjoint state variables
  Array1D<apf::Field*> z = disc->adjoint(step).global;
  Array1D<apf::Field*> phi = disc->adjoint(step).local[model_form];

  // perform initializations of the residual objects
  global->before_elems(disc);

  // variable telling us the current number of derivatives
  int nderivs = -1;

  // loop over all element sets in the discretization
  for (int es = 0; es < disc->num_elem_sets(); ++es) {

    local->before_elems(es, disc);

    // gather the elements in the current element set
    std::string const& es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);

    // loop over all elements in the element set
    for (size_t elem = 0; elem < elems.size(); ++elem) {

      // get the current mesh element
      apf::MeshEntity* e = elems[elem];
      apf::MeshElement* me = apf::createMeshElement(mesh, e);

      // peform operations on element input
      global->set_elem(me);
      local->set_elem(me);
      global->gather(x, x_prev);

      // grab some nodal solution information at the element
      EVector const z_nodes = global->gather_adjoint(z);
      EVector const x_diff = global->gather_difference(x_fine, x);
      EVector const x_prev_diff =
          global->gather_difference(x_prev_fine, x_prev);

      // grab the forced path
      int const path = disc->branch_paths()[step][es][elem];

      // loop over domain ip sets
      // ip_set = 0 -> coupled
      // ip_set > 0 -> global only
      Array1D<int> ip_sets = global->ip_sets();
      int const num_ip_sets = ip_sets.size();

      // the values that contribute to the error
      double E_R = 0.;
      double E_C = 0.;

      for (int ip_set = 0; ip_set < num_ip_sets; ++ip_set) {

        // get the quadrature order for the ip set
        int const q_order = ip_sets[ip_set];
        // loop over all integration points in the current element
        int const npts = apf::countIntPoints(me, q_order);

        for (int pt = 0; pt < npts; ++pt) {

          // get integration point specific information
          apf::Vector3 iota;
          apf::getIntPoint(me, q_order, pt, iota);
          double const w = apf::getIntWeight(me, q_order, pt);
          double const dv = apf::getDV(me, iota);

          if (ip_set == 0) {

            // grab local state variable data at the point
            EVector const phi_pt = local->gather_adjoint(pt, phi);
            EVector const xi_diff = local->gather_difference(pt, xi_fine, xi);
            EVector const xi_prev_diff =
                local->gather_difference(pt, xi_prev_fine, xi_prev);

            // evaluate derivatives wrt x
            global->zero_residual();
            nderivs = global->seed_wrt_x();
            global->interpolate(iota);
            local->gather(pt, xi, xi_prev);
            global->evaluate(local, iota, w, dv, ip_set);
            local->evaluate(global, force_path, path);
            EMatrix const dR_dx = global->eigen_jacobian(nderivs);
            EMatrix const dC_dx = local->eigen_jacobian(nderivs);

            // evaluate derivatives wrt xi
            global->unseed_wrt_x();
            global->zero_residual();
            nderivs = local->seed_wrt_xi();
            global->interpolate(iota);
            global->evaluate(local, iota, w, dv, ip_set);
            local->evaluate(global, force_path, path);
            EMatrix const dR_dxi = global->eigen_jacobian(nderivs);
            EMatrix const dC_dxi = local->eigen_jacobian(nderivs);

            // evaluate derivatives wrt x_prev
            local->unseed_wrt_xi();
            nderivs = global->seed_wrt_x_prev();
            global->interpolate(iota);
            local->evaluate(global, force_path, path);
            EMatrix const dC_dx_prev = local->eigen_jacobian(nderivs);

            // evaluate derivatives wrt xi_prev
            global->unseed_wrt_x_prev();
            global->interpolate(iota);
            nderivs = local->seed_wrt_xi_prev();
            local->evaluate(global, force_path, path);
            EMatrix const dC_dxi_prev = local->eigen_jacobian(nderivs);

            // evaluate the point level local linearization error
            EVector const C_contrib =
              - (dC_dx * x_diff) - (dC_dxi * xi_diff) -
              (dC_dx_prev * x_prev_diff) - (dC_dxi_prev * xi_prev_diff);
            E_C += phi_pt.dot(C_contrib);

            // evaluate point contribs to the global linearization error
            EVector const R_contrib = - (dR_dx * x_diff) - (dR_dxi * xi_diff);
            E_R += z_nodes.dot(R_contrib);

            // unseed on output
            local->unseed_wrt_xi_prev();

          }

          else {

            // evaluate the global residual linearization error contributions
            global->zero_residual();
            nderivs = global->seed_wrt_x();
            global->interpolate(iota);
            global->evaluate(local, iota, w, dv, ip_set);
            EMatrix const dR_dx = global->eigen_jacobian(nderivs);

            // evaluate point contribs to the global linearization error
            EVector const R_contrib = - (dR_dx * x_diff);
            E_R += z_nodes.dot(R_contrib);

            // unseed on output
            global->unseed_wrt_x();

          }

        }

      }

      apf::setScalar(R_error_field, e, 0, E_R + apf::getScalar(R_error_field, e, 0));
      apf::setScalar(C_error_field, e, 0, E_C + apf::getScalar(C_error_field, e, 0));

      // perform operations on element output
      apf::destroyMeshElement(me);
      global->unset_elem();
      local->unset_elem();

    }

  }

  // perform clean-ups of the residual objects
  local->after_elems();
  global->after_elems();

}

// TODO: Cauchy stress can have contributions from fields
// with different polynomials orders (e.g. constant + linear).
// Generalize to handle such cases.
apf::Field* eval_cauchy(RCP<State> state, int step) {

  // an assumption about the pressure index
  int const pressure_idx = 1;

  // gather discretization information
  RCP<Disc> disc = state->disc;
  apf::Mesh* mesh = disc->apf_mesh();
  int const ndims = mesh->getDimension();
  int const q_order = disc->lv_shape()->getOrder();

  // gather information from the state object
  int const model_form = state->model_form;
  RCP<LocalResidual<double>> local = state->residuals->local[model_form];
  RCP<GlobalResidual<double>> global = state->residuals->global;
  global->set_time_info(state->disc->time(step), state->disc->dt(step));
  Array1D<apf::Field*> x = disc->primal(step).global;
  Array1D<apf::Field*> xi = disc->primal(step).local[model_form];
  Array1D<apf::Field*> x_prev = disc->primal(step - 1).global;
  Array1D<apf::Field*> xi_prev = disc->primal(step - 1).local[model_form];

  // create the field to fill in
  apf::FieldShape* shape = state->disc->lv_shape();
  apf::Field* field = apf::createField(mesh, "sigma", apf::MATRIX, shape);
  apf::zeroField(field);
  if (step == 0) return field;

  // perform initializations of the residual objects
  global->before_elems(disc);

  // loop over all element sets in the discretization
  for (int es = 0; es < disc->num_elem_sets(); ++es) {

    local->before_elems(es, disc);

    // gather the elements in the current element set
    std::string const& es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);

    // loop over all elements in the element set
    for (size_t elem = 0; elem < elems.size(); ++elem) {

      // get the current mesh element
      apf::MeshEntity* e = elems[elem];
      apf::MeshElement* me = apf::createMeshElement(mesh, e);

      // peform operations on element input
      global->set_elem(me);
      local->set_elem(me);
      global->gather(x, x_prev);

      // loop over all integration points in the current element
      int const npts = apf::countIntPoints(me, q_order);
      for (int pt = 0; pt < npts; ++pt) {

        // get integration point specific information
        apf::Vector3 iota;
        apf::getIntPoint(me, q_order, pt, iota);

        // evaluate the cauchy stress tensor at the point
        global->interpolate(iota);
        local->gather(pt, xi, xi_prev);
        Tensor<double> const sigma = local->cauchy(global);

        // set the cauchy stress tensor to a field
        apf::Matrix3x3 apf_sigma(0, 0, 0, 0, 0, 0, 0, 0, 0);
        for (int i = 0; i < ndims; ++i) {
          for (int j = 0; j < ndims; ++j) {
            apf_sigma[i][j] = sigma(i, j);
          }
        }
        apf::setMatrix(field, e, pt, apf_sigma);

      }

      // perform operations on element output
      apf::destroyMeshElement(me);
      global->unset_elem();
      local->unset_elem();

    }
  }

  // perform clean-ups of the residual objects
  local->after_elems();
  global->after_elems();

  return field;

}

void eval_adjoint_measured_residual_and_grad(
    RCP<ParameterList> params,
    RCP<State> state,
    RCP<Disc> disc,
    Array1D<RCP<MultiVectorT>>& dR,
    Array3D<EMatrix>& local_hist,
    int step, double vp_mistach_scaled) {

  // gather discretization information
  apf::Mesh* mesh = disc->apf_mesh();

  // gather information from the state object
  int const model_form = state->model_form;
  RCP<LocalResidual<FADT>> local = state->d_residuals->local[model_form];
  RCP<GlobalResidual<FADT>> global = state->d_residuals->global;
  global->set_time_info(state->disc->time(step), state->disc->dt(step));
  Array1D<RCP<VectorT>>& RHS = state->la->b[GHOST];

  // measured displacement field
  Array1D<apf::Field*> x = disc->primal(step).global;
  Array1D<apf::Field*> x_prev = disc->primal(step-1).global;

  // local state variables
  Array1D<apf::Field*> xi = disc->primal(step).local[model_form];
  Array1D<apf::Field*> xi_prev = disc->primal(step-1).local[model_form];

  // perform initializations of the residual objects
  global->before_elems(disc);

  // variable telling us the current number of derivatives
  int nderivs = -1;

  // loop over all element sets in the discretization
  for (int es = 0; es < disc->num_elem_sets(); ++es) {

    local->before_elems(es, disc);

    // gather the elements in the current element set
    std::string const& es_name = disc->elem_set_name(es);
    ElemSet const& elems = disc->elems(es_name);

    // loop over all elements in the element set
    for (size_t elem = 0; elem < elems.size(); ++elem) {

      // get the current mesh element
      apf::MeshEntity* e = elems[elem];
      apf::MeshElement* me = apf::createMeshElement(mesh, e);

      // peform operations on element input
      global->set_elem(me);
      local->set_elem(me);
      global->gather(x, x_prev);

      // loop over domain ip sets
      // ip_set = 0 -> coupled
      Array1D<int> ip_sets = global->ip_sets();
      int const num_ip_sets = ip_sets.size();
      ALWAYS_ASSERT(num_ip_sets == 1);
      int const ip_set = 0;

      // get the quadrature order for the ip set
      int const q_order = ip_sets[ip_set];
      // loop over all integration points in the current element
      int const npts = apf::countIntPoints(me, q_order);

      for (int pt = 0; pt < npts; ++pt) {

        // get integration point specific information
        apf::Vector3 iota;
        apf::getIntPoint(me, q_order, pt, iota);
        double const w = apf::getIntWeight(me, q_order, pt);
        double const dv = apf::getDV(me, iota);

        // solve the local constitutive equations at the integration point
        // and store the resultant local residual and state variables
        global->interpolate(iota);
        local->gather(pt, xi, xi_prev);
        nderivs = local->seed_wrt_xi();
        int path = local->solve_nonlinear(global);
        local->scatter(pt, xi);
        EMatrix const dC_dxiT = local->eigen_jacobian(nderivs).transpose();

        global->zero_residual();
        global->evaluate(local, iota, w, dv, ip_set);
        EVector const elem_resid = global->eigen_residual();
        EMatrix const dR_dxiT = global->eigen_jacobian(nderivs).transpose();;
        local->unseed_wrt_xi();

        nderivs = local->seed_wrt_xi_prev();
        local->evaluate(global);
        EMatrix const dC_dxi_prevT = local->eigen_jacobian(nderivs).transpose();;
        local->unseed_wrt_xi_prev();

        nderivs = local->seed_wrt_params(es);
        local->evaluate(global);
        EMatrix const dC_dp = local->eigen_jacobian(nderivs);

        global->zero_residual();
        global->evaluate(local, iota, w, dv, ip_set);
        EMatrix const dR_dp = global->eigen_jacobian(nderivs);
        local->unseed_wrt_params(es);

        EMatrix const local_sens_pt_prev = local_hist[es][elem][pt];

        EMatrix const local_sens_rhs = -dR_dxiT * vp_mistach_scaled - local_sens_pt_prev;
        EMatrix const phi = dC_dxiT.fullPivLu().solve(local_sens_rhs);
        local_hist[es][elem][pt] = dC_dxi_prevT * phi;

        EMatrix const dL_dp = dR_dp * vp_mistach_scaled + phi.transpose() * dC_dp;

        global->scatter_rhs(disc, elem_resid, RHS);
        global->scatter_sens(disc, dL_dp, dR);
      }

      // perform operations on element output
      apf::destroyMeshElement(me);
      global->unset_elem();
      local->unset_elem();

    }

  }

  // perform clean-ups of the residual objects
  local->after_elems();
  global->after_elems();

}

}
