#pragma once

//! \file evaluations.hpp
//! \brief Evaluations based on residuals

#include <apf.h>
#include "arrays.hpp"
#include "defines.hpp"

namespace calibr8 {

class Disc;
class State;

//! \brief Evaluate the residual vector given measured displacement data
//! \param state The application state object
//! \param disc The discretization object
//! \param step The current load/time step
//! \details This will populate:
//!   state->la->b[GHOST] as the residual R
void eval_measured_residual(
    RCP<State> state,
    RCP<Disc> disc,
    int step);

//! \brief Evaluate the residual vector and its gradient given measured displacement data
//! \param state The application state object
//! \param disc The discretization object
//! \param dR The ghosted derivatives of the residual
//! \param local_sens The sensitivity matrices
//! \param step The current load/time step
//! \details This will populate:
//!   state->la->b[GHOST] as the residual R
//!   the residual gradient dR (ghost)
void eval_measured_residual_and_grad(
    RCP<State> state,
    RCP<Disc> disc,
    Array1D<RCP<MultiVectorT>>& dR,
    Array3D<EMatrix>& local_sens,
    int step);

//! \brief Evaluate the Jacobian matrix and residual vector
//! \param state The application state object
//! \param disc The discretization object
//! \param step The current load/time step
//! \details This will populate:
//!   state->la->A[GHOST] as the Jacobian dR_dx
//!   state->la->b[GHOST] as the residual R
void eval_forward_jacobian(
    RCP<State> state,
    RCP<Disc> disc,
    int step);

//! \brief Evaluate the residual vector (and wrong Jacobian)
//! \param state The application state object
//! \param disc The discretization object
//! \param step The current load/time step
//! \param evaluate_error Use the error weights to get error info
//! \details This will populate:
//!   state->la->b[GHOST] as the residual R
void eval_global_residual(
    RCP<State> state,
    RCP<Disc> disc,
    int step,
    bool evaluate_error=false,
    Array1D<apf::Field*> const& adjoint_fields=Array1D<apf::Field*>());

//! \brief Evaluate the Jacobian transpose matrix
//! \param state The application state object
//! \param disc The discretization object
//! \param g The local history variables
//! \param f The global history variables
//! \param step The current load/time step
//! \details This will populate:
//!   state->la->A[GHOST] as the Jacobian transpose dR_dxT
//!   state->local_history with a correction
void eval_adjoint_jacobian(
    RCP<State> state,
    RCP<Disc> disc,
    Array3D<EVector>& g,
    Array3D<EVector>& f,
    int step);

//! \brief Solve for the local adjoint variables
//! \param state The application state object
//! \param disc The discretization object
//! \param g The local history variables
//! \param f The global history variables
//! \param step The current load/time step
//! \details This will populate:
//!   state->global_history as the correct stuff
//!   state->local_history as the correct stuff without a correction
//!   state->adjoint_H.local
void solve_adjoint_local(
    RCP<State> state,
    RCP<Disc> disc,
    Array3D<EVector>& g,
    Array3D<EVector>& f,
    int step);

//! \brief Return the QoI evaluation at a step
//! \param state The application state object
//! \param disc The discretization object
//! \param step The current load/time step
double eval_qoi(RCP<State> state, RCP<Disc> disc, int step);

//! \brief Return the the QoI gradient evaluation at a step
//! \param state The application state object
//! \param step The current load/time step
Array1D<double> eval_qoi_gradient(RCP<State> state, int step);

//! \brief Evaluate contributions to the adjoint-based error at a step
//! \param state The application state object
//! \param disc The (nested) discretization object
//! \param R_error The global residual element-wise error field
//! \param C_error The local residual element-wise error field
//! \param step The current load/time step
void eval_error_contributions(
    RCP<State> state,
    RCP<Disc> disc,
    apf::Field* R_error,
    apf::Field* C_error,
    int step);

//! \brief Evaluate contributions to the linearization error at a step
//! \param state The application state object
//! \param disc The (nested) discretization object
//! \param step The current load/time step
//! \param E_lin_R The global residual linearization error
//! \param E_lin_C The local residual linearization error
//! \details Only valid for linear QoIs
void eval_linearization_errors(
    RCP<State> state,
    RCP<Disc> disc,
    int step,
    double& E_lin_R,
    double& E_lin_C);

//! \brief Evaluate the global residual linearization error vector and the
//! scalar local residual error
//! \param state The application state object
//! \param disc The (nested) discretization object
//! \param step The current load/time step
//! \param ELR The global residual linearization error in the global residual (vector)
//! \param E_lin_C The local residual linearization error in the QoI (scalar)
//! \details Only valid for linear QoIs
void eval_linearization_error_terms(
    RCP<State> state,
    RCP<Disc> disc,
    int step,
    Array1D<RCP<VectorT>>& ELR,
    apf::Field* C_error);

//! \brief Evalaute exact contributions to the error at a step
//! \param state The application state object
//! \param disc The (nested) discretization object
//! \param R_error The global residual element-wise error field
//! \param C_error The local residual element-wise error field
//! \param step The current load/time step
void eval_exact_errors(
    RCP<State> state,
    RCP<Disc> disc,
    apf::Field* R_error_field,
    apf::Field* C_error_field,
    int step);

//! \brief Evaluate and store the Cauchy stress tensor in a field
//! \param state The application state object
//! \param step The current load/time step
//! \details This is pretty specific to global residuals that deal
//! with momentum equations, and it assumes the pressure residual
//! is indexed at 1.
apf::Field* eval_cauchy(RCP<State> state, int step);

}
