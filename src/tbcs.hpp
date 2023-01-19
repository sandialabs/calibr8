#pragma once

//! \file tbcs.hpp
//! \brief Methods for traction boundary condtions

#include "arrays.hpp"
#include "defines.hpp"

//! \cond
// forward declarations
namespace apf {
class Field;
}
//! \endcond

namespace calibr8 {

//! \cond
// forward declarations
class Disc;
//! \endcond

//! \brief Apply traction BCs to the primal system
//! \param tbcs The traction BC parameter list
//! \param disc The discretization object
//! \param R The OWNED residual vectors
//! \param t The current time
//! \details This is specific to mechanics quantities,
//! specifically displacements
void apply_primal_tbcs(
    ParameterList const& tbcs,
    RCP<Disc> disc,
    Array1D<RCP<VectorT>>& R,
    double t);

//! \brief Apply adjoint-weighted residual traction BCs to the error
//! \param tbcs The traction BC parameter list
//! \param disc The discretization object
//! \param zfields The adjoint solution fields
//! \param R_error The element contributions to the global resid error
//! \param t The current time
void eval_tbcs_error_contributions(
    ParameterList const& tbcs,
    RCP<Disc> disc,
    Array1D<apf::Field*> zfields,
    apf::Field* R_error,
    double t);

//! \brief Sum the adjoint-weighted residual traction BCs contribution to the error
//! \param tbcs The traction BC parameter list
//! \param disc The discretization object
//! \param zfields The adjoint solution fields
//! \param t The current time
double sum_tbcs_error_contributions(
    ParameterList const& tbcs,
    RCP<Disc> disc,
    Array1D<apf::Field*> zfields,
    double t);

//! \brief Apply adjoint-weighted residual traction BCs to the error
//! \param tbcs The traction BC parameter list
//! \param disc The discretization object
//! \param zfields The adjoint solution fields
//! \param R_error The element contributions to the global resid error
//! \param t The current time
void eval_tbcs_error_contributions(
    ParameterList const& tbcs,
    RCP<Disc> disc,
    Array1D<apf::Field*> zfields,
    Array1D<RCP<VectorT>>& R_error,
    double t);

}
