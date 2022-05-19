#pragma once

//! \file dbcs.hpp
//! \brief Methods for Dirichlet boundary conditions

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

//! \brief Apply DBCs to the primal system from an analytical expression
//! \param dbcs The dirichlet BC parameter lsit
//! \param disc The discretization object
//! \param dR_dx The OWNED Jacobian matrices
//! \param R The OWNED residual vectors
//! \param x The apf fields corresponding to the global solutions
//! \param t The current time
void apply_expression_primal_dbcs(
    ParameterList const& dbcs,
    RCP<Disc> disc,
    Array2D<RCP<MatrixT>>& dR_dx,
    Array1D<RCP<VectorT>>& R,
    Array1D<apf::Field*>& x,
    double t,
    bool is_adjoint = false);

//! \brief Apply DBCs to the primal system from a field
//! \param dbcs The dirichlet BC parameter lsit
//! \param disc The discretization object
//! \param dR_dx The OWNED Jacobian matrices
//! \param R The OWNED residual vectors
//! \param x The apf fields corresponding to the global solutions
//! \param t The current time
void apply_field_primal_dbcs(
    ParameterList const& dbcs,
    RCP<Disc> disc,
    Array2D<RCP<MatrixT>>& dR_dx,
    Array1D<RCP<VectorT>>& R,
    Array1D<apf::Field*>& x,
    double t,
    int step,
    std::string const& prefix,
    bool is_adjoint = false);

//! \brief Apply DBCs to the primal system
//! \param dbcs The dirichlet BC parameter lsit
//! \param disc The discretization object
//! \param dR_dx The OWNED Jacobian matrices
//! \param R The OWNED residual vectors
//! \param x The apf fields corresponding to the global solutions
//! \param t The current time
void apply_primal_dbcs(
    ParameterList const& dbcs,
    RCP<Disc> disc,
    Array2D<RCP<MatrixT>>& dR_dx,
    Array1D<RCP<VectorT>>& R,
    Array1D<apf::Field*>& x,
    double t,
    int step,
    bool is_adjoint = false);

}
