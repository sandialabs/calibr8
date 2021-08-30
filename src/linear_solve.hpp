#pragma once

//! \file linear_solve.hpp
//! \brief Helpers for solving linear systems

#include "arrays.hpp"
#include "defines.hpp"

namespace calibr8 {

//! \cond
// forward declarations
class Disc;
//! \endcond

//! \brief Solve a block linear system Ax = b
//! \param params The solver parameters
//! \param disc The discretization object
//! \param A The block matrix A
//! \param x The block solution vector x
//! \param b The block right hand side vector b
void solve(
    ParameterList& params,
    RCP<Disc> disc,
    Array2D<RCP<MatrixT>>& A,
    Array1D<RCP<VectorT>>& x,
    Array1D<RCP<VectorT>>& b);

}
