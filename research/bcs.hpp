#pragma once

#include "defines.hpp"
#include "system.hpp"

namespace calibr8 {

void apply_resid_dbcs(
    ParameterList const& dbcs,
    RCP<Disc> disc,
    RCP<System> sys,
    int space);

void apply_jacob_dbcs(
    ParameterList const& dbcs,
    RCP<Disc> disc,
    RCP<MatrixT> A,
    RCP<MatrixT> x,
    RCP<MatrixT> b,
    bool adjoint);

}
