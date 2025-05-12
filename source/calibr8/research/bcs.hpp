#pragma once

#include "defines.hpp"
#include "linalg.hpp"

namespace calibr8 {

void apply_resid_dbcs(
    ParameterList const& dbcs,
    int space,
    RCP<Disc> disc,
    RCP<VectorT> u,
    System& sys);

void apply_jacob_dbcs(
    ParameterList const& dbcs,
    int space,
    RCP<Disc> disc,
    RCP<VectorT> u,
    System& sys,
    bool adjoint);

}
