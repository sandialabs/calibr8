#pragma once

#include "defines.hpp"
#include "linalg.hpp"

namespace calibr8 {

void apply_jacob_dbcs(
    ParameterList const& dbcs,
    int space,
    RCP<Disc> disc,
    RCP<VectorT> U,
    System& sys,
    bool adjoint);

}
