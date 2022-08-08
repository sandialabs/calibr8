#pragma once

#include "defines.hpp"
#include "linalg.hpp"

namespace calibr8 {

void apply_jacob_dbcs(
    ParameterList const& dbcs,
    int space,
    RCP<Disc> disc,
    System& sys,
    bool adjoint);

}
