#pragma once

#include <Teuchos_ParameterList.hpp>
#include "disc.hpp"

namespace calibr8 {

void snap_nodes(
    RCP<Disc> disc,
    Teuchos::ParameterList const& p);

}
