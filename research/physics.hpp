#pragma once

#include <apf.h>
#include "disc.hpp"
#include "defines.hpp"
#include "residual.hpp"

namespace calibr8 {

struct Fields {
  apf::Field* u[NUM_SPACE] = {nullptr};
  apf::Field* z[NUM_SPACE] = {nullptr};
  apf::Field* uH_h = nullptr;
  void project_uH_onto_h(RCP<Disc> disc);
  void destroy();
};

apf::Field* solve_primal(
    int space,
    RCP<ParameterList> params,
    RCP<Disc> disc,
    RCP<Residual<double>> resid,
    RCP<Residual<FADT>> jacobian);

}
