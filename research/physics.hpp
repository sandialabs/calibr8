#pragma once

#include <apf.h>
#include "disc.hpp"
#include "defines.hpp"
#include "qoi.hpp"
#include "residual.hpp"

namespace calibr8 {

struct Fields {
  apf::Field* u[NUM_SPACE] = {nullptr};
  apf::Field* z[NUM_SPACE] = {nullptr};
  apf::Field* uH_h = nullptr;
  apf::Field* zH_h = nullptr;
  apf::Field* uh_minus_uH_h = nullptr;
  apf::Field* zh_minus_zH_h = nullptr;
  apf::Field* E_L = nullptr;
  void destroy();
};

apf::Field* project(RCP<Disc> disc, apf::Field* from, std::string const& name);
apf::Field* subtract(RCP<Disc> disc, apf::Field* a, apf::Field* b, std::string const& name);

apf::Field* solve_primal(
    int space,
    RCP<ParameterList> params,
    RCP<Disc> disc,
    RCP<Residual<double>> resid,
    RCP<Residual<FADT>> jacobian);

double compute_qoi(
    int space,
    RCP<ParameterList> params,
    RCP<Disc> disc,
    RCP<Residual<double>> resid,
    RCP<QoI<double>> qoi,
    apf::Field* u_space);

apf::Field* compute_linearization_error(
    RCP<ParameterList> params,
    RCP<Disc> disc,
    RCP<Residual<double>> resid,
    RCP<Residual<FADT>> jacobian,
    apf::Field* uH_h,
    apf::Field* uh_minus_uH_h);

double compute_volume(apf::Mesh* mesh);

}
