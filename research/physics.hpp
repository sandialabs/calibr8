#pragma once

#include <apf.h>
#include "disc.hpp"
#include "defines.hpp"
#include "qoi.hpp"
#include "residual.hpp"

namespace calibr8 {

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

apf::Field* solve_adjoint(
    int space,
    RCP<ParameterList> params,
    RCP<Disc> disc,
    RCP<Residual<FADT>> jacobian,
    RCP<QoI<FADT>> qoi,
    apf::Field* u_space);

struct LE {
  apf::Field* field;
  double E_L;
  double Rh_uH_h;
};

LE compute_linearization_error(
    RCP<ParameterList> params,
    RCP<Disc> disc,
    RCP<Residual<double>> resid,
    RCP<Residual<FADT>> jacobian,
    apf::Field* uH_h,
    apf::Field* uh_minus_uH_h);

void do_stuff(
    RCP<Disc> disc,
    RCP<Residual<double>> resid,
    apf::Field* zh,
    apf::Field* uH_h);

}
