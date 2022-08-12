#pragma once

#include <apf.h>
#include "disc.hpp"
#include "defines.hpp"
#include "qoi.hpp"
#include "residual.hpp"

namespace calibr8 {

class Physics {
  public:
    Physics(RCP<ParameterList> params);
    void build_disc();
    apf::Field* solve_primal(int space);
    apf::Field* solve_adjoint(int space, apf::Field* u);
    apf::Field* prolong_u_coarse_onto_fine(apf::Field* u);
    double compute_qoi(int space, apf::Field* u);
    RCP<Disc> disc() { return m_disc; }
  private:
    RCP<ParameterList> m_params;
    RCP<Disc> m_disc;
    RCP<Residual<double>> m_residual;
    RCP<Residual<FADT>> m_jacobian;
    RCP<QoI<double>> m_qoi;
    RCP<QoI<FADT>> m_qoi_deriv;
};

apf::Field* subtract(
    RCP<Disc> disc,
    apf::Field* a,
    apf::Field* b,
    std::string const& name);

}
