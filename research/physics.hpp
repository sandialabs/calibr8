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
    RCP<Disc> disc() { return m_disc; }
  public:
    void build_disc();
    void destroy_disc();
    void destroy_residual_data();
  public:
    apf::Field* solve_primal(int space);
    double compute_qoi(int space, apf::Field* u);
    apf::Field* prolong_u_coarse_onto_fine(apf::Field* uH);
    apf::Field* solve_adjoint(int space, apf::Field* u);
    apf::Field* solve_linearized_error(apf::Field* u);
    apf::Field* solve_2nd_adjoint(apf::Field* u, apf::Field* e);
  private:
    RCP<ParameterList> m_params;
    RCP<Disc> m_disc;
    RCP<Residual<double>> m_residual;
    RCP<Residual<FADT>> m_jacobian;
    RCP<Residual<FAD2T>> m_hessian;
    RCP<QoI<double>> m_qoi;
    RCP<QoI<FADT>> m_qoi_deriv;
    RCP<QoI<FAD2T>> m_qoi_hessian;
};

}
