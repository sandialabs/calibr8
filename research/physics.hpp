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
    void build_disc();
    void destroy_disc();
    void destroy_residual_data();
    apf::Field* solve_primal(int space);
    double compute_qoi(int space, apf::Field* u);
    apf::Field* solve_adjoint(int space, apf::Field* u);
    apf::Field* prolong_u_coarse_onto_fine(apf::Field* u);
    apf::Field* prolong_z_coarse_onto_fine(apf::Field* z);
    apf::Field* subtract_u_coarse_from_u_fine(apf::Field* uh, apf::Field* uH);
    apf::Field* restrict_z_fine_onto_fine(apf::Field* z);
    apf::Field* subtract_z_coarse_from_z_fine(apf::Field* zh, apf::Field* zH);
    apf::Field* add_R_fine_to_EL_fine(apf::Field* Rh, apf::Field* ELh);
    apf::Field* recover_z_fine_from_z_coarse(apf::Field* zH);
    apf::Field* evaluate_residual(int space, apf::Field* u);
    apf::Field* evaluate_PU_residual(int space, apf::Field* u, apf::Field* z);
    apf::Field* localize_error(apf::Field* R, apf::Field* z);
    apf::Field* interpolate_to_ips(apf::Field* z);
    double estimate_error(apf::Field* eta);
    double estimate_error_bound(apf::Field* eta);
    double estimate_error2(apf::Field* R, apf::Field* Z);
    apf::Field* compute_linearization_error(
        apf::Field* uH_h,
        apf::Field* uh_minus_uH_h);
    double compute_eta_L(apf::Field* z, apf::Field* E_L);

    // debug
    apf::Field* compute_eta2(apf::Field* u, apf::Field* z);

  private:
    RCP<ParameterList> m_params;
    RCP<Disc> m_disc;
    RCP<Residual<double>> m_residual;
    RCP<Residual<FADT>> m_jacobian;
    RCP<QoI<double>> m_qoi;
    RCP<QoI<FADT>> m_qoi_deriv;
};

}
