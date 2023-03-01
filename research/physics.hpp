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
    apf::Field* solve_adjoint(int space, apf::Field* u);
  public:
    apf::Field* prolong_u_coarse_onto_fine(apf::Field* uH);
    apf::Field* restrict_u_fine_onto_fine(apf::Field* uh);
    apf::Field* recover_u_fine_from_u_coarse(apf::Field* u);
    apf::Field* subtract_u_coarse_from_u_fine(apf::Field* uh, apf::Field* uH);
    apf::Field* subtract_u_coarse_from_u_spr(apf::Field* uh_spr, apf::Field* uH);
  public:
    apf::Field* prolong_z_coarse_onto_fine(apf::Field* zH);
    apf::Field* restrict_z_fine_onto_fine(apf::Field* zh);
    apf::Field* recover_z_fine_from_z_coarse(apf::Field* z);
    apf::Field* subtract_z_coarse_from_z_fine(apf::Field* zh, apf::Field* zH);
    apf::Field* subtract_z_coarse_from_z_spr(apf::Field* zh_spr, apf::Field* zH);
  public:
    apf::Field* evaluate_fine_residual_at_coarse_u(apf::Field* uH_h);
    double dot(apf::Field* z, apf::Field* v,
        std::string const& str1, std::string const& str2);
  public:
    apf::Field* compute_residual_linearization_error(
        apf::Field* uH_h, apf::Field* uh_minus_uH_h, std::string const& str);
  private:
    RCP<ParameterList> m_params;
    RCP<Disc> m_disc;
    RCP<Residual<double>> m_residual;
    RCP<Residual<FADT>> m_jacobian;
    RCP<QoI<double>> m_qoi;
    RCP<QoI<FADT>> m_qoi_deriv;
};

}
