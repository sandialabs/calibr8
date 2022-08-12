#pragma once

#include <apf.h>
#include "disc.hpp"
#include "defines.hpp"
#include "qoi.hpp"
#include "residual.hpp"

namespace calibr8 {

double inline add(double a, double b) { return a+b; }
double inline subtract(double a, double b) { return a-b; }
double inline multiply(double a, double b) { return a*b; }

class Physics {
  public:
    Physics(RCP<ParameterList> params);
    RCP<Disc> disc() { return m_disc; }
    void build_disc();
    void destroy_disc();
    apf::Field* solve_primal(int space);
    apf::Field* solve_adjoint(int space, apf::Field* u);
    apf::Field* prolong_u_coarse_onto_fine(apf::Field* u);
    apf::Field* restrict_z_fine_onto_fine(apf::Field* z);
    apf::Field* compute_linearization_error(
        apf::Field* uH_h,
        apf::Field* uh_minus_uH_h,
        double& norm_R,
        double& norm_E);
    double compute_qoi(int space, apf::Field* u);
    double compute_eta(apf::Field* u, apf::Field* z);
    double compute_eta_L(apf::Field* z, apf::Field* E_L);
    apf::Field* op(
        std::function<double(double,double)> f,
        apf::Field* a,
        apf::Field* b,
        std::string const& name);
  private:
    RCP<ParameterList> m_params;
    RCP<Disc> m_disc;
    RCP<Residual<double>> m_residual;
    RCP<Residual<FADT>> m_jacobian;
    RCP<QoI<double>> m_qoi;
    RCP<QoI<FADT>> m_qoi_deriv;
};

}
