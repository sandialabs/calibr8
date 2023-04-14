#pragma once

#include <apf.h>
#include "disc.hpp"
#include "defines.hpp"
#include "qoi.hpp"
#include "residual.hpp"

namespace calibr8 {

struct nonlinear_in {
  std::string name_append = "";
  apf::Field* u_coarse = nullptr;
  apf::Field* u_fine = nullptr;
  apf::Field* ue = nullptr;
  double J_coarse = 0.;
  double J_fine = 0.;
};

struct nonlinear_out {
  apf::Field* u_star = nullptr;
  apf::Field* z_star = nullptr;
};

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
    apf::Field* solve_adjoint(int space, apf::Field* u);
    apf::Field* solve_linearized_error(apf::Field* u, std::string const& n);
    apf::Field* solve_2nd_adjoint(apf::Field* u, apf::Field* ee, std::string const& n);
    apf::Field* solve_ERL(apf::Field* u, apf::Field* ue, std::string const& n);
    nonlinear_out solve_nonlinear_adjoint(nonlinear_in in);
    apf::Field* evaluate_residual(int space, apf::Field* u);
    apf::Field* subtract(apf::Field* f, apf::Field* g, std::string const& n);
    apf::Field* prolong(apf::Field* f, std::string const& n);
    apf::Field* restrict(apf::Field* f, std::string const& n);
    apf::Field* recover(apf::Field* f, std::string const& n);
    apf::Field* modify_star(
        apf::Field* z, apf::Field* R, apf::Field* E, std::string const& n);
    apf::Field* diff(apf::Field* z, std::string const& n);
    apf::Field* localize(apf::Field* R, apf::Field* z, std::string const& n);
  public:
    double compute_qoi(int space, apf::Field* u);
    double dot(apf::Field* a, apf::Field* b);
    double compute_sum(apf::Field* e);
    double compute_bound(apf::Field* e);
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
