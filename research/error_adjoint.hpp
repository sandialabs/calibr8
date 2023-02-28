#pragma once

#include "error.hpp"

namespace calibr8 {

class Adjoint : public Error {
  public:
    Adjoint(ParameterList const& params);
    apf::Field* compute_error(RCP<Physics> physics) override;
    void destroy_intermediate_fields() override;
    void write_history(std::string const& file, double J_ex) override;
  private:
    apf::Field* m_uH = nullptr;                   // the primal solution solved on the coarse space
    apf::Field* m_uh = nullptr;                   // the primal solution solved on the fine space
  private:
    apf::Field* m_zH = nullptr;                   // the adjoint solution solved on the coarse space
    apf::Field* m_zh = nullptr;                   // the adjoint solution solved on the fine space
  private:
    apf::Field* m_uH_h = nullptr;                 // the prolongation of uH onto h
    apf::Field* m_uh_H = nullptr;                 // the restriction of uh onto H, represented on h
    apf::Field* m_uh_spr = nullptr;               // the recovery of uH onto h via SPR
    apf::Field* m_uh_minus_m_uH_h = nullptr;      // the exact primal discretization error
    apf::Field* m_uh_spr_minus_m_uH_h = nullptr;  // two approximated primal discretization error
  private:
    apf::Field* m_zH_h = nullptr;                 // the prolongation of zH onto h
    apf::Field* m_zh_H = nullptr;                 // the restriction of zh onto H, represented on h
    apf::Field* m_zh_spr = nullptr;               // the recovery of zH onto h via SPR
    apf::Field* m_zh_minus_m_zh_H = nullptr;      // the exact adjoint discretization error
    apf::Field* m_zh_spr_minus_m_zH_h = nullptr;  // the approximated adjoint discretization error
  private:
    apf::Field* m_Rh_uH_h = nullptr;              // the fine residual evaluated at the prolonged solution
  private:
    std::vector<int> m_nelems;
    std::vector<int> m_H_dofs;
    std::vector<int> m_h_dofs;
    std::vector<double> m_JH;
    std::vector<double> m_Jh;
    std::vector<double> m_eta1;
    std::vector<double> m_eta2;
    std::vector<double> m_eta1_spr;
    std::vector<double> m_eta2_spr;
  private:
    void solve_primal(int space, RCP<Physics> physics);
    void post_process_primal(RCP<Physics> physics);
    void solve_adjoint(RCP<Physics> physics);
    void post_process_adjoint(RCP<Physics> physics);
    void compute_first_order_errors(RCP<Physics> physics);
};

}
