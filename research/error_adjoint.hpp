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
    std::string m_error_field = "";             // the error field to use for adaptivity
  private:
    apf::Field* m_u_coarse = nullptr;           // the primal solution solved on the coarse space
    apf::Field* m_u_fine = nullptr;             // the primal solution solved on the fine space
    apf::Field* m_u_prolonged = nullptr;        // the coarse primal solution prolonged to the fine space
    apf::Field* m_u_restricted = nullptr;       // the fine primal solution restricted to the coarse space
    apf::Field* m_ue = nullptr;                 // the exact primal discretization error
    apf::Field* m_z_fine = nullptr;             // the adjoint solution solved on the fine space
    apf::Field* m_ERL = nullptr;                // the exact residual linearization error
    apf::Field* m_u_star = nullptr;             // the state at which qoi linearization errors vanish
    apf::Field* m_z_star = nullptr;             // the exact nonlinear adjoint solution
    apf::Field* m_z_star_star = nullptr;        // the exact shifted nonlinear adjoint solution
    apf::Field* m_R_prolonged = nullptr;        // the residual evaluated at the prolonged solution
    apf::Field* m_z_fine_diff = nullptr;        // the adjoint weight for AMR
    apf::Field* m_z_star_star_diff = nullptr;   // the adjoint weight for AMR
    apf::Field* m_eta1_local = nullptr;         // the traditional estimate localized
    apf::Field* m_eta2_local = nullptr;         // the new estimate localized
  private:
    std::vector<int> m_elems;
    std::vector<int> m_H_dofs;
    std::vector<int> m_h_dofs;
    std::vector<double> m_JH;
    std::vector<double> m_Jh;
    std::vector<double> m_eta1;
    std::vector<double> m_eta2;
    std::vector<double> m_etaR_L;
    std::vector<double> m_eta1_bound;
    std::vector<double> m_eta2_bound;
    std::vector<double> m_norm_ERL;
};

}
