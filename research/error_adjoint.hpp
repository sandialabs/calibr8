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
    apf::Field* m_u_coarse = nullptr;     // the primal solution solved on the coarse space
    apf::Field* m_u_fine = nullptr;       // the primal solution solved on the fine space
    apf::Field* m_u_prolonged = nullptr;  // the coarse primal solution prolonged to the fine space
    apf::Field* m_u_restricted = nullptr; // the fine primal solution restricted to the coarse space
    apf::Field* m_u_recovered = nullptr;  // the SPR primal solution on the fine space
  private:
    apf::Field* m_e_exact = nullptr;        // the exact discretization error
    apf::Field* m_e_linearized = nullptr;   // the linearized discretization error
    apf::Field* m_e_recovered = nullptr;    // the discretization error approximated using SPR



//  private:
//    apf::Field* m_error_exact = nullptr;        // the exact discretization error
//    apf::Field* m_error_linearized = nullptr;   // the linearized discretization error
//    apf::Field* m_z_coarse = nullptr;     // the adjoint solution on the coarse space
//    apf::Field* m_z_fine = nullptr;       // the adjoint solution on the fine space
//
//
//
//    apf::Field* m_uH_h = nullptr;   // the prolongation of uH onto the fine space
//    apf::Field* m_elh = nullptr;    // the linearized discretization error solution
//    apf::Field* m_zh = nullptr;     // the adjoint solution solved on the fine space
//    apf::Field* m_yh = nullptr;     // the 2nd order adjoint solved on the fine space
//    apf::Field* m_Rh_uH = nullptr;  // the fine space residual evaluated at the prolonged solution
  private:
    std::vector<int> m_nelems;
    std::vector<int> m_H_dofs;
    std::vector<int> m_h_dofs;
    std::vector<double> m_JH;
    std::vector<double> m_Jh;
};

}
