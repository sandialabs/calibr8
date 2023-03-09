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
    apf::Field* m_uH = nullptr;     // the primal solution solved on the coarse space
    apf::Field* m_uh = nullptr;     // the primal solution solved on the fine space
    apf::Field* m_uH_h = nullptr;   // the prolongation of uH onto the fine space
    apf::Field* m_eh = nullptr;     // the exact error in the primal solution
    apf::Field* m_elh = nullptr;    // the linearized discretization error solution
    apf::Field* m_zh = nullptr;     // the adjoint solution solved on the fine space
    apf::Field* m_yh = nullptr;     // the 2nd order adjoint solved on the fine space
    apf::Field* m_Rh_uH = nullptr;  // the fine space residual evaluated at the prolonged solution
  private:
    std::vector<int> m_nelems;
    std::vector<int> m_H_dofs;
    std::vector<int> m_h_dofs;
    std::vector<double> m_JH;
    std::vector<double> m_Jh;
};

}
