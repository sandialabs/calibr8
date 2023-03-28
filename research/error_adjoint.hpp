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
    apf::Field* m_u_coarse = nullptr;           // the primal solution solved on the coarse space
    apf::Field* m_u_fine = nullptr;             // the primal solution solved on the fine space
    apf::Field* m_u_prolonged = nullptr;        // the coarse primal solution prolonged to the fine space
    apf::Field* m_u_restricted = nullptr;       // the fine primal solution restricted to the coarse space
    apf::Field* m_u_recovered = nullptr;        // the SPR primal solution on the fine space
  private:
    apf::Field* m_ue_exact = nullptr;           // the exact primal discretization error
    apf::Field* m_ue_recovered = nullptr;       // the primal discretization error approximated using SPR
  private:
    apf::Field* m_z_coarse = nullptr;           // the adjoint solution solved on the coarse space
    apf::Field* m_z_fine = nullptr;             // the adjoint solution solved on the fine space
    apf::Field* m_z_prolonged = nullptr;        // the coarse adjoint solution prolonged to the fine space
    apf::Field* m_z_restricted = nullptr;       // the fine adjoint solution restricted to the coarse space
    apf::Field* m_z_restricted_fine = nullptr;  // the restricted adjoint prolonged onto the fine space
    apf::Field* m_z_recovered = nullptr;        // the SPR adjoint solution on the fine space
  private:
    apf::Field* m_ze_exact = nullptr;           // the exact adjoint discretization error
    apf::Field* m_ze_restricted = nullptr;      // the adjoint discretization error using the restricted adjoint
    apf::Field* m_ze_recovered = nullptr;       // the adjoint discretization error approximated using SPR
  private:
    apf::Field* m_y_exact = nullptr;            // the 2nd order adjoint solution using the exact error
    apf::Field* m_y_recovered = nullptr;        // the 2nd order adjoint solution using the recovered error
  private:
    apf::Field* m_ERL_exact = nullptr;          // the exact residual linearization error
    apf::Field* m_ERL_recovered = nullptr;      // the residual linearization error computed using SPR recovery
  private:
    apf::Field* m_u_star = nullptr;           // the state at which qoi linearization errors vanish
    apf::Field* m_z_star = nullptr;             // the exact nonlinear adjoint solution
  private:
    apf::Field* m_R_prolonged = nullptr;        // the residual evaluated at the prolonged solution
};

}
