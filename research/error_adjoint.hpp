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
    bool m_linear_physics = false;                            // whether or not the PDE residual is linear
    bool m_linear_qoi = false;                                // whether or not the QoI is linear
    std::string m_error_field = "";                           // the error field to use for adaptivity
  private:
    apf::Field* m_u_coarse = nullptr;                         // the primal solution solved on the coarse space
    apf::Field* m_u_fine = nullptr;                           // the primal solution solved on the fine space
    apf::Field* m_u_prolonged = nullptr;                      // the coarse primal solution prolonged to the fine space
    apf::Field* m_u_restricted = nullptr;                     // the fine primal solution restricted to the coarse space
    apf::Field* m_u_recovered = nullptr;                      // the SPR primal solution on the fine space
  private:
    apf::Field* m_ue_exact = nullptr;                         // the exact primal discretization error
    apf::Field* m_ue_recovered = nullptr;                     // the primal discretization error approximated using SPR
  private:
    apf::Field* m_z_coarse = nullptr;                         // the adjoint solution solved on the coarse space
    apf::Field* m_z_fine = nullptr;                           // the adjoint solution solved on the fine space
    apf::Field* m_z_recovered = nullptr;                      // the SPR adjoint solution on the fine space
  private:
    apf::Field* m_y_exact = nullptr;                          // the 2nd order adjoint solution using the exact error
    apf::Field* m_y_recovered = nullptr;                      // the 2nd order adjoint solution using the recovered error
  private:
    apf::Field* m_ERL_exact = nullptr;                        // the exact residual linearization error
    apf::Field* m_ERL_recovered = nullptr;                    // the residual linearization error computed using SPR recovery
  private:
    apf::Field* m_u_star = nullptr;                           // the state at which qoi linearization errors vanish
    apf::Field* m_z_star = nullptr;                           // the exact nonlinear adjoint solution
    apf::Field* m_z_star_star = nullptr;                      // the exact shifted nonlinear adjoint solution
    apf::Field* m_u_star_recovered = nullptr;                 // the recovered approximation to u_star
    apf::Field* m_z_star_recovered = nullptr;                 // the recovered approximation to z_star
    apf::Field* m_z_star_star_recovered = nullptr;            // the recovered shifted nonlinear adjoint solution
  private:
    apf::Field* m_R_prolonged = nullptr;                      // the residual evaluated at the prolonged solution
  private:
    apf::Field* m_z_fine_diff = nullptr;                      // the adjoint weight for AMR
    apf::Field* m_z_star_star_diff = nullptr;                 // the adjoint weight for AMR
    apf::Field* m_z_star_star_recovered_diff = nullptr;       // the adjoint weight for AMR
    apf::Field* m_y_exact_diff = nullptr;                     // the second-order adjoint weight for AMR
    apf::Field* m_y_recovered_diff = nullptr;                 // the second-order adjoint weight for AMR
  private:
    apf::Field* m_eta1_local = nullptr;                       // the traditional estimate localized
    apf::Field* m_eta_local = nullptr;                        // the new estimate localized
    apf::Field* m_eta_tilde_local = nullptr;                  // the new recovered estimate localized
    apf::Field* m_eta_quad_local = nullptr;                   // the new quadratic estimate localized
    apf::Field* m_eta_quad_tilde_local = nullptr;             // the new recovered quadratic estimate localized
  private:
    std::vector<int> m_elems;
    std::vector<int> m_H_dofs;
    std::vector<int> m_h_dofs;
    std::vector<double> m_JH;
    std::vector<double> m_Jh;
    std::vector<double> m_eta1;
    std::vector<double> m_eta2;
    std::vector<double> m_eta3;
    std::vector<double> m_eta4;
    std::vector<double> m_eta2_tilde;
    std::vector<double> m_eta3_tilde;
    std::vector<double> m_eta4_tilde;
    std::vector<double> m_eta;
    std::vector<double> m_eta_tilde;
    std::vector<double> m_eta_quad;
    std::vector<double> m_eta_quad_tilde;
    std::vector<double> m_eta1_bound;
    std::vector<double> m_eta_bound;
    std::vector<double> m_eta_tilde_bound;
    std::vector<double> m_eta_quad_bound;
    std::vector<double> m_eta_quad_tilde_bound;
    std::vector<double> m_norm_ERL;
};

}
