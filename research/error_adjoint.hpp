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
    void solve_primal(RCP<Physics> physics, double& JH, double& Jh);
    void solve_adjoint(RCP<Physics> physics);
    void compute_adjoint_weight(RCP<Physics> physics);
    void evaluate_prolonged_residual(RCP<Physics> physics);
    void compute_linearization_error(RCP<Physics> physics, double& eta_L);
    void localize_error(RCP<Physics> physics);
    void compute_error(RCP<Physics> physics, double& eta, double& eta_bound);
    void collect_data(
        RCP<Physics> physics,
        double JH,
        double Jh,
        double eta,
        double eta_bound,
        double eta_L);
  private:
    apf::Field* m_uH = nullptr;
    apf::Field* m_uh = nullptr;
    apf::Field* m_uH_h = nullptr;
    apf::Field* m_uh_minus_uH_h = nullptr;
    apf::Field* m_zH = nullptr;
    apf::Field* m_zh = nullptr;
    apf::Field* m_zh_H = nullptr;
    apf::Field* m_zH_h = nullptr;
    apf::Field* m_zh_spr = nullptr;
    apf::Field* m_zh_H_spr = nullptr;
    apf::Field* m_z_weight = nullptr;
    apf::Field* m_Rh_uH_h = nullptr;
    apf::Field* m_Rh_uH_h_plus_ELh = nullptr;
    apf::Field* m_ELh = nullptr;
    apf::Field* m_eta = nullptr;
  private:
    int adjoint = -1;
    int localization = -1;
    bool subtraction = false;
    bool linearization = false;
    std::vector<int> m_nelems;
    std::vector<int> m_H_dofs;
    std::vector<int> m_h_dofs;
    std::vector<double> m_JH;
    std::vector<double> m_Jh;
    std::vector<double> m_estimate;
    std::vector<double> m_estimate_bound;
    std::vector<double> m_estimate_L;
};

}
