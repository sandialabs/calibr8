#pragma once

#include "error.hpp"

namespace calibr8 {

class R_dot_zh_minus_zh_H : public Error {
  public:
    R_dot_zh_minus_zh_H(int ltype) : localization(ltype) {}
    apf::Field* compute_error(RCP<Physics> physics) override;
    void destroy_intermediate_fields() override;
    void write_history(std::string const& file, double J_ex) override;
  private:
    apf::Field* m_uH = nullptr;
    apf::Field* m_uh = nullptr;
    apf::Field* m_uH_h = nullptr;
    apf::Field* m_zh = nullptr;
    apf::Field* m_zh_H = nullptr;
    apf::Field* m_zh_minus_zh_H = nullptr;
    apf::Field* m_Rh_uH_h = nullptr;
    apf::Field* m_eta = nullptr;
  private:
    int localization = -1;
    std::vector<int> m_nelems;
    std::vector<int> m_H_dofs;
    std::vector<int> m_h_dofs;
    std::vector<double> m_JH;
    std::vector<double> m_Jh;
    std::vector<double> m_estimate;
    std::vector<double> m_estimate_bound;
};

}
