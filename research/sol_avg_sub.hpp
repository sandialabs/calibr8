#pragma once

#include "disc.hpp"
#include "qoi.hpp"

namespace calibr8 {

template <typename T>
class SolAvgSub : public QoI<T> {

  private:

    std::string m_elem_set = "";
    bool m_evaluate = false;

  public:

    SolAvgSub(ParameterList const& params) : QoI<T>() {
      m_elem_set = params.get<std::string>("elem set");
    }

    ~SolAvgSub() override {
    }

    void before_elems(int es_idx, RCP<Disc> disc) override {
      int const qoi_es_idx = disc->elem_set_idx(m_elem_set);
      if (es_idx == qoi_es_idx) m_evaluate = true;
      else m_evaluate = false;
    }

    void at_point(
        apf::Vector3 const& xi,
        double w,
        double dv,
        RCP<Residual<T>> residual,
        RCP<Disc> disc) override {
      if (!m_evaluate) return;
      double const wdetJ = w*dv;
      T integrand = 0.;
      residual->interp_basis(xi, disc);
      Array1D<T> const vals = residual->interp(xi);
      for (int eq = 0; eq < this->m_neqs; ++eq) {
        integrand += vals[eq];
      }
      integrand /= this->m_neqs;
      this->m_elem_value += integrand * wdetJ;
    }

};


}
