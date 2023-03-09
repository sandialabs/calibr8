#pragma once

#include "qoi.hpp"

namespace calibr8 {

template <typename T>
class QoI1 : public QoI<T> {

  private:

    int m_eq = -1;
    double m_beta = -1;

  public:

    QoI1(ParameterList const& params) : QoI<T> () {
      m_eq = params.get<int>("eq");
      m_beta = params.get<double>("beta");
    }

    ~QoI1() override {
    }

    void at_point(
        apf::Vector3 const& xi,
        double w,
        double dv,
        RCP<Residual<T>> residual,
        RCP<Disc> disc) override {
      residual->interp_basis(xi, disc);
      Array1D<T> const soln = residual->interp(xi);
      T const u = soln[m_eq];
      T const integrand = pow(u, m_beta);
      this->m_elem_value += integrand * w * dv;
    }

};

}
