#pragma once

#include <apf.h>

#include "qoi.hpp"

namespace calibr8 {

template <typename T>
class QoI2 : public QoI<T> {

  private:

    int m_eq = -1;
    double m_xmin = 0.;
    double m_xmax = 0.;
    double m_ymin = 0.;
    double m_ymax = 0.;
    double m_beta = -1;

  public:

    QoI2(ParameterList const& params) : QoI<T>() {
      m_eq = params.get<int>("eq");
      m_xmin = params.get<double>("xmin");
      m_xmax = params.get<double>("xmax");
      m_ymin = params.get<double>("ymin");
      m_ymax = params.get<double>("ymax");
      m_beta = params.get<double>("beta");
    }

    ~QoI2() override {
    }

    bool is_inside(apf::Vector3 const& x) {
      if ((m_xmin <= x[0]) && (x[0] <= m_xmax) &&
          (m_ymin <= x[1]) && (x[1] <= m_ymax)) {
        return true;
      } else {
        return false;
      }
    }

    void at_point(
        apf::Vector3 const& xi,
        double w,
        double dv,
        RCP<Residual<T>> residual,
        RCP<Disc> disc) override {
      apf::Vector3 x;
      apf::mapLocalToGlobal(this->m_mesh_elem, xi, x);
      if (!is_inside(x)) return;
      residual->interp_basis(xi, disc);
      Array1D<T> const soln = residual->interp(xi);
      T const u = soln[m_eq];
      T const integrand = pow(u, m_beta);
      this->m_elem_value += integrand * w * dv;
    }

};

}
