#include "qoi.hpp"

namespace calibr8 {

template <typename T>
class SolAvg : public QoI<T> {

  public:

    SolAvg(ParameterList const& params) : QoI<T>() {
      (void)params;
    }

    ~SolAvg() override {
    }

    void at_point(
        apf::Vector3 const& xi,
        double w,
        double dv,
        RCP<Residual<T>> residual,
        RCP<Disc> disc) override {
      double const wdetJ = w*dv;
      T integrand = 0.;
      Array1D<T> const vals = residual->interp(xi, disc);
      for (int eq = 0; eq < this->m_neqs; ++eq) {
        integrand += vals[eq];
      }
      integrand /= this->m_neqs;
      this->m_elem_value += integrand * w * dv;
    }

};

}
