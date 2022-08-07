#pragma once

#include "residual.hpp"

namespace calibr8 {

template <typename T>
class NLPoisson : public Residual<T> {
  public:
    NLPoisson(ParameterList const& params, int ndims) : Residual<T>(ndims) {
      this->m_neqs = 1;
      m_alpha = params.get<double>("alpha");
    }
    ~NLPoisson() override {
    }
    void at_point(apf::Vector3 const& xi, double w, double dv) override {
      (void)xi;
      (void)w;
      (void)dv;
    }
  private:
    double m_alpha = 0.;
};

}
