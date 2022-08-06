#pragma once

#include "residual.hpp"

namespace calibr8 {

template <typename T>
class NLPoisson : public Residual<T> {
  public:
    NLPoisson(ParameterList const& params, int ndims) {
      m_nims = ndims;
      this->m_neqs = 1;
    }
    ~NLPoisson() override {
    }
  private:
    int m_nims = -1;
};

}
