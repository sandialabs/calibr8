#pragma once

#include "defines.hpp"

namespace calibr8 {

template <typename T>
class Residual {
  public:
    Residual();
    virtual ~Residual();
    int num_eqs() { return m_neqs; }
  protected:
    int m_neqs = -1;
};

}
