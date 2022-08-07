#pragma once

#include "defines.hpp"

namespace calibr8 {

template <typename T>
class Residual {
  public:
    Residual(int ndims);
    virtual ~Residual();
    int num_eqs() { return m_neqs; }
    void gather(apf::Field* u, apf::MeshElement* me);
    void interpolate(apf::Vector3 const& xi);
    virtual void at_point(apf::Vector3 const& xi, double w, double dv) = 0;
    void scatter();
  protected:
    int m_neqs = -1;
    int m_ndims = -1;
    int m_nnodes = -1;
    std::vector<std::vector<T>> m_vals;
    std::vector<std::vector<T>> m_resid;
};

template <typename T>
RCP<Residual<T>> create_residual(ParameterList const& params, int ndims);

}
