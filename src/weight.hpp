#pragma once

#include <apf.h>

namespace calibr8 {

class Weight {
  public:
    Weight(apf::FieldShape* shape);
    void evaluate(apf::MeshElement* me, apf::Vector3 const& iota);
    double val(int i, int n, int eq);
    double grad(int i, int n, int eq, int dim);
  protected:
    apf::FieldShape* m_shape = nullptr;
    apf::NewArray<double> m_basis;
    apf::NewArray<apf::Vector3> m_grad_basis;
};

}
