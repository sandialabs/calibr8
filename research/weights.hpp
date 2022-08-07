#pragma once

#include "apf.h"

namespace calibr8 {

class Weight {
  public:
    Weight(apf::FieldShape* shape);
    virtual void evaluate(apf::MeshElement* me, apf::Vector3 const& xi);
    virtual double val(int node, int eq);
    virtual double grad(int node, int eq, int dim);
  protected:
    apf::FieldShape* m_shape = nullptr;
    apf::NewArray<double> m_BF;
    apf::NewArray<apf::Vector3> m_gBF;
};

}
