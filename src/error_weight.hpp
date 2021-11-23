#pragma once

#include <weight.hpp>

namespace calibr8 {

class ErrorWeight : public Weight {
  public:
    ErrorWeight(apf::FieldShape* shape);
    void evaluate(apf::MeshElement* me, apf::Vector3 const& iota);
    double val(int i, int n, int eq);
    double grad(int i, int n, int eq, int dim);
  protected:
};

}
