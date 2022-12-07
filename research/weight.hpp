#pragma once

#include <apf.h>

namespace calibr8 {

class Weight {
  public:
    Weight(apf::FieldShape* shape);
    virtual void at_point(apf::MeshElement* me, apf::Vector3 const& xi);
    virtual double val(int n, int eq);
    virtual double grad(int n, int eq, int dim);
  protected:
    apf::FieldShape* m_shape = nullptr;
    apf::NewArray<double> m_basis;
    apf::NewArray<apf::Vector3> m_grad_basis;
};

class AdjointWeight : public Weight {
  public:
    AdjointWeight(apf::FieldShape* PU, apf::Field* z);
    void at_point(apf::MeshElement* me, apf::Vector3 const& xi) override;
    double val(int n, int eq) override;
    double grad(int n, int eq, int dim) override;
  private:
    apf::Field* m_z = nullptr;
    std::vector<double> m_vals;
    std::vector<apf::Vector3> m_grads;
};

}
