#pragma once

#include <weight.hpp>
#include "arrays.hpp"

namespace calibr8 {

class ErrorWeight : public Weight {
  public:
    ErrorWeight(
        apf::FieldShape* shape,
        int dim,
        int nresids,
        int nnodes,
        Array1D<int> const& neqs,
        Array1D<apf::Field*> const& z);
    void evaluate(apf::MeshElement* me, apf::Vector3 const& iota);
    double val(int i, int n, int eq);
    double grad(int i, int n, int eq, int dim);
  protected:
    int m_ndim;
    int m_nresids;
    int m_nnodes;
    Array1D<int> m_neqs;
    Array1D<apf::Field*> m_z;
    Array3D<double> values; // values(resid_idx, node, eq)
    Array4D<double> gradients; // values(resid_idx, node, eq, dim)
};

}
