#pragma once

#include <apf.h>
#include "defines.hpp"

namespace calibr8 {

template <typename T>
class Residual;

class Disc;

template <typename T>
using Array1D = std::vector<T>;

template <typename T>
using Array2D = std::vector<std::vector<T>>;

template <typename T>
using Array3D = std::vector<std::vector<std::vector<T>>>;

class Weight {
  public:
    Weight(apf::FieldShape* shape);
    virtual ~Weight();
    virtual void in_elem(apf::MeshElement* me, RCP<Disc> disc);
    virtual void gather(RCP<Disc>, RCP<VectorT>) {}
    virtual void evaluate(apf::Vector3 const& xi);
    virtual double val(int node, int eq);
    virtual double grad(int node, int eq, int dim);
    virtual void out_elem();
  protected:
    apf::FieldShape* m_shape = nullptr;
    apf::MeshElement* m_mesh_elem = nullptr;
    apf::NewArray<double> m_BF;
    apf::NewArray<apf::Vector3> m_gBF;
};

class AdjointWeight : Weight {
  public:
    AdjointWeight(apf::FieldShape* shape);
    ~AdjointWeight();
    void in_elem(apf::MeshElement* me, RCP<Disc> disc) override;
    void gather(RCP<Disc> disc, RCP<VectorT> Z) override;
    void evaluate(apf::Vector3 const& xi) override;
    double val(int node, int eq) override;
    double grad(int node, int eq, int dim) override;
    void out_elem() override;
  private:
    int m_neqs = -1;
    int m_ndims = -1;
    int m_nnodes = -1;
    apf::MeshElement* m_mesh_element = nullptr;
    Array2D<double> m_vals;
    Array3D<double> m_grads;
};

}
