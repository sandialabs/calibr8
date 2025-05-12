#pragma once

#include "disc.hpp"
#include "linalg.hpp"
#include "weight.hpp"

namespace calibr8 {

enum {RESIDUAL, JACOBIAN, ADJOINT};

int get_index(int node, int eq, int neqs);

template <typename T> double val(T const& in);

template <typename T>
using Array1D = std::vector<T>;

template <typename T>
using Array2D = std::vector<std::vector<T>>;

template <typename T>
using Array3D = std::vector<std::vector<std::vector<T>>>;

template <typename T>
class Residual {
  public:
    Residual(int ndims);
    virtual ~Residual();
    int num_eqs() { return m_neqs; }
    void set_space(int space, RCP<Disc> disc);
    void set_mode(int mode) { m_mode = mode; }
    void set_weight(RCP<Weight> weight) { m_weight = weight; }
    virtual void before_elems(int es_idx, RCP<Disc> disc) {}
    void in_elem(apf::MeshElement* me, RCP<Disc> disc);
    void gather(RCP<Disc> disc, RCP<VectorT> u);
    void interp_basis(apf::Vector3 const& xi, RCP<Disc> disc);
    Array1D<T> interp(apf::Vector3 const& xi);
    Array2D<T> interp_grad(apf::Vector3 const& xi);
    virtual void at_point(
        apf::Vector3 const& xi,
        double w,
        double dv,
        RCP<Disc> disc) = 0;
    void scatter(RCP<Disc> disc, System const& sys);
    void out_elem();
    virtual void destroy_data() {}
    virtual apf::Field* assemble(
        apf::Field* u, apf::Field* z, std::string const& name) = 0;
  protected:
    int m_neqs = -1;
    int m_ndims = -1;
    int m_nnodes = -1;
    int m_ndofs = -1;
    int m_space = -1;
    int m_mode = -1;
    apf::MeshElement* m_mesh_elem = nullptr;
    apf::NewArray<double> m_BF;
    apf::NewArray<apf::Vector3> m_gBF;
    Array2D<T> m_vals;
    Array2D<T> m_resid;
    RCP<Weight> m_weight;
  private:
    void scatter_residual(RCP<Disc> disc, RCP<VectorT> r);
    void scatter_jacobian(RCP<Disc> disc, RCP<MatrixT> J);
    void scatter_adjoint(RCP<Disc> disc, RCP<MatrixT> J);
};

template <typename T>
RCP<Residual<T>> create_residual(ParameterList const& params, int ndims);

}
