#pragma once

#include "disc.hpp"
#include "linalg.hpp"
#include "weights.hpp"

namespace calibr8 {

enum {RESIDUAL, JACOBIAN, ADJOINT};

int get_index(int node, int eq, int neqs);

template <typename T> double val(T const& in);

template <typename T>
class Residual {
  public:
    Residual(int ndims);
    virtual ~Residual();
    int num_eqs() { return m_neqs; }
    void set_space(int space) { m_space = space; }
    void set_mode(int mode) { m_mode = mode; }
    void in_elem(apf::MeshElement* me, RCP<Disc> disc);
    void gather(RCP<Disc> disc, RCP<VectorT> u);
    Array1D<T> interp(apf::Vector3 const& xi, RCP<Disc> disc);
    Array2D<T> interp_grad(apf::Vector3 const& xi, RCP<Disc> disc);
    virtual void at_point(
        apf::Vector3 const& xi,
        double w,
        double dv,
        RCP<Weight> weight,
        RCP<Disc> disc) = 0;
    void scatter(RCP<Disc> disc, System const& sys);
    void out_elem();
  protected:
    int m_neqs = -1;
    int m_ndims = -1;
    int m_nnodes = -1;
    int m_ndofs = -1;
    int m_space = -1;
    int m_mode = -1;
    apf::MeshElement* m_mesh_elem = nullptr;
    Array2D<T> m_vals;
    Array2D<T> m_resid;
  private:
    void scatter_residual(RCP<Disc> disc, RCP<VectorT> r);
    void scatter_jacobian(RCP<Disc> disc, RCP<MatrixT> J);
    void scatter_adjoint(RCP<Disc> disc, RCP<MatrixT> J);
};

template <typename T>
RCP<Residual<T>> create_residual(ParameterList const& params, int ndims);

}
