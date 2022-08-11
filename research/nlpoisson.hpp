#pragma once

#include "control.hpp"
#include "residual.hpp"

namespace calibr8 {

template <typename T>
class NLPoisson : public Residual<T> {

  public:

    NLPoisson(ParameterList const& params, int ndims) : Residual<T>(ndims) {
      this->m_neqs = 1;
      m_alpha = params.get<double>("alpha");
      m_body_force = params.get<std::string>("body force");
    }

    ~NLPoisson() override {
    }

    void at_point(
        apf::Vector3 const& xi,
        double w,
        double dv,
        RCP<Weight> weight,
        RCP<Disc> disc) override {

      int const eq = 0;
      double const wdetJ = w*dv;

      apf::Vector3 x;
      apf::mapLocalToGlobal(this->m_mesh_elem, xi, x);
      double const b = eval(m_body_force, x[0], x[1], x[2], 0.);

      weight->evaluate(xi);
      Array1D<T> const vals = this->interp(xi, disc);
      Array2D<T> const dvals = this->interp_grad(xi, disc);

      for (int node = 0; node < this->m_nnodes; ++node) {
        for (int dim = 0; dim < this->m_ndims; ++dim) {
          T const u = vals[eq];
          T const grad_u = dvals[eq][dim];
          double const w = weight->val(node, eq);
          double const grad_w = weight->grad(node, eq, dim);
          this->m_resid[node][eq] +=
            ((1.0 + m_alpha*u*u)*grad_u*grad_w - b*w)*wdetJ;
        }
      }

    }

  private:

    double m_alpha = 0.;
    std::string m_body_force = "";

};

}
