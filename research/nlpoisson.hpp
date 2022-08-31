#pragma once

#include "control.hpp"
#include "residual.hpp"

namespace calibr8 {

enum {EXPR, SIN, SIN_EXP};

double eval_sin_body_force(apf::Vector3 const& x, double alpha);
double eval_sin_exp_body_force(apf::Vector3 const& x, double alpha);

template <typename T>
class NLPoisson : public Residual<T> {

  public:

    NLPoisson(ParameterList const& params, int ndims) : Residual<T>(ndims) {
      this->m_neqs = 1;
      m_alpha = params.get<double>("alpha");
      m_body_force = params.get<std::string>("body force");
      if (m_body_force == "sin") m_body_force_type = SIN;
      else if (m_body_force == "sin_exp") m_body_force_type = SIN_EXP;
      else m_body_force_type = EXPR;
    }

    ~NLPoisson() override {
    }

    void at_point(
        apf::Vector3 const& xi,
        double w,
        double dv,
        RCP<Disc> disc) override {

      int const eq = 0;
      double const wdetJ = w*dv;

      apf::Vector3 x;
      apf::mapLocalToGlobal(this->m_mesh_elem, xi, x);
      double b = 0.;
      if (m_body_force_type == EXPR) {
        b = eval(m_body_force, x[0], x[1], x[2], 0.);
      } else if (m_body_force_type == SIN) {
        b = eval_sin_body_force(x, m_alpha);
      } else if (m_body_force_type == SIN_EXP) {
        b = eval_sin_exp_body_force(x, m_alpha);
      }

      this->interp_basis(xi, disc);
      Array1D<T> const vals = this->interp(xi);
      Array2D<T> const dvals = this->interp_grad(xi);

      for (int node = 0; node < this->m_nnodes; ++node) {
        for (int dim = 0; dim < this->m_ndims; ++dim) {
          T const u = vals[eq];
          T const grad_u = dvals[eq][dim];
          double const grad_w = this->m_gBF[node][dim];
          this->m_resid[node][eq] += (1.0 + m_alpha*u*u)*grad_u*grad_w*wdetJ;
        }
        double const w = this->m_BF[node];
        this->m_resid[node][eq] += b*w*wdetJ;
      }

    }

  private:

    double m_alpha = 0.;
    int m_body_force_type = -1;
    std::string m_body_force = "";

};

}
