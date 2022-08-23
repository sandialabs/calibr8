#pragma once

#include <minitensor.h>
#include "residual.hpp"

// references for linear elasticity w/ thermal expansion:
// * http://mmc.rmee.upc.edu/documents/Slides/Ch6_v17.pdf, slides 72+73
// * http://solidmechanics.org/Text/Chapter3_2/Chapter3_2.php

namespace calibr8 {

template <typename T>
class Elastic : public Residual<T> {

  public:

    Elastic(ParameterList const& params, int ndims) : Residual<T>(ndims) {
      this->m_neqs = ndims;
      this->m_E = params.get<double>("E");
      this->m_nu = params.get<double>("nu");
      this->m_cte = params.get<double>("cte");
      this->m_dT = params.get<double>("dT");
    }

    ~Elastic() override {
    }

    void at_point(
        apf::Vector3 const& xi,
        double w,
        double dv,
        RCP<Disc> disc) override {

      using minitensor::Tensor;

      double const wdetJ = w*dv;
      double const E = this->m_E;
      double const nu = this->m_nu;
      double const alpha = this->m_cte;
      double const dT = this->m_dT;
      double const lambda = (E*nu)/((1.+nu)*(1.-2.*nu));
      double const mu = E/(2.*(1.+nu));
      double const beta = (E/(1.-2.*nu))*alpha;

      this->interp_basis(xi, disc);
      Array1D<T> const dofs = this->interp(xi);
      Array2D<T> const grad_dofs = this->interp_grad(xi);

      Tensor<T> grad_u(this->m_ndims);
      for (int i = 0; i < this->m_ndims; ++i) {
        for (int j = 0; j < this->m_ndims; ++j) {
          grad_u(i,j) = grad_dofs[i][j];
        }
      }

      Tensor<T> const I = minitensor::eye<T>(this->m_ndims);
      Tensor<T> const grad_uT = minitensor::transpose(grad_u);
      Tensor<T> const eps = 0.5*(grad_u + grad_uT);
      T const tr_eps = minitensor::trace(eps);
      Tensor<T> const sigma = lambda*tr_eps*I + 2.*mu*eps - beta*dT*I;

      for (int node = 0; node < this->m_nnodes; ++node) {
        for (int i = 0; i < this->m_ndims; ++i) {
          for (int j = 0; j < this->m_ndims; ++j) {
            double const grad_w = this->m_gBF[node][j];
            this->m_resid[node][i] += sigma(i,j) * grad_w * wdetJ;
          }
        }
      }

    }

  private:

    double m_E = 0.;
    double m_nu = 0.;
    double m_cte = 0.;
    double m_dT = 0.;

};

}
