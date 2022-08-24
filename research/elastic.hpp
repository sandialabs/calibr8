#pragma once

#include "MiniTensor.h"
#include "residual.hpp"

// references for linear elasticity w/ thermal expansion:
// * http://mmc.rmee.upc.edu/documents/Slides/Ch6_v17.pdf, slides 72+73
// * http://solidmechanics.org/Text/Chapter3_2/Chapter3_2.php

namespace calibr8 {

template <typename T>
class Elastic : public Residual<T> {

  public:

    Elastic(ParameterList const& params, int ndims) : Residual<T>(ndims) {
      m_params = params;
      this->m_neqs = ndims;
    }

    ~Elastic() override {
    }

    void before_elems(int es_idx, RCP<Disc> disc) override {
      std::string const es_name = disc->elem_set_name(es_idx);
      ParameterList const& mat = this->m_params.sublist(es_name);
      double const E = mat.get<double>("E");
      double const nu = mat.get<double>("nu");
      double const alpha = mat.get<double>("cte");
      this->m_lambda = (E*nu)/((1.+nu)*(1.-2.*nu));
      this->m_mu = E/(2.*(1.+nu));
      this->m_beta = (E/(1.-2.*nu))*alpha;
      this->m_dT = mat.get<double>("dT");
      if constexpr (std::is_same_v<T, double>) {
        if (!m_sigma_field) {
          m_sigma_field = apf::createIPField(disc->apf_mesh(), "sigma", apf::MATRIX, 1);
        }
      }
    }

    minitensor::Tensor<T> compute_sigma(apf::Vector3 const& xi, RCP<Disc> disc) {
      using minitensor::Tensor;
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
      Tensor<T> const sigma = m_lambda*tr_eps*I + 2.*m_mu*eps - m_beta*m_dT*I;
      return sigma;
    }

    void at_point(
        apf::Vector3 const& xi,
        double w,
        double dv,
        RCP<Disc> disc) override {

      using minitensor::Tensor;
      double const wdetJ = w*dv;

      // compute the stress divergence residual
      Tensor<T> sigma = compute_sigma(xi, disc);
      for (int node = 0; node < this->m_nnodes; ++node) {
        for (int i = 0; i < this->m_ndims; ++i) {
          for (int j = 0; j < this->m_ndims; ++j) {
            double const grad_w = this->m_gBF[node][j];
            this->m_resid[node][i] += sigma(i,j) * grad_w * wdetJ;
          }
        }
      }

      // save the stress field if this is the coarse solve
      if constexpr (std::is_same_v<T, double>) {
        if (this->m_space == COARSE) {
          apf::MeshEntity* ent = apf::getMeshEntity(this->m_mesh_elem);
          apf::Matrix3x3 apf_sigma(0,0,0,0,0,0,0,0,0);
          for (int i = 0; i < this->m_ndims; ++i) {
            for (int j = 0; j < this->m_ndims; ++j) {
              apf_sigma[i][j] = val(sigma(i,j));
            }
          }
          apf::setMatrix(this->m_sigma_field, ent, 0, apf_sigma);
        }
      }

    }

  private:

    double m_lambda = 0.;
    double m_mu = 0.;
    double m_beta = 0.;
    double m_dT = 0.;

    ParameterList m_params;
    apf::Field* m_sigma_field = nullptr;

};

}
