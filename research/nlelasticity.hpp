#pragma once

#include <apf.h>

#include <MiniTensor.h>

#include "control.hpp"
#include "residual.hpp"

namespace calibr8 {

using minitensor::Tensor;

template <typename T>
class NLElasticity : public Residual<T> {

  public:

    NLElasticity(ParameterList const& params, int ndims) : Residual<T>(ndims) {
      m_params = params;
      this->m_neqs = ndims;
    }

    ~NLElasticity() override {
    }

    void before_elems(int es_idx, RCP<Disc> disc) override {
      std::string const es_name = disc->elem_set_name(es_idx);
      ParameterList const& mat = this->m_params.sublist(es_name);
      double const E = mat.get<double>("E");
      double const nu = mat.get<double>("nu");
      this->m_kappa = E/(3.*(1.-2.*nu));
      this->m_mu = E/(2.*(1.+nu));
      if constexpr (std::is_same_v<T, double>) {
        if (!m_sigma_field) {
          m_sigma_field = apf::createIPField(disc->apf_mesh(), "sigma", apf::MATRIX, 1);
          apf::zeroField(m_sigma_field);
        }
      }
    }

    Tensor<T> compute_sigma(
        Tensor<T> const& F,
        T const& J)
    {
      T const mu = this->m_mu;
      T const kappa = this->m_kappa;
      T const Jm13 = 1./std::cbrt(J);
      T const Jm23 = Jm13 * Jm13;
      T const Jm53 = Jm23 * Jm13 * Jm13;
      T const p = 0.5 * kappa * (J - 1./J);
      Tensor<T> const I = minitensor::eye<T>(this->m_ndims);
      Tensor<T> const b = F * minitensor::transpose(F);
      Tensor<T> const sigma = mu * Jm53 * minitensor::dev(b) + p * I;
      return sigma;
    }

    void at_point(
        apf::Vector3 const& xi,
        double w,
        double dv,
        RCP<Disc> disc) override {

      double const wdetJ = w*dv;

      // compute the kinematic quantities
      this->interp_basis(xi, disc);
      Array1D<T> const dofs = this->interp(xi);
      Array2D<T> const grad_dofs = this->interp_grad(xi);
      Tensor<T> grad_u(this->m_ndims);
      for (int i = 0; i < this->m_ndims; ++i) {
        for (int j = 0; j < this->m_ndims; ++j) {
          grad_u(i, j) = grad_dofs[i][j];
        }
      }
      Tensor<T> const I = minitensor::eye<T>(this->m_ndims);
      Tensor<T> const F = grad_u + I;
      Tensor<T> const Finv = minitensor::inverse(F);
      Tensor<T> const FinvT = minitensor::transpose(Finv);
      T const J = minitensor::det(F);

      // compute the stress divergence residual
      Tensor<T> const sigma = compute_sigma(F, J);
      Tensor<T> const P = J * sigma * FinvT;
      for (int node = 0; node < this->m_nnodes; ++node) {
        for (int i = 0; i < this->m_ndims; ++i) {
          for (int j = 0; j < this->m_ndims; ++j) {
            double const grad_w = this->m_gBF[node][j];
            this->m_resid[node][i] += P(i,j) * grad_w * wdetJ;
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

    apf::Field* assemble(
        apf::Field* u_field,
        apf::Field* z_field,
        std::string const& name) override {
      // debug
      apf::FieldShape* PU = apf::getLagrange(1);
      apf::Mesh* mesh = apf::getMesh(u_field);
      apf::Field* eta = apf::createPackedField(mesh, name.c_str(), this->m_neqs, PU);
      apf::zeroField(eta);
      return eta;
    }

    void destroy_data() override {
      if (m_sigma_field) {
        apf::destroyField(m_sigma_field);
        m_sigma_field = nullptr;
      }
    }

  private:

    double m_kappa = 0.;
    double m_mu = 0.;

    ParameterList m_params;
    apf::Field* m_sigma_field = nullptr;

};

}
