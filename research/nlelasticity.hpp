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
      this->m_lambda = (E*nu)/((1.+nu)*(1.-2.*nu));
      if constexpr (std::is_same_v<T, double>) {
        if (!m_sigma_field) {
          m_sigma_field = apf::createIPField(disc->apf_mesh(), "sigma", apf::MATRIX, 1);
          apf::zeroField(m_sigma_field);
        }
      }
    }

    Tensor<T> compute_def_grad(
        apf::Vector3 const& xi,
        RCP<Disc> disc) {
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
      return F;
    }

    Tensor<double> compute_def_grad(apf::Matrix3x3 const& apf_grad_u) {
      Tensor<double> grad_u(this->m_ndims);
      for (int i = 0; i < this->m_ndims; ++i) {
        for (int j = 0; j < this->m_ndims; ++j) {
          grad_u(i, j) = apf_grad_u[i][j];
        }
      }
      Tensor<double> const I = minitensor::eye<double>(this->m_ndims);
      Tensor<double> const F = grad_u + I;
      return F;
    }

    template <class ScalarT>
    Tensor<ScalarT> compute_sigma(Tensor<ScalarT> const& F) {
      ScalarT const J = minitensor::det(F);
      ScalarT const mu = this->m_mu;
      ScalarT const kappa = this->m_kappa;
      ScalarT const Jm13 = 1./std::cbrt(J);
      ScalarT const Jm23 = Jm13 * Jm13;
      ScalarT const Jm53 = Jm23 * Jm13 * Jm13;
      ScalarT const p = 0.5 * kappa * (J - 1./J);
      Tensor<ScalarT> const I = minitensor::eye<ScalarT>(this->m_ndims);
      Tensor<ScalarT> const b = F * minitensor::transpose(F);
      Tensor<ScalarT> const sigma = mu * Jm53 * minitensor::dev(b) + p * I;
      return sigma;
    }

    template <class ScalarT>
    Tensor<ScalarT> compute_sigma_elastic(apf::Vector3 const& xi, RCP<Disc> disc) {
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
      Tensor<T> const sigma = m_lambda*tr_eps*I + 2.*m_mu*eps;
      return sigma;
    }

    void at_point(
        apf::Vector3 const& xi,
        double w,
        double dv,
        RCP<Disc> disc) override {

      double const wdetJ = w*dv;

      // compute the kinematic quantities
      Tensor<T> const F = compute_def_grad(xi, disc);
      Tensor<T> const Finv = minitensor::inverse(F);
      Tensor<T> const FinvT = minitensor::transpose(Finv);
      T const J = minitensor::det(F);

      // compute the stress divergence residual
      Tensor<T> const sigma = compute_sigma<T>(F);
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
      apf::Mesh* mesh = apf::getMesh(u_field);
      apf::FieldShape* PU = apf::getLagrange(1);
      apf::Field* eta = apf::createPackedField(mesh, name.c_str(), this->m_neqs, PU);
      apf::zeroField(eta);
      int const ndims = mesh->getDimension();
      int const q_order = 6;
      auto elems = mesh->begin(ndims);
      apf::Vector3 assembled;
      apf::Vector3 z;
      apf::Matrix3x3 grad_u;
      apf::Matrix3x3 grad_z;
      apf::Matrix3x3 grad_zT;
      apf::Matrix3x3 grad_uT;
      apf::NewArray<double> psi;
      apf::NewArray<apf::Vector3> grad_psi;
      apf::Downward verts;
      apf::MeshEntity* elem;
      while ((elem = mesh->iterate(elems))) {
        apf::MeshElement* mesh_elem = apf::createMeshElement(mesh, elem);
        apf::Element* u_elem = apf::createElement(u_field, mesh_elem);
        apf::Element* z_elem = apf::createElement(z_field, mesh_elem);
        int const npts = apf::countIntPoints(mesh_elem, q_order);
        int const entity_type = mesh->getType(elem);
        int const nnodes = PU->getEntityShape(entity_type)->countNodes();
        for (int pt = 0; pt < npts; ++pt) {
          apf::Vector3 xi;
          apf::getIntPoint(mesh_elem, q_order, pt, xi);
          double const w = apf::getIntWeight(mesh_elem, q_order, pt);
          double const dv = apf::getDV(mesh_elem, xi);
          apf::getVectorGrad(u_elem, xi, grad_uT);
          apf::getVectorGrad(z_elem, xi, grad_zT);
          apf::getVector(z_elem, xi, z);
          grad_z = apf::transpose(grad_zT);
          grad_u = apf::transpose(grad_uT);
          apf::getBF(PU, mesh_elem, xi, psi);
          apf::getGradBF(PU, mesh_elem, xi, grad_psi);
          mesh->getDownward(elem, 0, verts);
          Tensor<double> const F = compute_def_grad(grad_u);
          Tensor<double> const Finv = minitensor::inverse(F);
          Tensor<double> const FinvT = minitensor::transpose(Finv);
          double const J = minitensor::det(F);
          Tensor<double> const sigma = compute_sigma(F);
          Tensor<double> const P = J * sigma * FinvT;
          for (int n = 0; n < nnodes; ++n) {
            apf::MeshEntity* vert = verts[n];
            assembled = apf::Vector3(0,0,0);
            apf::getVector(eta, vert, 0, assembled);
            for (int i = 0; i < ndims; ++i) {
              for (int j = 0; j < ndims; ++j) {
                assembled[i] += P(i, j) * (grad_z[i][j] * psi[n] + z[i] * grad_psi[n][j]) * w * dv;
              }
            }
            apf::setVector(eta, vert, 0, assembled);
          }
        }
        apf::destroyElement(z_elem);
        apf::destroyElement(u_elem);
        apf::destroyMeshElement(mesh_elem);
      }
      return eta;
    }

    void destroy_data() override {
      if (m_sigma_field) {
        apf::destroyField(m_sigma_field);
        m_sigma_field = nullptr;
      }
    }

  private:

    double m_lambda = 0.;
    double m_kappa = 0.;
    double m_mu = 0.;

    ParameterList m_params;
    apf::Field* m_sigma_field = nullptr;

};

}
