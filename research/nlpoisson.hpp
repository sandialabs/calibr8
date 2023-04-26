#pragma once

#include <apf.h>

#include "control.hpp"
#include "residual.hpp"

namespace calibr8 {

enum {EXPR, MANUFACTURED};

double eval_manufactured_force(apf::Vector3 const& x, double alpha);

template <typename T>
class NLPoisson : public Residual<T> {

  public:

    NLPoisson(ParameterList const& params, int ndims) : Residual<T>(ndims) {
      this->m_neqs = 1;
      m_alpha = params.get<double>("alpha");
      m_body_force = params.get<std::string>("body force");
      if (m_body_force == "manufactured") m_body_force_type = MANUFACTURED;
      else m_body_force_type = EXPR;
    }

    ~NLPoisson() override {
    }

    double eval_body_force(
        apf::MeshElement* mesh_elem,
        apf::Vector3 const& xi) {
      apf::Vector3 x;
      apf::mapLocalToGlobal(mesh_elem, xi, x);
      double b = 0.;
      if (m_body_force_type == EXPR) {
        b = eval(m_body_force, x[0], x[1], x[2], 0.);
      } else if (m_body_force_type == MANUFACTURED) {
        b = eval_manufactured_force(x, m_alpha);
      }
      return b;
    }

    void at_point(
        apf::Vector3 const& xi,
        double w,
        double dv,
        RCP<Disc> disc) override {

      int const eq = 0;
      double const wdetJ = w*dv;
      double const b = eval_body_force(this->m_mesh_elem, xi);

      this->interp_basis(xi, disc);
      Array1D<T> const vals = this->interp(xi);
      Array2D<T> const dvals = this->interp_grad(xi);

      for (int node = 0; node < this->m_nnodes; ++node) {
        for (int dim = 0; dim < this->m_ndims; ++dim) {
          T const u = vals[eq];
          T const grad_u = dvals[eq][dim];
          double const grad_w = this->m_weight->grad(node, eq, dim);
          this->m_resid[node][eq] += (1.0 + m_alpha*u*u)*grad_u*grad_w*wdetJ;
        }
      }

      for (int node = 0; node < this->m_nnodes; ++node) {
        double const w = this->m_weight->val(node, eq);
        this->m_resid[node][eq] -= b*w*wdetJ;
      }

    }

    apf::Field* assemble(
        apf::Field* u_field,
        apf::Field* z_field,
        std::string const& name) override {
      apf::Mesh* mesh = apf::getMesh(u_field);
      apf::Field* eta = apf::createStepField(mesh, name.c_str(), apf::SCALAR);
      apf::zeroField(eta);
      int const ndims = mesh->getDimension();
      int q_order = 6;
      auto elems = mesh->begin(ndims);
      apf::Vector3 grad_u;
      apf::Vector3 grad_z;
      apf::MeshEntity* elem;
      while ((elem = mesh->iterate(elems))) {
        apf::MeshElement* mesh_elem = apf::createMeshElement(mesh, elem);
        apf::Element* u_elem = apf::createElement(u_field, mesh_elem);
        apf::Element* z_elem = apf::createElement(z_field, mesh_elem);
        int const npts = apf::countIntPoints(mesh_elem, q_order);
        for (int pt = 0; pt < npts; ++pt) {
          apf::Vector3 xi;
          apf::getIntPoint(mesh_elem, q_order, pt, xi);
          double const w = apf::getIntWeight(mesh_elem, q_order, pt);
          double const dv = apf::getDV(mesh_elem, xi);
          double const u = apf::getScalar(u_elem, xi);
          double const z = apf::getScalar(z_elem, xi);
          double const b = eval_body_force(mesh_elem, xi);
          apf::getGrad(u_elem, xi, grad_u);
          apf::getGrad(z_elem, xi, grad_z);
          double assembled = apf::getScalar(eta, elem, 0);
          assembled += b*z*w*dv;
          for (int dim = 0; dim < ndims; ++dim) {
            assembled -= (1. + m_alpha*u*u)*grad_u[dim]*grad_z[dim]*w*dv;
          }
          apf::setScalar(eta, elem, 0, assembled);
        }
        apf::destroyElement(z_elem);
        apf::destroyElement(u_elem);
        apf::destroyMeshElement(mesh_elem);
      }
      mesh->end(elems);
      return eta;
    }

  private:

    double m_alpha = 0.;
    int m_body_force_type = -1;
    std::string m_body_force = "";

};

}
