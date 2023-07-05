#pragma once

#include <MiniTensor.h>

#include "nlelasticity.hpp"
#include "qoi.hpp"

namespace calibr8 {

template <typename T>
class QoI_VM : public QoI<T> {

  private:

    double m_xmin = 0.;
    double m_xmax = 0.;
    double m_ymin = 0.;
    double m_ymax = 0.;
    double m_zmin = 0.;
    double m_zmax = 0.;

  public:

    QoI_VM(ParameterList const& params) : QoI<T> ()
    {
      m_xmin = params.get<double>("xmin");
      m_xmax = params.get<double>("xmax");
      m_ymin = params.get<double>("ymin");
      m_ymax = params.get<double>("ymax");
      m_zmin = params.get<double>("zmin");
      m_zmax = params.get<double>("zmax");
    }

    bool is_inside(apf::Vector3 const& x)
    {
      if ((m_xmin <= x[0]) && (x[0] <= m_xmax) &&
          (m_ymin <= x[1]) && (x[1] <= m_ymax) &&
          (m_zmin <= x[2]) && (x[2] <= m_zmax)) {
        return true;
      } else {
        return false;
      }
    }

    void at_point(
        apf::Vector3 const& xi,
        double w,
        double dv,
        RCP<Residual<T>> residual,
        RCP<Disc> disc) override
    {
      using minitensor::Tensor;
      apf::Vector3 x;
      apf::mapLocalToGlobal(this->m_mesh_elem, xi, x);
      if (!is_inside(x)) return;
      RCP<NLElasticity<T>> R = Teuchos::rcp_dynamic_cast<NLElasticity<T>>(residual);
      R->interp_basis(xi, disc);
      R->interp_grad(xi);
      Tensor<T> const F = R->compute_def_grad(xi, disc);
      Tensor<T> const sigma = R->compute_sigma(F);
      Tensor<T> const dev = minitensor::dev(sigma);
      T const sqrt32 = std::sqrt(3./2.);
      T const dev_norm = minitensor::norm(dev);
      T const vm_stress = sqrt32 * dev_norm;
      this->m_elem_value += vm_stress * w * dv;
    }

};

}
