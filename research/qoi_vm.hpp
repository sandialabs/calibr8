#pragma once

#include <MiniTensor.h>

#include "nlelasticity.hpp"
#include "qoi.hpp"

namespace calibr8 {

template <typename T>
class QoI_VM : public QoI<T> {

  private:

    std::string m_elem_set = "";
    bool in_set = false;

  public:

    QoI_VM(ParameterList const& params) : QoI<T> () {
      m_elem_set = params.get<std::string>("elem set");
    }

    virtual void before_elems(int es_idx, RCP<Disc> disc) override {
      in_set = false;
      std::string const name = disc->elem_set_name(es_idx);
      if (name == m_elem_set) in_set = true;
    }

    void at_point(
        apf::Vector3 const& xi,
        double w,
        double dv,
        RCP<Residual<T>> residual,
        RCP<Disc> disc) override {
      using minitensor::Tensor;
      if (!in_set) return;
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
