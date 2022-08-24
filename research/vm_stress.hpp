#include <MiniTensor.h>
#include "elastic.hpp"

namespace calibr8 {

template <typename T>
T compute_von_mises_stress(minitensor::Tensor<T> const& sigma) {
  T s1 = (sigma(0,0)-sigma(1,1))*(sigma(0,0)-sigma(1,1));
  T s2 = (sigma(1,1)-sigma(2,2))*(sigma(1,1)-sigma(2,2));
  T s3 = (sigma(2,2)-sigma(0,0))*(sigma(2,2)-sigma(0,0));
  T s4 = sigma(0,1)*sigma(0,1);
  T s5 = sigma(1,2)*sigma(1,2);
  T s6 = sigma(2,0)*sigma(2,0);
  T s7 = 0.5*(s1+s2+s3+6.0*(s4+s5+s6));
  return std::sqrt(s7);
}

template <typename T>
class VMStress : public QoI<T> {

  public:

    VMStress(ParameterList const& params) : QoI<T>() {
      m_elem_set = params.get<std::string>("elem set");
    }

    ~VMStress() override {
    }

    void before_elems(int es_idx, RCP<Disc> disc) override {
      std::string const current_elem_set = disc->elem_set_name(es_idx);
      if (current_elem_set == m_elem_set) m_evaluate = true;
      else m_evaluate = false;
    }

    void at_point(
        apf::Vector3 const& xi,
        double w,
        double dv,
        RCP<Residual<T>> residual,
        RCP<Disc> disc) override {
      if (!m_evaluate) return;
      using minitensor::Tensor; 
      double const wdetJ = w*dv;
      residual->interp_basis(xi, disc);
      auto elastic = Teuchos::rcp_dynamic_cast<Elastic<T>>(residual);
      Tensor<T> const sigma = elastic->compute_sigma(xi, disc);
      T const vm = compute_von_mises_stress(sigma); 
      this->m_elem_value += vm * w * dv;
    }

  private:

    std::string m_elem_set = "";
    bool m_evaluate = false;

};

}
