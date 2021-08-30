#include "avg_disp.hpp"
#include "global_residual.hpp"
#include "local_residual.hpp"

namespace calibr8 {

template <typename T>
AvgDisp<T>::AvgDisp() {
}

template <typename T>
AvgDisp<T>::~AvgDisp() {
}

template <typename T>
void AvgDisp<T>::evaluate(
    RCP<GlobalResidual<T>> global,
    RCP<LocalResidual<T>> local,
    apf::Vector3 const&,
    double w,
    double dv) {
  static constexpr int disp_idx = 0;
  this->value_pt = T(0.);
  Vector<T> const u = global->vector_x(disp_idx);
  T const first_value = local->first_value();
  for (int i = 0; i < this->m_num_dims; ++i) {
    this->value_pt += u[i] * w * dv + 0. * first_value;
  }
  this->value_pt /= this->m_num_dims;
}

template class AvgDisp<double>;
template class AvgDisp<FADT>;

}
