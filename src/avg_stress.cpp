#include "avg_stress.hpp"
#include "disc.hpp"
#include "global_residual.hpp"
#include "local_residual.hpp"

namespace calibr8 {

template <typename T>
AvgStress<T>::AvgStress(ParameterList const& params) {
  m_elem_set = params.get<std::string>("elem set");
}

template <typename T>
AvgStress<T>::~AvgStress() {
}

template <typename T>
void AvgStress<T>::before_elems(RCP<Disc> disc, int step) {
  int const nsets = disc->num_elem_sets();
  m_elem_set_names.resize(nsets);
  for (int i = 0; i < nsets; ++i) {
    m_elem_set_names[i] = disc->elem_set_name(i);
  }
}

template <typename T>
void AvgStress<T>::evaluate(
    int elem_set,
    int elem,
    RCP<GlobalResidual<T>> global,
    RCP<LocalResidual<T>> local,
    apf::Vector3 const& iota,
    double w,
    double dv) {

  this->initialize_value_pt();

  // do some stuff if we are in the subdomain of interest
  std::string const es_name = m_elem_set_names[elem_set];
  if (es_name == m_elem_set) {

    // sum von mises stress contributions 
    Tensor<T> const dev_cauchy = local->dev_cauchy(global);
    T const vm_stress = minitensor::norm(dev_cauchy);
    this->value_pt += vm_stress * w * dv;

  }

}

template class AvgStress<double>;
template class AvgStress<FADT>;

}
