#include "avg_local_var.hpp"
#include "disc.hpp"
#include "global_residual.hpp"
#include "local_residual.hpp"

namespace calibr8 {

template <typename T>
AvgLocalVar<T>::AvgLocalVar(ParameterList const& params) {
  m_resid_idx = params.get<int>("residual");
  m_elem_set = params.get<std::string>("elem set");
}

template <typename T>
AvgLocalVar<T>::~AvgLocalVar() {
}

template <typename T>
void AvgLocalVar<T>::before_elems(RCP<Disc> disc, int step) {
  int const nsets = disc->num_elem_sets();
  m_elem_set_names.resize(nsets);
  for (int i = 0; i < nsets; ++i) {
    m_elem_set_names[i] = disc->elem_set_name(i);
  }
}

template <typename T>
void AvgLocalVar<T>::evaluate(
    int elem_set,
    int elem,
    RCP<GlobalResidual<T>> global,
    RCP<LocalResidual<T>> local,
    apf::Vector3 const& iota,
    double w,
    double dv) {
  this->initialize_value_pt();
  std::string const es_name = m_elem_set_names[elem_set];
  if (es_name == m_elem_set) {
    T const xi = local->scalar_xi(m_resid_idx);
    this->value_pt += xi * w * dv;
  }
}

template class AvgLocalVar<double>;
template class AvgLocalVar<FADT>;

}
