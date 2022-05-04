#include "disp_comp.hpp"
#include "disc.hpp"
#include "global_residual.hpp"
#include "local_residual.hpp"

namespace calibr8 {

template <typename T>
DispComp<T>::DispComp(ParameterList const& params) {
  m_elem_set = params.get<std::string>("elem set");
  m_component = params.get<int>("component");

  std::cout << "m_elem_set: !!!! " << m_elem_set << "\n";
  std::cout << "component: !!!! " << m_component << "\n";
}

template <typename T>
DispComp<T>::~DispComp() {
}

template <typename T>
void DispComp<T>::before_elems(RCP<Disc> disc, int step) {
  this->m_num_dims = disc->num_dims();
  int const nsets = disc->num_elem_sets();
  m_elem_set_names.resize(nsets);
  for (int i = 0; i < nsets; ++i) {
    m_elem_set_names[i] = disc->elem_set_name(i);
  }
}

template <typename T>
void DispComp<T>::evaluate(
    int elem_set,
    int,
    RCP<GlobalResidual<T>> global,
    RCP<LocalResidual<T>> local,
    apf::Vector3 const&,
    double w,
    double dv) {

  this->initialize_value_pt();

  // only do some stuff over the subdomain of interest
  std::string const es_name = m_elem_set_names[elem_set];
  if (es_name == m_elem_set) {
    static constexpr int disp_idx = 0;
    Vector<T> const u = global->vector_x(disp_idx);
    this->value_pt = u[m_component] * w * dv;
  }

}

template class DispComp<double>;
template class DispComp<FADT>;

}
