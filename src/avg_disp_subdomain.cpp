#include "avg_disp_subdomain.hpp"
#include "disc.hpp"
#include "global_residual.hpp"
#include "local_residual.hpp"

namespace calibr8 {

template <typename T>
AvgDispSubdomain<T>::AvgDispSubdomain(ParameterList const& params) {
  m_elem_set = params.get<std::string>("elem set");
}

template <typename T>
AvgDispSubdomain<T>::~AvgDispSubdomain() {
}


template <typename T>
void AvgDispSubdomain<T>::before_elems(RCP<Disc> disc, int step) {
  int const nsets = disc->num_elem_sets();
  m_elem_set_names.resize(nsets);
  for (int i = 0; i < nsets; ++i) {
    m_elem_set_names[i] = disc->elem_set_name(i);
  }
}

template <typename T>
void AvgDispSubdomain<T>::evaluate(
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
    for (int i = 0; i < this->m_num_dims; ++i) {
      this->value_pt += u[i] * w * dv;
    }
    this->value_pt /= this->m_num_dims;
  }

}

template class AvgDispSubdomain<double>;
template class AvgDispSubdomain<FADT>;

}
