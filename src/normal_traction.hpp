#pragma once

#include "arrays.hpp"
#include "qoi.hpp"

namespace calibr8 {

template <typename T>
class NormalTraction : public QoI<T> {
  public:
    NormalTraction(ParameterList const& params);
    ~NormalTraction();
    void before_elems(RCP<Disc> disc, int step);
    void evaluate(
        int elem_set,
        int elem,
        RCP<GlobalResidual<T>> global,
        RCP<LocalResidual<T>> local,
        apf::Vector3 const&,
        double,
        double);
  private:
    std::string m_side_set = "";
    std::string m_elem_set = "";
    int m_elem_set_idx = -1;
    Array2D<int> m_mapping; // m_mapping[es_idx][elem_idx]
};

}
