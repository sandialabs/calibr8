#pragma once

#include "arrays.hpp"
#include "qoi.hpp"

namespace calibr8 {

template <typename T>
class AvgElementStress : public QoI<T> {
  public:
    AvgElementStress(ParameterList const& params);
    ~AvgElementStress();
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
    std::string m_elem_set = "";
    int m_elem_set_idx = -1;
    Array1D<int> m_stress_idx {-1, 1};
    bool m_transform_to_polar = false;
    int m_stress_i_idx = -1;
    int m_stress_j_idx = -1;
    double m_volume = 0.;
};

}
