#pragma once

#include "arrays.hpp"
#include "qoi.hpp"

namespace calibr8 {

template <typename T>
class PointWise : public QoI<T> {
  public:
    PointWise(ParameterList const& params);
    ~PointWise();
    void before_elems(RCP<Disc> disc, int step);
    void evaluate(
        int,
        int,
        RCP<GlobalResidual<T>>,
        RCP<LocalResidual<T>>,
        apf::Vector3 const&,
        double,
        double);
    void postprocess(double& J);
    void modify_state(RCP<State> state);
  private:
    int m_component = -1;
    int m_step = -1;
    int m_current_step = -1;
    std::string m_node_set;
    RCP<Disc> m_disc;
};

}
