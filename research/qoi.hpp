#pragma once

#include "disc.hpp"
#include "linalg.hpp"
#include "residual.hpp"

namespace calibr8 {

template <typename T>
class QoI {
  public:
    QoI();
    virtual ~QoI();
    void set_space(int space) { m_space = space; }
    void reset();
    void in_elem(
        apf::MeshElement* me,
        RCP<Residual<T>> residual,
        RCP<Disc> disc);
    virtual void at_point(
        apf::Vector3 const& xi,
        double w,
        double dv,
        RCP<Residual<T>> residual,
        RCP<Disc> disc) = 0;
    void scatter(RCP<Disc> disc, System* sys);
    void out_elem();
    virtual void post(RCP<Disc> disc, System* sys) {}
    double value();
  protected:
    int m_neqs = -1;
    int m_nnodes = -1;
    int m_space = -1;
    apf::MeshElement* m_mesh_elem = nullptr;
    T m_value;
    T m_elem_value;
};

template <typename T>
RCP<QoI<T>> create_QoI(ParameterList const& params);

}
