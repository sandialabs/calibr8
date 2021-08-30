#pragma once

//! \file avg_disp.hpp
//! \brief The interface for average displacement QoIs

#include "qoi.hpp"

namespace calibr8 {

//! \brief The evalaution of an average displacement QoI
template <typename T>
class AvgDisp : public QoI<T> {

  public:

    //! \brief The average displacement constructor
    AvgDisp();

    //! \brief The average displacement destructor
    ~AvgDisp();

    //! \brief Evaluate the qoi at an integration point
    //! \param global The global residual object
    //! \param local The local residual object
    //! \param w The integration point weight
    //! \param dv The differential volume (Jacobian) of the element at the point
    void evaluate(
        RCP<GlobalResidual<T>> global,
        RCP<LocalResidual<T>> local,
        apf::Vector3 const& iota,
        double w,
        double dv);

};

}
