#pragma once

//! \file surface_mismatch.hpp
//! \brief The interface for displacement mismatch QoIs

#include "qoi.hpp"

namespace calibr8 {

//! \brief The evalaution of a displacement mismatch over a surface
// TODO: change this name to displacement mismatch
template <typename T>
class SurfaceMismatch : public QoI<T> {

  public:

    //! \brief The average displacement constructor
    //! \param params The QoI parameter list
    SurfaceMismatch(ParameterList const& params);

    //! \brief The average displacement destructor
    ~SurfaceMismatch();

    //! \brief Perform initializations before the loop over elements
    //! \param disc The discretization object
    void before_elems(RCP<Disc> disc);

    //! \brief Evaluate the qoi at an integration point
    //! \param global The global residual object
    //! \param local The local residual object
    void evaluate(
        int elem_set,
        int elem,
        RCP<GlobalResidual<T>> global,
        RCP<LocalResidual<T>> local,
        apf::Vector3 const&,
        double,
        double);

  private:

    bool is_initd = false;
    std::string m_side_set = "";
    Array2D<int> m_mapping; // m_mapping[es_idx][elem_idx]

};

}
