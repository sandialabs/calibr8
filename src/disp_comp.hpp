#pragma once

//! \file disp_comp.hpp
//! \brief The interface for average displacement QoIs over
//!  a subdomain

#include "qoi.hpp"

namespace calibr8 {

//! \brief The evalaution of an average displacement QoI
template <typename T>
class DispComp : public QoI<T> {

  private:

    int m_component;
    std::string m_elem_set;
    std::vector<std::string> m_elem_set_names;

  public:

    //! \brief The average displacement constructor
    DispComp(ParameterList const& params);

    //! \brief The average displacement destructor
    ~DispComp();

    //! \brief Perform initializations before the loop over elements
    //! \param disc The discretization object
    void before_elems(RCP<Disc> disc, int step);

    //! \brief Evaluate the qoi at an integration point
    //! \param elem_set The index of the current element set
    //! \param elem_idx The index of the current element in the element set
    //! \param global The global residual object
    //! \param local The local residual object
    //! \param w The integration point weight
    //! \param dv The differential volume (Jacobian) of the element at the point
    void evaluate(
        int elem_set,
        int elem,
        RCP<GlobalResidual<T>> global,
        RCP<LocalResidual<T>> local,
        apf::Vector3 const& iota,
        double w,
        double dv);

};

}
