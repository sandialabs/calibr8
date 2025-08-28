#pragma once

//! \file avg_local_var.hpp
//! \brief The interface for generic averaged local QoIs

#include "qoi.hpp"

namespace calibr8 {

template <typename T>
class AvgLocalVar : public QoI<T> {

  private:

    int m_resid_idx = -1;

    std::string m_elem_set;
    std::vector<std::string> m_elem_set_names;

  public:

    //! \brief The average local variable QoI constructor
    AvgLocalVar(ParameterList const& params);

    //! \brief The destructor
    ~AvgLocalVar();

    //! \brief Perform initialization before element loop
    void before_elems(RCP<Disc> disc, int step);

    //! \brief Evaluate the QoI at the integration point
    //! \param elem_set The current element set index
    //! \param elem_idx The current element in the element set
    //! \param global The global residual object
    //! \param local The local residual object
    //! \param iota The parametric integration point
    //! \param w The integration point weight
    //! \param dv The differential volume of the element
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
