#pragma once

//! \file avg_stress.hpp
//! \brief The interface for average stress QoIs

#include "qoi.hpp"

namespace calibr8 {

template <typename T>
class AvgStress : public QoI<T> {

  private:

    std::string m_elem_set;
    std::vector<std::string> m_elem_set_names;

  public:

    //! \brief The average stress QoI constructor
    AvgStress(ParameterList const& params);

    //! \brief The average stress destructor
    ~AvgStress();

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
