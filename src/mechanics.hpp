#pragma once

//! \file mechanics.hpp
//! \brief The interface for mechanics global residuals

#include "global_residual.hpp"

namespace calibr8 {

//! \brief The global residual for mechanics problems
//! \tparam T The underlying scalar type used for evaluations
//! \details This implements a concrete instance of the GlobalResidual
//! base class for mechanics with a stabilized displacement/pressure
//! formulation
template <typename T>
class Mechanics : public GlobalResidual<T> {

  public:

    //! \brief the mechanics constructor
    //! \param params The input global residual parameters
    //! \param ndims The number of spatial dimensions
    Mechanics(ParameterList const& params, int ndims);

    //! \brief The mechanics destructor
    ~Mechanics();

    //! \brief Evaluate the residual at an integration point
    //! \param local The local residual object
    //! \param iota The integration point in the reference element space
    //! \param w The integration point weight
    //! \param dv The differential volume (Jacobian) of the element at the point
    //! \param ip_set The integration point set index
    void evaluate(
        RCP<LocalResidual<T>> local,
        apf::Vector3 const& iota,
        double w,
        double dv,
        int ip_set);

    //! \brief Evaluate the residual at an integration point
    //! \param local The local residual object
    //! \param iota The integration point in the reference element space
    //! \param w The integration point weight
    //! \param dv The differential volume (Jacobian) of the element at the point
    void evaluate_extra(
        RCP<LocalResidual<T>> local,
        apf::Vector3 const& iota,
        double w,
        double dv);

};

}
