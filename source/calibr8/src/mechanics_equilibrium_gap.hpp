#pragma once

//! \file mechanics_equilibrium_gap.hpp
//! \brief The interface for equilibrium gap global residual

#include "global_residual.hpp"

namespace calibr8 {

//! \brief The global residual for mechanics problems
//! \tparam T The underlying scalar type used for evaluations
//! \details This implements a concrete instance of the GlobalResidual
//! base class for mechanics with a stabilized displacement/pressure
//! formulation
template <typename T>
class MechanicsEquilibriumGap : public GlobalResidual<T> {

  public:

    //! \brief the mechanics constructor
    //! \param params The input global residual parameters
    //! \param ndims The number of spatial dimensions
    MechanicsEquilibriumGap(ParameterList const& params, int ndims);

    //! \brief The mechanics destructor
    ~MechanicsEquilibriumGap();

    //! \brief Perform initializations before the loop over elements
    //! \param disc The discretization object
    //! \param mode The type of weight to use
    //! \details This will initialize the local element-level quantities
    //! that this class is responsible for using/computing.
    //! It will also set up the m_mapping variable.
    void before_elems(
        RCP<Disc> disc,
        int mode=NORMAL_WEIGHT,
        Array1D<apf::Field*> const& adjoint_fields=Array1D<apf::Field*>()) override;

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
        int ip_set) override;

  private:
    double m_thickness = 1.;
    Array1D<std::string> m_side_set_names;
    bool m_mapping_is_initd = false;
    Array2D<int> m_mapping; // m_mapping[es_idx][elem_idx]

};

}
