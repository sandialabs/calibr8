#pragma once

#include "defines.hpp"

//! \file virtual_power.hpp
//! \brief The interface for solving primal problems

namespace calibr8 {

//! \cond
// forward declarations
class Disc;
class State;
//! \endcond

class EquilibriumGap {

  public:

    //! \brief Construct the Equilibrium Gap object
    //! \param params The input parameters describing the problem
    //! \param state The state object
    //! \param disc The discretization object
    //! \param num_params The number of calibration parameters
    EquilibriumGap(
        RCP<ParameterList> params,
        RCP<State> state,
        RCP<Disc> disc,
        int num_params = 0);

    ~EquilibriumGap();

    //! \brief Compute the virtual equilibrium gap at a step
    //! \param step The current step to compute at
    //! \details TODO: maybe we don't need this?
    double compute_at_step(int step);

    //! \brief Compute the equilibrium gap at a step based on adjoint sens
    //! \param step The current step to compute at
    //! \param grad The gradient of the objective function
    void compute_at_step_adjoint(int step, Array1D<double>& grad);

  private:

    void initialize_adjoint_history_vectors();

  private:

    int m_num_params = 0;
    RCP<ParameterList> m_params;
    RCP<State> m_state;
    RCP<Disc> m_disc;
    Array3D<EVector> m_local_history_vectors;   // (elem_set_idx, elem, pt)

};

}
