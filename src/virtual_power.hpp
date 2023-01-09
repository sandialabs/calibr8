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

//! \brief The interface for solving primal problems
class VirtualPower {

  public:

    //! \brief Construct the primal problem object
    //! \param params The input parameters describing the problem
    //! \param disc The discretization object
    VirtualPower(
        RCP<ParameterList> params,
        RCP<State> state,
        RCP<Disc> disc);

    //! \brief Compute the squared virtual power mismatch at a step
    //! \param step The current step to compute at
    //! \param t The simulation time at the current step
    //! \param dt The time increment used to get to the current time
    double compute_at_step(int step, double t, double dt);

    ~VirtualPower() { resize(m_vf_vec[OWNED], 0); resize(m_vf_vec[GHOST], 0); }

  private:

    RCP<ParameterList> m_params;
    RCP<State> m_state;
    RCP<Disc> m_disc;
    Array1D<RCP<VectorT>> m_vf_vec[NUM_DISTRIB];

};

}
