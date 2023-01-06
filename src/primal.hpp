#pragma once

#include "defines.hpp"

//! \file primal.hpp
//! \brief The interface for solving primal problems

namespace calibr8 {

//! \cond
// forward declarations
class Disc;
class State;
//! \endcond

//! \brief The interface for solving primal problems
class Primal {

  public:

    //! \brief Construct the primal problem object
    //! \param params The input parameters describing the problem
    //! \param disc The discretization object
    Primal(
        RCP<ParameterList> params,
        RCP<State> state,
        RCP<Disc> disc);

    //! \brief Solve the primal problem at a given step
    //! \param step The current step to solve at
    //! \param t The simulation time at the current step
    //! \param dt The time increment used to get to the current time
    void solve_at_step(int step, double t, double dt);

  private:

    RCP<ParameterList> m_params;
    RCP<State> m_state;
    RCP<Disc> m_disc;

};

}
