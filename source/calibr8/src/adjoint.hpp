#pragma once

#include "arrays.hpp"
#include "defines.hpp"

//! \file adjoint.hpp
//! \brief The interface for solving adjoint problems

namespace calibr8 {

//! \cond
// forward declarations
class Disc;
class State;
//! \endcond

//! \brief The interface for solving adjoint problems
class Adjoint {

  public:

    //! \brief Construct the adjoint problem object
    //! \param params The input parameters describing the problem
    //! \param state The application state object
    //! \param disc The discretization object
    Adjoint(RCP<ParameterList> params, RCP<State> state, RCP<Disc> disc);

    //! \brief Solve the adjoint problem at a given step
    //! \param step The curent step to solve at
    void solve_at_step(int step);

  private:

    RCP<ParameterList> m_params;
    RCP<State> m_state;
    RCP<Disc> m_disc;

    Array3D<EVector> global_history; // (elem_set_idx, elem, int pt)
    Array3D<EVector> local_history;  // (elem_set_idx, elem, int pt)

    void initialize_history_vectors();

};

}
