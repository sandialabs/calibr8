#pragma once

#include "arrays.hpp"
#include "defines.hpp"

//! \file adjoint_aux.hpp
//! \brief The interface for solving adjoint problems with auxiliary variables

namespace calibr8 {

//! \cond
// forward declarations
class Disc;
class State;
//! \endcond

//! \brief The interface for solving adjoint problems with auxiliary variables
class AdjointAux {

  public:

    //! \brief Construct the adjoint_aux problem object
    //! \param params The input parameters describing the problem
    //! \param state The application state object
    //! \param disc The discretization object
    AdjointAux(RCP<ParameterList> params, RCP<State> state, RCP<Disc> disc);

    //! \brief Solve the adjoint problem at a given step
    //! \param step The curent step to solve at
    //! \param C_error The local residual element-wise error field
    //! \param D_error The aux residual element-wise error field
    void solve_and_compute_error_at_step(int step, apf::Field* C_error, apf::Field* D_error);

  private:

    RCP<ParameterList> m_params;
    RCP<State> m_state;
    RCP<Disc> m_disc;

    Array3D<EVector> aux_history; // (elem_set_idx, elem, int pt)
    Array3D<EVector> global_history; // (elem_set_idx, elem, int pt)
    Array3D<EVector> local_history;  // (elem_set_idx, elem, int pt)

    void initialize_history_vectors();

};

}
