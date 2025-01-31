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
    //! \param state The state object
    //! \param disc The discretization object
    //! \param num_params The number of calibration parameters
    VirtualPower(
        RCP<ParameterList> params,
        RCP<State> state,
        RCP<Disc> disc,
        int num_params=0);

    //! \brief Compute the residual at a step
    //! \param step The current step to compute at
    void compute_residual_at_step(int step);

    //! \brief Compute the internal virtual power without computing the residual
    //! \param step The current step to compute at
    double compute_internal_virtual_power();

    //! \brief Compute the internal virtual power at a step
    //! \param step The current step to compute at
    double compute_at_step(int step);

    //! \brief Compute the squared virtual power mismatch and its gradient with forward sensitivities at a step
    //! \param step The current step to compute at
    //! \param step The internal virtual power
    //! \param step The gradient of the internal virtual power
    void compute_at_step_forward_sens(int step, double& internal_virtual_power,
        Array1D<double>& grad);

    //! \brief Compute the squared virtual power mismatch at a step based on adjoint sens
    //! \param step The current step to compute at
    //! \param step The scaled virtual power mismatch
    //! \param step The gradient of the objective function
    void compute_at_step_adjoint(int step, double scaled_virtual_power_mismatch,
        Array1D<double>& grad);

    void populate_vf_vector();

    ~VirtualPower();

  private:

    RCP<ParameterList> m_params;
    RCP<State> m_state;
    RCP<Disc> m_disc;
    Array1D<RCP<VectorT>> m_vf_vec[NUM_DISTRIB];
    Array1D<RCP<MultiVectorT>> m_mvec[NUM_DISTRIB];
    Array3D<EMatrix> m_local_sens; // (elem_set_idx, elem, int pt)
    Array3D<EVector> m_local_history_vectors;  // (elem_set_idx, elem, int pt)

    int m_num_params = 0;
    void initialize_sens_matrices();
    void initialize_adjoint_history_vectors();

};

}
