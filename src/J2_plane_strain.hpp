#pragma once

//! \file J2_plane_strain.hpp
//! \brief The interface for J2_plane_strain local plasticity residuals

#include "local_residual.hpp"

namespace calibr8 {

//! \brief The local residual for finite deformation J2 plane strain plasticity models
//! \tparam T The underlying scalar type used for evaluations
//! \details This implements a concrete instance of the LocalResidual
//! base class for a finite deformation J2 plane strain plasticity model
template <typename T>
class J2_plane_strain : public LocalResidual<T> {

  public:

    //! \brief The J2_plane_strain constructor
    //! \param inputs The local residual parameterlist
    //! \param ndims The number of spatial dimensions
    J2_plane_strain(ParameterList const& inputs, int ndims);

    //! \brief The J2_plane_strain destructor
    ~J2_plane_strain();

    //! \brief Initialize the local variables
    void init_variables_impl();

    //! \brief Solve the constitutive equations at the current point
    //! \param global The global residual equations
    int solve_nonlinear(RCP<GlobalResidual<T>> global);

    //! \brief Evaluate the constitutive equations at the current point
    //! \param global The global residual equations
    //! \param force_path Force a specific evaluation path
    //! \param path The evaluation path to force
    int evaluate(
        RCP<GlobalResidual<T>> global, 
        bool force_path = false,
        int path = 0);

    //! \brief Do these equations correspond to finite deformation
    bool is_finite_deformation() { return true; }

    //! \brief Get the deviatoric part of the Cauchy stress tensor
    //! \param global The global residual equations
    Tensor<T> dev_cauchy(RCP<GlobalResidual<T>> global);

  private:

    int m_max_iters;
    double m_abs_tol;
    double m_rel_tol;

    enum {ELASTIC = 0, PLASTIC = 1};

};

}