#pragma once

//! \file small_hill.hpp
//! \brief The interface for small strain Hill local plasticity residuals

#include "local_residual.hpp"

namespace calibr8 {

//! \brief The local residual for small strain Hill plasticity models
//! \tparam T The underlying scalar type used for evaluations
//! \details This implements a concrete instance of the LocalResidual
//! base class for a small strain J2 plasticity model
template <typename T>
class SmallHill : public LocalResidual<T> {

  public:

    //! \brief The SmallHill constructor
    //! \param inputs The local residual parameterlist
    //! \param ndims The number of spatial dimensions
    SmallHill(ParameterList const& inputs, int ndims);

    //! \brief The SmallHill destructor
    ~SmallHill();

    //! \brief Initialize the parameters
    void init_params();

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
    bool is_finite_deformation() { return false; }

    //! \brief Get the Cauchy stress tensor
    //! \param global The global residual equations
    Tensor<T> cauchy(RCP<GlobalResidual<T>> global);

    //! \brief Get the deviatoric part of the Cauchy stress tensor
    //! \param global The global residual equations
    Tensor<T> dev_cauchy(RCP<GlobalResidual<T>> global);

    //! \brief Get the hydrostatic part of the Cauchy stress tensor
    //! \param global The global residual equations
    T hydro_cauchy(RCP<GlobalResidual<T>> global);

    //! \brief Get the pressure variable scale factor
    //! \param global The global residual equations
    T pressure_scale_factor();

  private:

    int m_max_iters;
    double m_abs_tol;
    double m_rel_tol;

    enum {ELASTIC = 0, PLASTIC = 1};

};

}
