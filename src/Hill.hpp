#pragma once

//! \file Hill.hpp
//! \brief The interface for hypoelastic Hill local plasticity residuals

#include "local_residual.hpp"

namespace calibr8 {

//! \brief The local residual for Hill plasticity models
//! \tparam T The underlying scalar type used for evaluations
//! \details This implements a concrete instance of the LocalResidual
//! base class for a Hill plasticity model
template <typename T>
class Hill : public LocalResidual<T> {

  public:

    //! \brief The Hill constructor
    //! \param inputs The local residual parameterlist
    //! \param ndims The number of spatial dimensions
    Hill(ParameterList const& inputs, int ndims);

    //! \brief The Hill destructor
    ~Hill();

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
    bool is_finite_deformation() { return true; }

    //! \brief Get the rotated (not deviatoric) Cauchy stress tensor
    //! \param global The global residual equations
    Tensor<T> dev_cauchy(RCP<GlobalResidual<T>> global);

    //! \brief Get the deviatoric part of the Cauchy stress tensor
    //! \param global The global residual equations
    //! \param p The pressure
    Tensor<T> cauchy(RCP<GlobalResidual<T>> global, T p);

  private:

    int m_max_iters;
    double m_abs_tol;
    double m_rel_tol;

    enum {ELASTIC = 0, PLASTIC = 1};

};

}
