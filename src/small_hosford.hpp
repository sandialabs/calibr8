#pragma once

//! \file hosford.hpp
//! \brief The interface for the small strain Hosford local plasticity residuals

#include "local_residual.hpp"

namespace calibr8 {

//! \brief The local residual for the small strain Hosford plasticity model
//! \tparam T The underlying scalar type used for evaluations
//! \details This implements a concrete instance of the LocalResidual
//! base class for a small strain Hosford plasticity model
template <typename T>
class SmallHosford : public LocalResidual<T> {

  public:

    //! \brief The SmallHosford constructor
    //! \param inputs The local residual parameterlist
    //! \param ndims The number of spatial dimensions
    SmallHosford(ParameterList const& inputs, int ndims);

    //! \brief The SmallHosford destructor
    ~SmallHosford();

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

    void evaluate_phi_and_normal(RCP<GlobalResidual<T>> global,
        T const& a, T& phi, Tensor<T>& n);

    int m_max_iters;
    double m_abs_tol;
    double m_rel_tol;
    double m_ls_beta;
    double m_ls_eta;
    int m_ls_max_evals;
    bool m_ls_print;

    enum {ELASTIC = 0, PLASTIC = 1};

};

}
