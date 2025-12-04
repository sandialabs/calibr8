#pragma once

//! \file hypo_barlat.hpp
//! \brief The interface for hypoelastic Barlat local plasticity residuals

#include "local_residual.hpp"

namespace calibr8 {

//! \brief The local residual for HypoBarlat plasticity models
//! \tparam T The underlying scalar type used for evaluations
//! \details This implements a concrete instance of the LocalResidual
//! base class for a HypoBarlat plasticity model
template <typename T>
class HypoBarlat : public LocalResidual<T> {

  public:

    //! \brief The HypoBarlat constructor
    //! \param inputs The local residual parameterlist
    //! \param ndims The number of spatial dimensions
    HypoBarlat(ParameterList const& inputs, int ndims);

    //! \brief The HypoBarlat destructor
    ~HypoBarlat();

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

    Tensor<T> eval_d(RCP<GlobalResidual<T>> global);
    void compute_Q(RCP<GlobalResidual<T>> global);

    //! \brief Get the rotated Cauchy stress tensor
    //! \param global The global residual equations
    Tensor<T> rotated_cauchy(RCP<GlobalResidual<T>> global);

    void evaluate_phi_and_normal(T const& a, T& phi, Tensor<T>& n);
    void compute_cartesian_lab_to_mat_rotation(ParameterList const& inputs);

    int m_max_iters;
    double m_abs_tol;
    double m_rel_tol;
    double m_ls_beta;
    double m_ls_eta;
    int m_ls_max_evals;
    bool m_ls_print;

    bool m_compute_cylindrical_transform = false;
    // lab to material cartesian coordinate system transformation matrix
    Eigen::Matrix3d m_cartesian_lab_to_mat_rotation = Eigen::Matrix3d::Identity(3, 3);
    // lab to material cylindrical coordinate system transformation matrix
    Tensor<T> m_Q = minitensor::eye<T>(3);

    enum {ELASTIC = 0, PLASTIC = 1};

};

}
