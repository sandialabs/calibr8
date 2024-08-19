#pragma once

//! \file hybrid_hyper_J2_plane_stress.hpp
//! \brief The interface for HybridHyperJ2PlaneStress local plasticity residuals

#include "local_residual.hpp"
#include "NN.hpp"

namespace calibr8 {

//! \brief The local residual for finite deformation hyperelastic J2 plane stress plasticity
//! \tparam T The underlying scalar type used for evaluations
//! \details This implements a concrete instance of the LocalResidual
//! base class for a finite deformation hyperelastic J2 plane strain plasticity model
template <typename T>
class HybridHyperJ2PlaneStress : public LocalResidual<T> {

  public:

    //! \brief The HybridHyperJ2PlaneStress constructor
    //! \param inputs The local residual parameterlist
    //! \param ndims The number of spatial dimensions
    HybridHyperJ2PlaneStress(ParameterList const& inputs, int ndims);

    //! \brief The HybridHyperJ2PlaneStress destructor
    ~HybridHyperJ2PlaneStress();

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

    EVector get_embedded_params() const;

    void set_embedded_params(EVector const& nn_values);

  private:

    int m_max_iters;
    double m_abs_tol;
    double m_rel_tol;

    double m_nn_input_scale = 1.;
    double m_nn_output_scale = 1.;

    RCP<ML::FFNN<T>> m_neural_network;
    T nn_hardening(T alpha);

    enum {ELASTIC = 0, PLASTIC = 1};

};

}
