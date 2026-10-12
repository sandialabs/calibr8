#pragma once

//! \file hybrid_hypo_hill.hpp
//! \brief Hypoelastic Hill plasticity with an ICNN effective stress correction

#include <cmath>
#include <string>
#include <vector>
#include "local_residual.hpp"

namespace ML {
template <class ScalarT> class FICNN;
}

namespace calibr8 {

//! \brief The local residual for hypoelastic Hill plasticity with an ICNN
//! effective stress correction
//! \tparam T The underlying scalar type used for evaluations
template <typename T>
class HybridHypoHill : public LocalResidual<T> {

  public:

    //! \brief The HybridHypoHill constructor
    //! \param inputs The local residual parameterlist
    //! \param ndims The number of spatial dimensions
    HybridHypoHill(ParameterList const& inputs, int ndims);

    //! \brief The HybridHypoHill destructor
    ~HybridHypoHill();

    //! \brief Initialize the parameters
    void init_params() override;

    //! \brief Initialize the local variables
    void init_variables_impl() override;

    //! \brief Solve the constitutive equations at the current point
    //! \param global The global residual equations
    int solve_nonlinear(RCP<GlobalResidual<T>> global) override;

    //! \brief Evaluate the constitutive equations at the current point
    //! \param global The global residual equations
    //! \param force_path Force a specific evaluation path
    //! \param path The evaluation path to force
    int evaluate(
        RCP<GlobalResidual<T>> global,
        bool force_path = false,
        int path = 0) override;

    //! \brief Do these equations correspond to finite deformation
    bool is_finite_deformation() override { return true; }

    //! \brief Get the Cauchy stress tensor
    //! \param global The global residual equations
    Tensor<T> cauchy(RCP<GlobalResidual<T>> global) override;

    //! \brief Get the deviatoric part of the Cauchy stress tensor
    //! \param global The global residual equations
    Tensor<T> dev_cauchy(RCP<GlobalResidual<T>> global) override;

    //! \brief Get the hydrostatic part of the Cauchy stress tensor
    //! \param global The global residual equations
    T hydro_cauchy(RCP<GlobalResidual<T>> global) override;

    //! \brief Get the pressure variable scale factor
    //! \param global The global residual equations
    T pressure_scale_factor() override;

    //! \brief Get the embedded ICNN weights as DFAD parameters
    EVector get_dfad_params() const override;

    //! \brief Set the embedded ICNN weights from DFAD parameters
    //! \param nn_values The new weight values
    void set_dfad_params(EVector const& nn_values) override;

  private:

    //! \brief Get the rotated Cauchy stress tensor
    //! \param global The global residual equations
    Tensor<T> rotated_cauchy(RCP<GlobalResidual<T>> global);

    //! \brief Get the unrotated rate of deformation increment
    //! \param global The global residual equations
    Tensor<T> eval_d(RCP<GlobalResidual<T>> global);

    //! \brief The homogeneous effective stress and flow normal
    //! \param TC The unrotated Cauchy stress
    //! \param hill_params The Hill coefficients
    //! \param Y The initial yield stress
    //! \param n Filled with the flow normal
    T effective_stress_and_normal(
        Tensor<T> const& TC,
        Vector<T> const& hill_params,
        T const& Y,
        Tensor<T>& n);

    int m_max_iters;
    double m_abs_tol;
    double m_rel_tol;

    Tensor<T> m_d;
    bool m_kinematics_cached = false;

    std::string m_activation;
    std::vector<int> m_topology;
    double m_nn_input_scale = 1.;
    double m_nn_output_scale = 1.;
    bool m_symmetrize = true;
    RCP<ML::FICNN<T>> m_icnn;

    enum {ELASTIC = 0, PLASTIC = 1};

};

//! \brief The ICNN effective stress correction and its stress gradient
template <typename T>
void icnn_value_and_grad(
    ML::FICNN<T>& nn,
    char const* activation,
    std::vector<int> const& topology,
    double in_scale,
    double out_scale,
    bool symmetrize,
    Tensor<T> const& stress,
    T& value,
    Tensor<T>& grad);

//! \brief Solve phi(beta * TC) = Y for the effective stress scale beta
template <typename T, typename CorrectionFn>
T solve_effective_stress_scale(
    Tensor<T> const& TC,
    T const& hill_TC,
    T const& Y,
    int max_iters,
    double tol,
    CorrectionFn correction,
    T& c,
    Tensor<T>& grad_c) {

  T beta = Y / hill_TC;
  double const beta_tol = tol * val(Y);
  int const dim = TC.get_dimension();
  for (int it = 0; it < max_iters; ++it) {
    Tensor<T> const u = beta * TC;
    correction(u, c, grad_c);
    T const r = beta * hill_TC + c - Y;
    if (std::abs(val(r)) < beta_tol) {
      break;
    }
    T slope = hill_TC;
    for (int i = 0; i < dim; ++i) {
      for (int j = 0; j < dim; ++j) {
        slope += grad_c(i, j) * TC(i, j);
      }
    }
    beta -= r / slope;
  }
  return beta;
}

}
