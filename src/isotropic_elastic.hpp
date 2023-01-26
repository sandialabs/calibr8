#pragma once

//! \file isotropic_elastic.hpp
//! \brief The interface for isotropic linear elastic local residuals

#include "local_residual.hpp"

namespace calibr8 {

enum {DISPLACEMENT = 0, MIXED = 1};

template <typename T>
class IsotropicElastic : public LocalResidual<T> {
  public:
    IsotropicElastic(ParameterList const& inputs, int ndims);
    ~IsotropicElastic();
    void init_params();
    void init_variables_impl();
    int solve_nonlinear(RCP<GlobalResidual<T>> global);
    int evaluate(
        RCP<GlobalResidual<T>> global,
        bool force_path = false,
        int path = 0);
    bool is_finite_deformation() { return false; }
    Tensor<T> dev_cauchy(RCP<GlobalResidual<T>> global);
    Tensor<T> cauchy(RCP<GlobalResidual<T>> global);
    T hydro_cauchy(RCP<GlobalResidual<T>> global);
    T pressure_scale_factor();

  private:
    Tensor<T> cauchy_mixed(RCP<GlobalResidual<T>> global);
    int m_mode = -1;
};

}
