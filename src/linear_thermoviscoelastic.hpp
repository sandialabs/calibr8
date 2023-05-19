#pragma once

//! \file linear_thermoviscoelastic.hpp
//! \brief The interface for linear thermoviscoelastic local residual

#include "local_residual.hpp"

namespace calibr8 {

template <typename T>
class LTVE : public LocalResidual<T> {
  public:
    LTVE(ParameterList const& inputs, int ndims);
    ~LTVE();
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

  //private:
};

}
