#pragma once

//! \file elastic.hpp
//! \brief The interface for linear elastic local residuals

#include "local_residual.hpp"

namespace calibr8 {

template <typename T>
class Elastic : public LocalResidual<T> {
  public:
    Elastic(ParameterList const& inputs, int ndims);
    ~Elastic();
    void init_params();
    void init_variables_impl();
    int solve_nonlinear(RCP<GlobalResidual<T>> global);
    int evaluate(
        RCP<GlobalResidual<T>> global,
        bool force_path = false,
        int path = 0);
    bool is_finite_deformation() { return false; }
    Tensor<T> cauchy(RCP<GlobalResidual<T>> global);
    Tensor<T> dev_cauchy(RCP<GlobalResidual<T>> global);
    T hydro_cauchy(RCP<GlobalResidual<T>> global);
    T pressure_scale_factor();
};

}
