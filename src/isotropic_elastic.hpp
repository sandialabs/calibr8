#pragma once

//! \file isotropic_elastic.hpp
//! \brief The interface for isotropic linear elastic local residuals

#include "local_residual.hpp"

namespace calibr8 {

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
    bool is_hypoelastic() { return false; }
    bool is_plane_stress() { return false; }
    Tensor<T> dev_cauchy(RCP<GlobalResidual<T>> global);
    Tensor<T> cauchy(RCP<GlobalResidual<T>> global, T p);
};

}
