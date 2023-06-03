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
    int solve_nonlinear(RCP<GlobalResidual<T>> global, int step);
    int evaluate(
        RCP<GlobalResidual<T>> global,
        bool force_path = false,
        int path = 0,
        int step = 1);
    bool is_finite_deformation() { return false; }
    Tensor<T> dev_cauchy(RCP<GlobalResidual<T>> global);
    Tensor<T> cauchy(RCP<GlobalResidual<T>> global);
    T hydro_cauchy(RCP<GlobalResidual<T>> global);
    T pressure_scale_factor();

    void compute_present_aux_variables(RCP<GlobalResidual<T>> global, int step);

    EMatrix daux_dxT(RCP<GlobalResidual<T>> global, int step);
    EMatrix daux_dx_prevT(RCP<GlobalResidual<T>> global, int step);
    EVector daux_dchi_prev_diag(int step);
    EMatrix dlocal_dchi_prevT(int step);
    EVector eigen_aux_residual(RCP<GlobalResidual<T>> global, int step);

  private:
    void read_prony_series(ParameterList const& prony_files);
    void compute_temperature(ParameterList const& inputs);
    void compute_shift_factors();
    double compute_J3(Array1D<double> const& J3_k_prev);
    Array1D<double> compute_J3_k(double const psi, Array1D<double> const& J3_k_prev,
        bool compute_func = true);
    void residual_and_deriv(double const psi, Array1D<double> const& J3_k_prev,
        double const temp, double& r_val, double& r_deriv);
    double lag_nonlinear_solve(double const psi, Array1D<double> const& J3_k_prev,
        double const temp, double const tol = 1e-10, int const max_iters = 10);

    void compute_delta_strains(RCP<GlobalResidual<T>> global,
        T& delta_vol_eps, Tensor<T>& delta_dev_eps);

    Tensor<T> cauchy_mixed(RCP<GlobalResidual<T>> global);

    double m_delta_t = 0.;
    double m_delta_temp = 0.;
    double m_temp_ref = 0.;
    double m_C_1 = 0.;
    double m_C_2 = 0.;

    Array2D<double> m_vol_prony;
    Array2D<double> m_shear_prony;
    Array1D<double> m_temperature;
    Array1D<double> m_log10_shift_factor;
    Array1D<double> m_J3;

    int m_mode = -1;
};

}
