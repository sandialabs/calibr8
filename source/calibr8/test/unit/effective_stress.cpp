#include <cmath>

#include <Eigen/Dense>
#include <fad.hpp>
#include <hybrid_hypo_hill.hpp>
#include <ICNN.hpp>
#include <yield_functions.hpp>

#include <gtest/gtest.h>

using namespace calibr8;

namespace {

template <typename T>
struct QuadraticCorrection {
  double k;
  void operator()(Tensor<T> const& u, T& value, Tensor<T>& grad) const {
    Tensor<T> const dev_u = minitensor::dev(u);
    T norm_sq = T(0.);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        norm_sq += dev_u(i, j) * dev_u(i, j);
      }
    }
    value = k * norm_sq;
    grad = T(2. * k) * dev_u;
  }
};

double dev_norm_squared(Tensor<double> const& A) {
  Tensor<double> const dev_A = minitensor::dev(A);
  double norm_sq = 0.;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      norm_sq += dev_A(i, j) * dev_A(i, j);
    }
  }
  return norm_sq;
}

Vector<double> representative_hill_params() {
  return compute_hill_params(1.0, 0.9, 1.1, 1.0, 1.0, 1.0);
}

Tensor<double> representative_stress() {
  Tensor<double> TC = minitensor::zero<double>(3);
  TC(0, 0) = 120.;
  TC(1, 1) = -60.;
  TC(2, 2) = 30.;
  TC(0, 1) = TC(1, 0) = 40.;
  TC(1, 2) = TC(2, 1) = -25.;
  TC(2, 0) = TC(0, 2) = 15.;
  return TC;
}

int const COMP_I[6] = {0, 1, 2, 0, 1, 2};
int const COMP_J[6] = {0, 1, 2, 1, 2, 0};

}  // namespace

TEST(effective_stress, scale_solves_yield_surface)
{
  double const Y = 200.;
  double const k = 1.0e-3;
  Vector<double> const hill_params = representative_hill_params();
  Tensor<double> const TC = representative_stress();
  double const hill_TC = compute_hill_value(TC, hill_params);

  double c = 0.;
  Tensor<double> grad_c = minitensor::zero<double>(3);
  double const beta = solve_effective_stress_scale(
      TC, hill_TC, Y, 50, 1.0e-12, QuadraticCorrection<double>{k}, c, grad_c);

  double const dev_norm_sq = dev_norm_squared(TC);
  double const residual = beta * hill_TC + k * beta * beta * dev_norm_sq - Y;
  EXPECT_NEAR(residual, 0., 1.0e-9);

  double const a = k * dev_norm_sq;
  double const beta_exact =
      (-hill_TC + std::sqrt(hill_TC * hill_TC + 4. * a * Y)) / (2. * a);
  EXPECT_NEAR(beta, beta_exact, 1.0e-10 * beta_exact);
}

TEST(effective_stress, recovers_hill_scale_without_correction)
{
  double const Y = 200.;
  Vector<double> const hill_params = representative_hill_params();
  Tensor<double> const TC = representative_stress();
  double const hill_TC = compute_hill_value(TC, hill_params);

  double c = 0.;
  Tensor<double> grad_c = minitensor::zero<double>(3);
  double const beta = solve_effective_stress_scale(
      TC, hill_TC, Y, 50, 1.0e-12, QuadraticCorrection<double>{0.}, c, grad_c);

  EXPECT_NEAR(beta, Y / hill_TC, 1.0e-12 * (Y / hill_TC));
}

TEST(effective_stress, scale_is_degree_minus_one_homogeneous)
{
  double const Y = 200.;
  double const k = 1.0e-3;
  double const s = 2.5;
  Vector<double> const hill_params = representative_hill_params();
  Tensor<double> const TC = representative_stress();

  double c = 0.;
  Tensor<double> grad_c = minitensor::zero<double>(3);
  double const beta = solve_effective_stress_scale(
      TC, compute_hill_value(TC, hill_params), Y, 50, 1.0e-12,
      QuadraticCorrection<double>{k}, c, grad_c);

  Tensor<double> const sTC = s * TC;
  double const beta_s = solve_effective_stress_scale(
      sTC, compute_hill_value(sTC, hill_params), Y, 50, 1.0e-12,
      QuadraticCorrection<double>{k}, c, grad_c);

  EXPECT_NEAR(beta_s, beta / s, 1.0e-10 * (beta / s));
}

TEST(effective_stress, scale_derivative_matches_finite_difference)
{
  double const Y_val = 200.;
  double const k = 1.0e-3;
  int const nderiv = 6;
  Tensor<double> const TC0 = representative_stress();
  Vector<double> const hill_params_d = representative_hill_params();

  Tensor<FADT> TC = minitensor::zero<FADT>(3);
  for (int idx = 0; idx < nderiv; ++idx) {
    int const a = COMP_I[idx];
    int const b = COMP_J[idx];
    FADT const comp(nderiv, idx, TC0(a, b));
    TC(a, b) = comp;
    if (a != b) {
      TC(b, a) = comp;
    }
  }

  Vector<FADT> const hill_params = compute_hill_params(
      FADT(1.0), FADT(0.9), FADT(1.1), FADT(1.0), FADT(1.0), FADT(1.0));
  FADT const Y(Y_val);
  FADT const hill_TC = compute_hill_value(TC, hill_params);
  FADT c = FADT(0.);
  Tensor<FADT> grad_c = minitensor::zero<FADT>(3);
  FADT const beta = solve_effective_stress_scale(
      TC, hill_TC, Y, 50, 1.0e-12, QuadraticCorrection<FADT>{k}, c, grad_c);

  auto solve_double = [&](Tensor<double> const& tc) {
    double c_d = 0.;
    Tensor<double> grad_c_d = minitensor::zero<double>(3);
    return solve_effective_stress_scale(
        tc, compute_hill_value(tc, hill_params_d), Y_val, 50, 1.0e-12,
        QuadraticCorrection<double>{k}, c_d, grad_c_d);
  };

  double const h = 1.0e-3;
  for (int idx = 0; idx < nderiv; ++idx) {
    int const a = COMP_I[idx];
    int const b = COMP_J[idx];
    Tensor<double> tc_p = TC0;
    Tensor<double> tc_m = TC0;
    tc_p(a, b) += h;
    tc_m(a, b) -= h;
    if (a != b) {
      tc_p(b, a) += h;
      tc_m(b, a) -= h;
    }
    double const fd = (solve_double(tc_p) - solve_double(tc_m)) / (2. * h);
    EXPECT_NEAR(beta.dx(idx), fd, 1.0e-5 * std::abs(fd) + 1.0e-9);
  }
}

TEST(effective_stress, icnn_correction_gradient_matches_finite_difference)
{
  char const* activation = "softplus";
  std::vector<int> const topology = {6, 4, 1};
  double const in_scale = 0.01;
  double const out_scale = 5.0;
  ML::FICNN<double> nn(activation, topology);

  Tensor<double> const TC = representative_stress();

  double value = 0.;
  Tensor<double> grad = minitensor::zero<double>(3);
  icnn_value_and_grad(
      nn, activation, topology, in_scale, out_scale, false, TC, value, grad);

  double const h = 1.0e-4;
  for (int idx = 0; idx < 6; ++idx) {
    int const a = COMP_I[idx];
    int const b = COMP_J[idx];
    Tensor<double> V = minitensor::zero<double>(3);
    V(a, b) = 1.;
    V(b, a) = 1.;

    double grad_dot_V = 0.;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        grad_dot_V += grad(i, j) * V(i, j);
      }
    }

    double vp = 0.;
    double vm = 0.;
    Tensor<double> g = minitensor::zero<double>(3);
    icnn_value_and_grad(
        nn, activation, topology, in_scale, out_scale, false, TC + h * V, vp, g);
    icnn_value_and_grad(
        nn, activation, topology, in_scale, out_scale, false, TC - h * V, vm, g);
    double const fd = (vp - vm) / (2. * h);
    EXPECT_NEAR(grad_dot_V, fd, 1.0e-5 * std::abs(fd) + 1.0e-9);
  }
}

TEST(effective_stress, scale_derivative_with_icnn_matches_finite_difference)
{
  char const* activation = "softplus";
  std::vector<int> const topology = {6, 4, 1};
  double const in_scale = 0.01;
  double const out_scale = 5.0;
  double const Y_val = 200.;
  int const nderiv = 6;
  Tensor<double> const TC0 = representative_stress();
  Vector<double> const hill_params_d = representative_hill_params();

  ML::FICNN<double> nn_d(activation, topology);
  Eigen::Matrix<double, Eigen::Dynamic, 1> const weights = nn_d.get_params();
  ML::FICNN<FADT> nn(activation, topology);
  Eigen::Matrix<FADT, Eigen::Dynamic, 1> fadt_weights(weights.size());
  for (int p = 0; p < weights.size(); ++p) {
    fadt_weights(p) = FADT(weights(p));
  }
  nn.set_params(fadt_weights);

  Tensor<FADT> TC = minitensor::zero<FADT>(3);
  for (int idx = 0; idx < nderiv; ++idx) {
    int const a = COMP_I[idx];
    int const b = COMP_J[idx];
    FADT const comp(nderiv, idx, TC0(a, b));
    TC(a, b) = comp;
    if (a != b) {
      TC(b, a) = comp;
    }
  }

  Vector<FADT> const hill_params = compute_hill_params(
      FADT(1.0), FADT(0.9), FADT(1.1), FADT(1.0), FADT(1.0), FADT(1.0));
  FADT const Y(Y_val);
  FADT const hill_TC = compute_hill_value(TC, hill_params);
  FADT c = FADT(0.);
  Tensor<FADT> grad_c = minitensor::zero<FADT>(3);
  FADT const beta = solve_effective_stress_scale(
      TC, hill_TC, Y, 50, 1.0e-12,
      [&](Tensor<FADT> const& u, FADT& value, Tensor<FADT>& grad) {
        icnn_value_and_grad(
            nn, activation, topology, in_scale, out_scale, false, u, value, grad);
      },
      c, grad_c);

  auto solve_double = [&](Tensor<double> const& tc) {
    double c_d = 0.;
    Tensor<double> grad_c_d = minitensor::zero<double>(3);
    return solve_effective_stress_scale(
        tc, compute_hill_value(tc, hill_params_d), Y_val, 50, 1.0e-12,
        [&](Tensor<double> const& u, double& value, Tensor<double>& grad) {
          icnn_value_and_grad(
              nn_d, activation, topology, in_scale, out_scale, false, u, value,
              grad);
        },
        c_d, grad_c_d);
  };

  double const h = 1.0e-3;
  for (int idx = 0; idx < nderiv; ++idx) {
    int const a = COMP_I[idx];
    int const b = COMP_J[idx];
    Tensor<double> tc_p = TC0;
    Tensor<double> tc_m = TC0;
    tc_p(a, b) += h;
    tc_m(a, b) -= h;
    if (a != b) {
      tc_p(b, a) += h;
      tc_m(b, a) -= h;
    }
    double const fd = (solve_double(tc_p) - solve_double(tc_m)) / (2. * h);
    EXPECT_NEAR(beta.dx(idx), fd, 1.0e-5 * std::abs(fd) + 1.0e-8);
  }
}

TEST(effective_stress, icnn_symmetric_correction)
{
  char const* activation = "softplus";
  std::vector<int> const topology = {6, 4, 1};
  double const in_scale = 0.01;
  double const out_scale = 5.0;
  ML::FICNN<double> nn(activation, topology);

  Tensor<double> const TC = representative_stress();

  double v_plus = 0.;
  double v_minus = 0.;
  Tensor<double> grad = minitensor::zero<double>(3);
  Tensor<double> grad_minus = minitensor::zero<double>(3);
  icnn_value_and_grad(
      nn, activation, topology, in_scale, out_scale, true, TC, v_plus, grad);
  icnn_value_and_grad(
      nn, activation, topology, in_scale, out_scale, true, -1. * TC, v_minus,
      grad_minus);
  EXPECT_NEAR(v_plus, v_minus, 1.0e-10 * (std::abs(v_plus) + 1.));

  double const h = 1.0e-4;
  for (int idx = 0; idx < 6; ++idx) {
    int const a = COMP_I[idx];
    int const b = COMP_J[idx];
    Tensor<double> V = minitensor::zero<double>(3);
    V(a, b) = 1.;
    V(b, a) = 1.;

    double grad_dot_V = 0.;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        grad_dot_V += grad(i, j) * V(i, j);
      }
    }

    double vp = 0.;
    double vm = 0.;
    Tensor<double> g = minitensor::zero<double>(3);
    icnn_value_and_grad(
        nn, activation, topology, in_scale, out_scale, true, TC + h * V, vp, g);
    icnn_value_and_grad(
        nn, activation, topology, in_scale, out_scale, true, TC - h * V, vm, g);
    double const fd = (vp - vm) / (2. * h);
    EXPECT_NEAR(grad_dot_V, fd, 1.0e-5 * std::abs(fd) + 1.0e-9);
  }

  Tensor<double> const zero_stress = minitensor::zero<double>(3);
  double v_zero = 0.;
  Tensor<double> grad_zero = minitensor::zero<double>(3);
  icnn_value_and_grad(
      nn, activation, topology, in_scale, out_scale, true, zero_stress, v_zero,
      grad_zero);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(grad_zero(i, j), 0., 1.0e-12);
    }
  }
}
