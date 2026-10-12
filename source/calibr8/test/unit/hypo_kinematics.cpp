#include <cmath>

#include <gtest/gtest.h>

#include <hypo_kinematics.hpp>

using namespace calibr8;

namespace {

void expect_tensor_near_zero(Tensor<double> const& A, double tol) {
  int const ndims = A.get_dimension();
  for (int i = 0; i < ndims; ++i) {
    for (int j = 0; j < ndims; ++j) {
      EXPECT_NEAR(A(i, j), 0., tol);
    }
  }
}

Tensor<double> rotation_z(double theta) {
  Tensor<double> R = minitensor::zero<double>(3);
  R(0, 0) = std::cos(theta);
  R(0, 1) = -std::sin(theta);
  R(1, 0) = std::sin(theta);
  R(1, 1) = std::cos(theta);
  R(2, 2) = 1.;
  return R;
}

}

TEST(hypo_kinematics, zero_increment)
{
  Tensor<double> const F = minitensor::eye<double>(3);
  Tensor<double> const d =
      compute_unrotated_rate_of_deformation_increment(F, F);
  expect_tensor_near_zero(d, 1.0e-14);
}

TEST(hypo_kinematics, trace_matches_log_volume_ratio)
{
  Tensor<double> F = minitensor::eye<double>(3);
  Tensor<double> const F_prev = minitensor::eye<double>(3);
  F(0, 0) = 1.2;
  F(1, 1) = 0.9;
  F(2, 2) = 1.1;

  Tensor<double> const d =
      compute_unrotated_rate_of_deformation_increment(F, F_prev);

  double const expected =
      std::log(minitensor::det(F) / minitensor::det(F_prev));
  EXPECT_NEAR(minitensor::trace(d), expected, 1.0e-14);
}

TEST(hypo_kinematics, pure_volumetric_stretch_is_isotropic)
{
  double const stretch = 1.2;
  Tensor<double> const F = stretch * minitensor::eye<double>(3);
  Tensor<double> const F_prev = minitensor::eye<double>(3);

  Tensor<double> const d =
      compute_unrotated_rate_of_deformation_increment(F, F_prev);

  double const expected = std::log(stretch);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(d(i, i), expected, 1.0e-14);
  }
  EXPECT_NEAR(d(0, 1), 0., 1.0e-14);
  EXPECT_NEAR(d(0, 2), 0., 1.0e-14);
  EXPECT_NEAR(d(1, 2), 0., 1.0e-14);
}

TEST(hypo_kinematics, small_increment_matches_symmetric_gradient_increment)
{
  Tensor<double> F = minitensor::eye<double>(3);
  Tensor<double> const F_prev = minitensor::eye<double>(3);

  Tensor<double> H = minitensor::zero<double>(3);
  H(0, 0) = 1.0e-7;
  H(1, 1) = -2.0e-7;
  H(2, 2) = 3.0e-7;
  H(0, 1) = 4.0e-7;
  H(1, 0) = -1.0e-7;
  H(1, 2) = 2.0e-7;

  F += H;

  Tensor<double> const d =
      compute_unrotated_rate_of_deformation_increment(F, F_prev);
  Tensor<double> const expected = 0.5 * (H + minitensor::transpose(H));

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_NEAR(d(i, j), expected(i, j), 1.0e-13);
    }
  }
}

TEST(hypo_kinematics, rigid_rotation_produces_zero_increment)
{
  Tensor<double> const F_prev = minitensor::eye<double>(3);
  Tensor<double> const F = rotation_z(0.25);

  Tensor<double> const d =
      compute_unrotated_rate_of_deformation_increment(F, F_prev);

  expect_tensor_near_zero(d, 1.0e-13);
}

TEST(hypo_kinematics, diagonal_stretch_uses_midpoint_deviatoric_part)
{
  Tensor<double> F = minitensor::eye<double>(3);
  Tensor<double> const F_prev = minitensor::eye<double>(3);
  F(0, 0) = 1.4;
  F(1, 1) = 0.8;
  F(2, 2) = 1.1;

  Tensor<double> const d =
      compute_unrotated_rate_of_deformation_increment(F, F_prev);

  double midpoint[3];
  for (int i = 0; i < 3; ++i) {
    midpoint[i] = (F(i, i) - 1.) / (0.5 * (F(i, i) + 1.));
  }

  double const midpoint_trace =
      midpoint[0] + midpoint[1] + midpoint[2];
  double const volumetric = std::log(minitensor::det(F));

  for (int i = 0; i < 3; ++i) {
    double const expected =
        midpoint[i] - midpoint_trace / 3. + volumetric / 3.;
    EXPECT_NEAR(d(i, i), expected, 1.0e-14);
  }

  EXPECT_NEAR(d(0, 1), 0., 1.0e-14);
  EXPECT_NEAR(d(0, 2), 0., 1.0e-14);
  EXPECT_NEAR(d(1, 2), 0., 1.0e-14);
}
