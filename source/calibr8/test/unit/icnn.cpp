#include <cmath>

#include <ICNN.hpp>

#include <gtest/gtest.h>

using namespace ML;

TEST(input_convex_nn, does_rfad_dfad_work)
{
  DFADT yvar;
  yvar = 1.0;
  yvar.diff(0, 1);
  Sacado::Rad::ADvar<DFADT> y = yvar;
  auto const f = y*y*y;
  Sacado::Rad::ADvar<DFADT>::Gradcomp();
  auto const df = y.adj();
  EXPECT_EQ(df.val(), 3);
  EXPECT_EQ(df.fastAccessDx(0), 6);
}

TEST(input_convex_nn, ficnn_input_gradient)
{
  char const* activation = "softplus";
  std::vector<int> const topology = {6, 4, 1};
  int const n = topology.front();

  FICNN<double> nn_double(activation, topology);
  Eigen::Matrix<double, Eigen::Dynamic, 1> const params = nn_double.get_params();
  int const np = params.size();
  Eigen::Matrix<double, Eigen::Dynamic, 1> x(n);
  x(0) = 0.3; x(1) = -0.5; x(2) = 0.7; x(3) = 0.2; x(4) = -0.1; x(5) = 0.4;

  FICNN<RAD_SFADT> nn(activation, topology);
  Eigen::Matrix<RAD_SFADT, Eigen::Dynamic, 1> rad_params(np);
  for (int p = 0; p < np; ++p) {
    rad_params(p) = RAD_SFADT(params(p));
  }
  nn.set_params(rad_params);
  Eigen::Matrix<RAD_SFADT, Eigen::Dynamic, 1> y(n);
  for (int i = 0; i < n; ++i) {
    SFADT xi = x(i);
    xi.diff(i, n);
    y(i) = RAD_SFADT(xi);
  }
  RAD_SFADT const phi = nn.evaluate(y)(0);
  Sacado::Rad::ADvar<SFADT>::Gradcomp();

  auto normal_at = [&](Eigen::Matrix<double, Eigen::Dynamic, 1> const& xv) {
    FICNN<SFADT> nnf(activation, topology);
    Eigen::Matrix<SFADT, Eigen::Dynamic, 1> wf(np);
    for (int p = 0; p < np; ++p) wf(p) = SFADT(params(p));
    nnf.set_params(wf);
    Eigen::Matrix<SFADT, Eigen::Dynamic, 1> yf(n);
    for (int i = 0; i < n; ++i) {
      SFADT xi = xv(i);
      xi.diff(i, n);
      yf(i) = xi;
    }
    SFADT const phif = nnf.evaluate(yf)(0);
    Eigen::Matrix<double, Eigen::Dynamic, 1> g(n);
    for (int i = 0; i < n; ++i) g(i) = phif.dx(i);
    return g;
  };

  Eigen::Matrix<double, Eigen::Dynamic, 1> const g0 = normal_at(x);
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(y(i).adj().val(), g0(i), 1.e-9);
  }
  double const h = 1.e-5;
  for (int j = 0; j < n; ++j) {
    Eigen::Matrix<double, Eigen::Dynamic, 1> xp = x, xm = x;
    xp(j) += h; xm(j) -= h;
    Eigen::Matrix<double, Eigen::Dynamic, 1> const gp = normal_at(xp);
    Eigen::Matrix<double, Eigen::Dynamic, 1> const gm = normal_at(xm);
    for (int i = 0; i < n; ++i) {
      double const fd = (gp(i) - gm(i)) / (2. * h);
      EXPECT_NEAR(y(i).adj().dx(j), fd, 1.e-4 * std::abs(fd) + 1.e-6);
    }
  }
}

TEST(input_convex_nn, ficnn_param_gradient)
{
  char const* activation = "softplus";
  std::vector<int> const topology = {6, 4, 1};
  int const n = topology.front();

  FICNN<double> nn_double(activation, topology);
  Eigen::Matrix<double, Eigen::Dynamic, 1> const params = nn_double.get_params();
  int const np = params.size();
  Eigen::Matrix<double, Eigen::Dynamic, 1> x(n);
  x(0) = 0.3; x(1) = -0.5; x(2) = 0.7; x(3) = 0.2; x(4) = -0.1; x(5) = 0.4;

  FICNN<RAD_DFADT> nn(activation, topology);
  Eigen::Matrix<RAD_DFADT, Eigen::Dynamic, 1> rad_params(np);
  for (int p = 0; p < np; ++p) {
    DFADT wp = params(p);
    wp.diff(p, np);
    rad_params(p) = RAD_DFADT(wp);
  }
  nn.set_params(rad_params);
  Eigen::Matrix<RAD_DFADT, Eigen::Dynamic, 1> y(n);
  for (int i = 0; i < n; ++i) {
    y(i) = RAD_DFADT(x(i));
  }
  RAD_DFADT const phi = nn.evaluate(y)(0);
  Sacado::Rad::ADvar<DFADT>::Gradcomp();

  auto normal_at = [&](Eigen::Matrix<double, Eigen::Dynamic, 1> const& w) {
    FICNN<SFADT> nnf(activation, topology);
    Eigen::Matrix<SFADT, Eigen::Dynamic, 1> wf(np);
    for (int p = 0; p < np; ++p) wf(p) = SFADT(w(p));
    nnf.set_params(wf);
    Eigen::Matrix<SFADT, Eigen::Dynamic, 1> yf(n);
    for (int i = 0; i < n; ++i) {
      SFADT xi = x(i);
      xi.diff(i, n);
      yf(i) = xi;
    }
    SFADT const phif = nnf.evaluate(yf)(0);
    Eigen::Matrix<double, Eigen::Dynamic, 1> g(n);
    for (int i = 0; i < n; ++i) g(i) = phif.dx(i);
    return g;
  };

  double const h = 1.e-5;
  for (int p = 0; p < np; ++p) {
    Eigen::Matrix<double, Eigen::Dynamic, 1> w_plus = params, w_minus = params;
    w_plus(p) += h; w_minus(p) -= h;
    Eigen::Matrix<double, Eigen::Dynamic, 1> const gp = normal_at(w_plus);
    Eigen::Matrix<double, Eigen::Dynamic, 1> const gm = normal_at(w_minus);
    for (int i = 0; i < n; ++i) {
      double const fd = (gp(i) - gm(i)) / (2. * h);
      EXPECT_NEAR(y(i).adj().dx(p), fd, 1.e-4 * std::abs(fd) + 1.e-6);
    }
  }
}

TEST(input_convex_nn, does_eigen_with_rfad_work)
{
  Eigen::Matrix<RAD_DFADT, Eigen::Dynamic, 1> y(1);
  Sacado::Rad::ADvar<DFADT> y_var = DFADT(1, 0, 1.23);
  y(0) = y_var;
}

TEST(input_convex_nn, scalar)
{
  using NETWORK = FICNN<double>;
  char const* activation = "softplus";
  std::vector<int> topology = {1, 3, 1};
  NETWORK nn(activation, topology);
  NETWORK::Vector y(1);
  y(0) = 2.0;
  nn.evaluate(y);
}

TEST(input_convex_nn, rfad_dfadt)
{
  using NETWORK = FICNN<RAD_DFADT>;
  char const* activation = "softplus";
  std::vector<int> topology = {1, 3, 1};
  NETWORK nn(activation, topology);
  DFADT y_fad;
  y_fad = 1.0;
  y_fad.diff(0, 1);
  NETWORK::Vector y(1);
  y(0) = RAD_DFADT(y_fad);
  nn.evaluate(y);
}
