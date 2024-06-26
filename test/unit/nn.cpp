#include <NN.hpp>

#include <gtest/gtest.h>

using namespace ML;

static void check_components_wrt_params(
    const char* activation,
    std::vector<int> const& topology,
    FFNN<DFADT>::Vector const& x)
{
  using Vector = FFNN<DFADT>::Vector;
  assert(x.size() == 1);
  assert(topology[0] == 1);
  assert(topology.back() == 1);
  FFNN<DFADT> network(activation, topology);
  Vector const p = network.get_params();
  Vector const y = network.evaluate(x);
  DFADT const h = 1.e-8;
  for (int i = 0; i < p.size(); ++i) {
    Vector p_perturb = p;
    p_perturb[i] += h;
    network.set_params(p_perturb);
    Vector const y_perturb = network.evaluate(x);
    Vector const dy_dp_fd = (y_perturb - y)/h;
    printf("dy[0]/dp[%d], FD: %.15e, AD: %.15e\n",
        i,
        dy_dp_fd[0].val(),
        y[0].fastAccessDx(i));
  }
  printf("\n");
}

static void check_components_wrt_inputs(
    const char* activation,
    std::vector<int> const& topology,
    FFNN<SFADT>::Vector const& x)
{
  using Vector = FFNN<SFADT>::Vector;
  assert(x.size() == 1);
  assert(topology[0] == 1);
  assert(topology.back() == 1);
  FFNN<SFADT> network(activation, topology);
  Vector const p = network.get_params();
  Vector const y = network.evaluate(x);
  SFADT const h = 1.e-6;
  Vector x_perturb = x;
  x_perturb[0] += h;
  Vector const y_perturb = network.evaluate(x_perturb);
  Vector const dy_dx_fd = (y_perturb - y)/h;
  printf("dy[0]/dx, FD: %.15e, AD: %.15e\n",
      dy_dx_fd[0].val(),
      y[0].fastAccessDx(0));
}

static void check_direction(
    const char* activation,
    std::vector<int> const& topology,
    FFNN<DFADT>::Vector const& x)
{
  using Vector = FFNN<DFADT>::Vector;
  assert(topology[0] == 1);
  assert(topology.back() == 1);
  assert(x.size() == 1);
  FFNN<DFADT> network(activation, topology);
  Vector const p = network.get_params();
  Vector const y = network.evaluate(x);
  Vector dir = Vector(p.size());
  dir.setOnes();
  double dy_dir = 0.;
  for (int i = 0; i < p.size(); ++i) {
    dy_dir += y[0].fastAccessDx(i) * dir[i].val();
  }
  for (int i = 0; i >= -14; --i) {
    DFADT const h = std::pow(10., double(i));
    Vector const p_perturb = p + dir * h;
    network.set_params(p_perturb);
    Vector const y_perturb = network.evaluate(x);
    Vector const dy_dir_fd = (y_perturb - y) / h;
    double const err = std::abs(dy_dir - dy_dir_fd[0].val());
    printf("[%.2e] %.15e\n", h.val(), err);
  }
}

TEST(FFNN, components_wrt_params)
{
  FFNN<DFADT>::Vector x(1);
  x[0] = 1.;
  check_components_wrt_params("relu", {1,4,4,1}, x);
  check_components_wrt_params("sigmoid", {1,2,3,1}, x);
  check_components_wrt_params("tanh", {1,2,1}, x);
}

TEST(FFNN, components_wrt_inputs)
{
  FFNN<SFADT>::Vector x(1);
  x[0] = 1.23;
  x[0].diff(0, 1);
  check_components_wrt_inputs("relu", {1,5,1}, x);
  check_components_wrt_inputs("sigmoid", {1,3,2,1}, x);
  check_components_wrt_inputs("tanh", {1,3,1}, x);
}

TEST(FFNN, check_direction)
{
  FFNN<DFADT>::Vector x(1);
  x[0] = 1.0;
  check_direction("relu", {1,4,4,1}, x);
}
