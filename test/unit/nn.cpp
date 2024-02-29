#include <NN.hpp>
#include <NN2.hpp>

#include <gtest/gtest.h>

using namespace ML;

static void check_components_wrt_params(
    const char* activation,
    std::vector<int> const& topology,
    Vector const& x)
{
  assert(x.size() == 1);
  assert(topology[0] == 1);
  assert(topology.back() == 1);
  NN network(activation, topology);
  Vector const p = network.get_params();
  Vector const y = network.feed_forward(x);
  auto const D = network.differentiate(y);
  double const h = 1.e-8;
  for (int i = 0; i < p.size(); ++i) {
    Vector p_perturb = p;
    p_perturb[i] += h;
    network.set_params(p_perturb);
    Vector const y_perturb = network.feed_forward(x);
    Vector const dy_dp_fd = (y_perturb - y)/h;
    printf("dy[0]/dp[%d], FD: %.15e, backprop: %.15e\n", i, dy_dp_fd[0], D[0](0,i));
  }
  printf("\n");
}

static void check_components_wrt_inputs(
    const char* activation,
    std::vector<int> const& topology,
    Vector const& x)
{
  assert(x.size() == 1);
  assert(topology[0] == 1);
  assert(topology.back() == 1);
  NN network(activation, topology);
  Vector const p = network.get_params();
  Vector const y = network.feed_forward(x);
  auto const D = network.differentiate(y);
  double const h = 1.e-6;
  for (int i = 0; i < x.size(); ++i) {
    Vector x_perturb = x;
    x_perturb[i] += h;
    Vector const y_perturb = network.feed_forward(x_perturb);
    Vector const dy_dx_fd = (y_perturb - y)/h;
    printf("dy[0]/dx[%d], FD: %.15e, backprop: %.15e\n", i, dy_dx_fd[0], D[1](0,i));
  }
  printf("\n");
}

static void check_direction(
    const char* activation,
    std::vector<int> const& topology,
    Vector const& x)
{
  assert(topology[0] == 1);
  assert(topology.back() == 1);
  assert(x.size() == 1);
  NN network(activation, topology);
  Vector const p = network.get_params();
  Vector const y = network.feed_forward(x);
  auto const D = network.differentiate(y);
  Vector dir = Vector(p.size());
  dir.setOnes();
  Vector const dy_dir = D[0] * dir;
  for (int i = 0; i >= -14; --i) {
    double const h = std::pow(10., double(i));
    Vector const p_perturb = p + dir * h;
    network.set_params(p_perturb);
    Vector const y_perturb = network.feed_forward(x);
    Vector const dy_dir_fd = (y_perturb - y) / h;
    double const err = std::abs(dy_dir[0] - dy_dir_fd[0]);
    printf("[%.2e] %.15e\n", h, err);
  }
  printf("\n");
}

TEST(NN, components_wrt_params)
{
  Vector x(1);
  x[0] = 1.;
  check_components_wrt_params("relu", {1, 4, 4, 1}, x);
  check_components_wrt_params("sigmoid", {1, 2, 3, 1}, x);
  check_components_wrt_params("tanh", {1, 2, 1}, x);
}

TEST(NN, components_wrt_inputs)
{
  Vector x(1);
  x[0] = 1.23;
  check_components_wrt_inputs("relu", {1, 5, 1}, x);
  check_components_wrt_inputs("sigmoid", {1, 3, 2, 1}, x);
  check_components_wrt_inputs("tanh", {1, 3, 1}, x);
}

TEST(NN, check_direction)
{
  Vector x(1);
  x[0] = 1.;
  check_direction("relu", {1, 4, 4, 1}, x);
  check_direction("sigmoid", {1, 2, 3, 1}, x);
  check_direction("tanh", {1, 10, 1}, x);
}

TEST(NN, ADrewrite)
{
  using namespace ML2;
  FFNN<DFADT>::Vector x(1);
  x[0] = 1;
  FFNN<DFADT> nn("relu", {1, 4, 4, 1});
  auto y = nn.evaluate(x);
  std::cout << y << "\n";
}
