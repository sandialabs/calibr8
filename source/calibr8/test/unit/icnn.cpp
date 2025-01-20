#include <ICNN.hpp>

#include <gtest/gtest.h>

using namespace ML;

TEST(input_convex_nn, dummy)
{
  const char* activation = "softplus";
  std::vector<int> topology = {1,3,1};
  FICNN<RFAD_SFADT> network(activation, topology);
}
