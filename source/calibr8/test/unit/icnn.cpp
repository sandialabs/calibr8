#include <ICNN.hpp>

#include <gtest/gtest.h>

using namespace ML;

TEST(input_convex_nn, dummy)
{
  const char* activation = "softplus";
  std::vector<int> topology = {1,3,1};
  /* TODO: segfault on construction
     almost guaranteed due to indexing in setting up sizes */
  //FICNN<RFAD_SFADT> network(activation, topology);

  using NETWORK = FICNN<double>;
  NETWORK nn(activation, topology);

  NETWORK::Vector y(1);
  y(0) = 0.0;
  nn.evaluate(y);

}
