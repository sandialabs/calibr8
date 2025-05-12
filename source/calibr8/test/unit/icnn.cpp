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

TEST(input_convex_nn, scalar)
{
  using NETWORK = FICNN<RFAD_DFADT>;
  const char* activation = "softplus";
  std::vector<int> topology = {1,3,1};
  NETWORK nn(activation, topology);
  DFADT y_fad;
  y_fad = 1.0;
  y_fad.diff(0, 1);
  Sacado::Rad::ADvar<DFADT> y_rad = y_fad;
  NETWORK::Vector y(1);
  y(0) = y_rad;

  /* segfaults */
  //nn.evaluate(y);
}
